from typing import Dict

import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, InputVariationalDropout
from allennlp.nn.util import (
    get_text_field_mask,
    masked_softmax,
    weighted_sum,
    masked_max,
)
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.matrix_attention.dot_product_matrix_attention import DotProductMatrixAttention
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.optimizers import DenseSparseAdam
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import LstmSeq2SeqEncoder


class ESIM(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        token_embedder: TokenEmbedder,
        num_labels: int
    ) -> None:
        super().__init__(vocab)

        self.word_embedders = BasicTextFieldEmbedder(
            {"tokens": token_embedder})
        self._encoder = LstmSeq2SeqEncoder(300, 300, 2)
        # self._encoder = PytorchTransformer(300, 3, 300, 4)

        self._matrix_attention = DotProductMatrixAttention()
        self._projection_feedforward = FeedForward(300 * 4, 1, 300, torch.nn.ReLU(), 0.2)

        self._inference_encoder = LstmSeq2SeqEncoder(300, 300, 2)
        # self._inference_encoder = PytorchTransformer(300, 3, 300, 4)

        self.dropout = torch.nn.Dropout(0.3)
        self.rnn_input_dropout = InputVariationalDropout(0.3)

        self._output_feedforward = FeedForward(1200, 1, 300, torch.nn.ReLU(), 0.2)
        self._output_logit = FeedForward(300, 1, 3, lambda x: x)

        self._num_labels = num_labels

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(  # type: ignore
        self,
        sent1: TextFieldTensors,
        sent2: TextFieldTensors,
        label: torch.IntTensor = None,
    ) -> Dict[str, torch.Tensor]:
        embedded_sent1 = self.word_embedders(sent1)
        embedded_sent2 = self.word_embedders(sent2)
        sent1_mask = get_text_field_mask(sent1)
        sent2_mask = get_text_field_mask(sent2)

        # apply dropout for LSTM
        if self.rnn_input_dropout:
            embedded_sent1 = self.rnn_input_dropout(embedded_sent1)
            embedded_sent2 = self.rnn_input_dropout(embedded_sent2)

        # encode sent1 and sent2
        encoded_sent1 = self._encoder(embedded_sent1, sent1_mask)
        encoded_sent2 = self._encoder(embedded_sent2, sent2_mask)

        # Shape: (batch_size, sent1_length, sent2_length)
        similarity_matrix = self._matrix_attention(encoded_sent1, encoded_sent2)

        # Shape: (batch_size, sent1_length, sent2_length)
        p2h_attention = masked_softmax(similarity_matrix, sent2_mask)
        # Shape: (batch_size, sent1_length, embedding_dim)
        attended_sent2 = weighted_sum(encoded_sent2, p2h_attention)

        # Shape: (batch_size, sent2_length, sent1_length)
        h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), sent1_mask)
        # Shape: (batch_size, sent2_length, embedding_dim)
        attended_sent1 = weighted_sum(encoded_sent1, h2p_attention)

        # the "enhancement" layer
        sent1_enhanced = torch.cat(
            [
                encoded_sent1,
                attended_sent2,
                encoded_sent1 - attended_sent2,
                encoded_sent1 * attended_sent2,
            ],
            dim=-1,
        )
        sent2_enhanced = torch.cat(
            [
                encoded_sent2,
                attended_sent1,
                encoded_sent2 - attended_sent1,
                encoded_sent2 * attended_sent1,
            ],
            dim=-1,
        )

        # The projection layer down to the model dimension.  Dropout is not applied before
        # projection.
        projected_enhanced_sent1 = self._projection_feedforward(sent1_enhanced)
        projected_enhanced_sent2 = self._projection_feedforward(sent2_enhanced)

        # Run the inference layer
        if self.rnn_input_dropout:
            projected_enhanced_sent1 = self.rnn_input_dropout(projected_enhanced_sent1)
            projected_enhanced_sent2 = self.rnn_input_dropout(projected_enhanced_sent2)
        v_ai = self._inference_encoder(projected_enhanced_sent1, sent1_mask)
        v_bi = self._inference_encoder(projected_enhanced_sent2, sent2_mask)

        # The pooling layer -- max and avg pooling.
        # (batch_size, model_dim)
        v_a_max = masked_max(v_ai, sent1_mask.unsqueeze(-1), dim=1)
        v_b_max = masked_max(v_bi, sent2_mask.unsqueeze(-1), dim=1)

        v_a_avg = torch.sum(v_ai * sent1_mask.unsqueeze(-1), dim=1) / torch.sum(
            sent1_mask, 1, keepdim=True
        )
        v_b_avg = torch.sum(v_bi * sent2_mask.unsqueeze(-1), dim=1) / torch.sum(
            sent2_mask, 1, keepdim=True
        )

        # Now concat
        # (batch_size, model_dim * 2 * 4)
        v_all = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        # the final MLP -- apply dropout to input, and MLP applies to output & hidden
        if self.dropout:
            v_all = self.dropout(v_all)

        output_hidden = self._output_feedforward(v_all)
        label_logits = self._output_logit(output_hidden)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"logits": label_logits, "probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self._accuracy.get_metric(reset)}

    def get_optimizer(self):
        return DenseSparseAdam(self.named_parameters(), lr=5e-4)