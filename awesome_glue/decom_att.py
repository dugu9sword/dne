from typing import Dict, Optional, List, Any

import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TimeDistributed
from allennlp.modules.matrix_attention import DotProductMatrixAttention
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.optimizers import DenseSparseAdam, AdagradOptimizer, AdadeltaOptimizer
from allennlpx.training import adv_utils


class DecomposableAttention(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        token_embedder: TokenEmbedder,
        num_labels: int
    ) -> None:
        super().__init__(vocab)

        self._text_field_embedder = BasicTextFieldEmbedder(
            {"tokens": token_embedder})
        dim = token_embedder.get_output_dim()
        self._attend_feedforward = TimeDistributed(FeedForward(dim, 1, 100, torch.nn.ReLU(), 0.2))
        self._matrix_attention = DotProductMatrixAttention()
        self._compare_feedforward = TimeDistributed(FeedForward(dim * 2, 1, 100, torch.nn.ReLU(), 0.2))
        self._aggregate_feedforward = FeedForward(200, 1, num_labels, torch.nn.ReLU(), 0.2)

        self._num_labels = num_labels

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(  # type: ignore
        self,
        sent1: TextFieldTensors,
        sent2: TextFieldTensors,
        label: torch.IntTensor = None,
    ) -> Dict[str, torch.Tensor]:
        with adv_utils.forward_context("sent1"):
            embedded_sent1 = self._text_field_embedder(sent1)
        with adv_utils.forward_context("sent2"):
            embedded_sent2 = self._text_field_embedder(sent2)
        sent1_mask = get_text_field_mask(sent1)
        sent2_mask = get_text_field_mask(sent2)

        projected_sent1 = self._attend_feedforward(embedded_sent1)
        projected_sent2 = self._attend_feedforward(embedded_sent2)
        # Shape: (batch_size, sent1_length, sent2_length)
        similarity_matrix = self._matrix_attention(projected_sent1, projected_sent2)

        # Shape: (batch_size, sent1_length, sent2_length)
        p2h_attention = masked_softmax(similarity_matrix, sent2_mask)
        # Shape: (batch_size, sent1_length, embedding_dim)
        attended_sent2 = weighted_sum(embedded_sent2, p2h_attention)

        # Shape: (batch_size, sent2_length, sent1_length)
        h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), sent1_mask)
        # Shape: (batch_size, sent2_length, embedding_dim)
        attended_sent1 = weighted_sum(embedded_sent1, h2p_attention)

        sent1_compare_input = torch.cat([embedded_sent1, attended_sent2], dim=-1)
        sent2_compare_input = torch.cat([embedded_sent2, attended_sent1], dim=-1)

        compared_sent1 = self._compare_feedforward(sent1_compare_input)
        compared_sent1 = compared_sent1 * sent1_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_sent1 = compared_sent1.sum(dim=1)

        compared_sent2 = self._compare_feedforward(sent2_compare_input)
        compared_sent2 = compared_sent2 * sent2_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_sent2 = compared_sent2.sum(dim=1)

        aggregate_input = torch.cat([compared_sent1, compared_sent2], dim=-1)
        label_logits = self._aggregate_feedforward(aggregate_input)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {
            "logits": label_logits,
            "probs": label_probs,
        }

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self._accuracy.get_metric(reset)}

    def get_optimizer(self):
        return DenseSparseAdam(self.named_parameters(), lr=5e-4)
        # return AdadeltaOptimizer(self.named_parameters())