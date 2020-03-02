from typing import Dict

import torch
import torch.nn as nn
from torch.nn import functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy

class EmbeddingDropout(nn.Module):
    def __init__(self, p=0.5):
        super(EmbeddingDropout, self).__init__()
        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, x, eps=1e-12):
        if self.training:
            x_mask = torch.bernoulli(x.new_full(x.shape[:2], 1 - self.p))
            x *= x_mask.unsqueeze(dim=-1)
        return x


@Model.register('rnn_classifier')
class RnnClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 seq2vec_encoder: Seq2VecEncoder,
                 dropout: float = 0.,
                 label_namespace: str = 'labels',
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)

        self._text_field_embedder = text_field_embedder
        self._seq2vec_encoder = seq2vec_encoder
        self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = nn.Dropout(dropout)
            self.word_dropout = EmbeddingDropout(dropout)
        else:
            self._dropout = lambda x: x
            self.word_dropout = lambda x: x

        self._num_labels = vocab.get_vocab_size(namespace=label_namespace)
        self._classification_layer = nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                ensemble: bool = False) -> Dict[str, torch.Tensor]:
        '''
        ensemble : a bool variable to control the current batch is an ensemble of a sentence
        '''
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()
        embedded_text = self.word_dropout(embedded_text)
        encoded_text = self._dropout(self._seq2vec_encoder(embedded_text, mask=mask))

        logits = self._classification_layer(encoded_text)
        probs = F.softmax(logits, dim=1)

        output_dict = {'logits': logits, 'probs': probs}

        if label is not None:
            if not ensemble:
                loss = self._loss(logits, label.long().view(-1))
                output_dict['loss'] = loss
                self._accuracy(logits, label)
            else:
                ensemble_logit = torch.mean(logits, dim=0, keepdim=True)
                ensemble_probs = F.softmax(ensemble_logit, dim=1)
                ensemble_label = label.new_full((1,), label[0], dtype=label.dtype, device=label.device)
                output_dict = {'logits': ensemble_logit, 'probs': ensemble_probs}
                loss = self._loss(ensemble_logit, ensemble_label)
                output_dict['loss'] = loss
                self._accuracy(ensemble_logit, ensemble_label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}