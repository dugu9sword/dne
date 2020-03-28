import torch
import torch.nn.functional as F
from allennlp.models import Model
from allennlp.modules.text_field_embedders import (BasicTextFieldEmbedder,
                                                   TextFieldEmbedder)
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders.elmo_token_embedder import \
    ElmoTokenEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.optimizers import DenseSparseAdam

from allennlpx.modules.lstm import LSTM, LSTMCell
from allennlpx.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import \
    PytorchSeq2VecWrapper
from allennlpx.modules.token_embedders.embedding import \
    _read_pretrained_embeddings_file
from awesome_glue.utils import EMBED_DIM, WORD2VECS
from luna import (LabelSmoothingLoss, auto_create, flt2str, log, log_config,
                  ram_globalize, ram_read, ram_reset, ram_write)

class EmbeddingDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super(EmbeddingDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            x_mask = torch.bernoulli(x.new_full(x.shape[:2], 1 - self.p))
            x *= x_mask.unsqueeze(dim=-1)
        return x


class MagicLstmClassifier(Model):
    def __init__(self,
                 vocab,
                 num_labels,
                 pretrain,
                 word_drop_rate=0.3,
                 finetunable=True,
                 cache_embed_path=None):
        super().__init__(vocab)
        embedding_path = WORD2VECS[pretrain]
        weight = auto_create(
            cache_embed_path, lambda: _read_pretrained_embeddings_file(
                embedding_path,
                embedding_dim=EMBED_DIM[pretrain],
                vocab=vocab,
                namespace="tokens"), True)
        token_embedder = Embedding(
            num_embeddings=vocab.get_vocab_size('tokens'),
            embedding_dim=EMBED_DIM[pretrain],
            weight=weight,
            sparse=True,
            trainable=finetunable)

        self.word_embedders = BasicTextFieldEmbedder({"tokens": token_embedder})

        self.word_drop = EmbeddingDropout(word_drop_rate)

        self.encoder = LSTM(cell_class=LSTMCell,
                            input_size=EMBED_DIM[pretrain],
                            hidden_size=300,
                            dropout=0.0,
                            num_layers=1,
                            batch_first=True)
        self.linear = torch.nn.Linear(in_features=300,
                                      out_features=num_labels)

        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()


#         self.loss_function = LabelSmoothingLoss(0.1)

    def get_optimizer(self):
        return DenseSparseAdam(self.named_parameters(), lr=1e-3)

    def forward(self, sent, label=None):
        mask = get_text_field_mask(sent)
        embeddings = self.word_embedders(sent)
        embeddings = self.word_drop(embeddings)
        lengths = mask.sum(dim=1)
        outputs, (h, c) = self.encoder(embeddings, length=lengths)
        encoder_out = h[0]

        logits = self.linear(encoder_out)
        output = {"logits": logits, "probs": F.softmax(logits, dim=1)}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}
