import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from allennlp.models import Model
from allennlp.modules.text_field_embedders import (BasicTextFieldEmbedder,
                                                   TextFieldEmbedder)
from allennlp.modules.token_embedders import Embedding, TokenEmbedder
from allennlp.modules.token_embedders.elmo_token_embedder import \
    ElmoTokenEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.optimizers import DenseSparseAdam

from allennlpx.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import \
    PytorchSeq2VecWrapper
from allennlpx.modules.token_embedders.embedding import \
    _read_pretrained_embeddings_file
from awesome_glue.utils import EMBED_DIM, WORD2VECS
from luna import (LabelSmoothingLoss, auto_create, flt2str, log, log_config,
                  ram_globalize, ram_read, ram_reset, ram_write)

ELMO_OPTION = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
ELMO_WEIGHT = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'


class LstmClassifier(Model):
    def __init__(self, 
                 vocab, 
                 num_labels, 
                 pretrain, 
                 finetunable=False, 
                 cache_embed_path=None):
        super().__init__(vocab)
        if pretrain in WORD2VECS:
            embedding_path = WORD2VECS[pretrain]
#             cache_embed_path = hashlib.md5(embedding_path.encode()).hexdigest()
            weight = auto_create(
                cache_embed_path,
                lambda: _read_pretrained_embeddings_file(embedding_path,
                                                         embedding_dim=EMBED_DIM[pretrain],
                                                         vocab=vocab,
                                                         namespace="tokens"), True)
            token_embedder = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                       embedding_dim=EMBED_DIM[pretrain],
                                       weight=weight,
                                       sparse=True,
                                       trainable=finetunable)
        elif pretrain == 'elmo':
            token_embedder = ElmoTokenEmbedder(ELMO_OPTION, ELMO_WEIGHT, requires_grad=finetunable)
        elif pretrain == 'random':
            token_embedder = Embedding(
                num_embeddings=vocab.get_vocab_size('tokens'),
                embedding_dim=EMBED_DIM[pretrain],
                #    sparse=True,
                trainable=finetunable)

        self.word_embedders = BasicTextFieldEmbedder({"tokens": token_embedder},
                                                     allow_unmatched_keys=True)

        self.encoder = PytorchSeq2VecWrapper(
            torch.nn.LSTM(EMBED_DIM[pretrain], 
                          hidden_size=300, 
                          num_layers=2, 
                          batch_first=True))
        self.linear = torch.nn.Linear(in_features=self.encoder.get_output_dim(),
                                      out_features=num_labels)

        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()
#         self.loss_function = LabelSmoothingLoss(0.1)

    def get_optimizer(self):
        return DenseSparseAdam(self.parameters(), lr=1e-3)

    def forward(self, sent, label=None):
        mask = get_text_field_mask(sent)
        embeddings = self.word_embedders(sent)
        # if self.training \
        #     and ram_read("config").pretrain!='bert':
        #     embeddings = noise(embeddings, ram_read("config").embed_noise)

        encoder_out = self.encoder(embeddings, mask)

        logits = self.linear(encoder_out)
        output = {"logits": logits, "probs": F.softmax(logits, dim=1)}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}
