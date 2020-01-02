# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders.embedding import  _read_pretrained_embeddings_file
from allennlp.nn.util import get_text_field_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.regularizers.regularizer_applicator import RegularizerApplicator

from luna import auto_create
from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from collections import defaultdict

from sst.args import ProgramArgs

# from allennlpx.interpret.attackers.embedding_searcher import EmbeddingSearcher
# from luna import load_word2vec
# emb_searcher = EmbeddingSearcher(token_embedding.weight,
#                                  word2idx=vocab.get_token_index,
#                                  idx2word=vocab.get_token_from_index)

# cf_embedding = torch.nn.Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
#                                   embedding_dim=300)
# load_word2vec(cf_embedding, vocab._token_to_index["tokens"],
#               "../counter-fitting/results/counter_fitted_vectors.txt")
# cf_searcher = EmbeddingSearcher(cf_embedding.weight,
#                                 word2idx=vocab.get_token_index,
#                                 idx2word=vocab.get_token_from_index)

# emb_searcher.find_neighbours("happy", "euc", topk=20, verbose=True)

WORD2VECS = {
    "fasttext": "/disks/sdb/zjiehang/embeddings/fasttext/crawl-300d-2M.vec",
    "sgns": "/disks/sdb/zjiehang/frequency/pretrained_embedding/word2vec/GoogleNews-vectors-negative300.txt",
    "glove": "/disks/sdb/zjiehang/embeddings/glove/glove.42B.300d.txt",
    "fasttext_ol": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
}

ELMO_OPTION = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
ELMO_WEIGHT = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

EMBED_DIM = defaultdict(lambda: 300, {"elmo": 256})

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


class LstmClassifier(Model):
    def __init__(self, vocab: Vocabulary, config: ProgramArgs, regularizer: RegularizerApplicator = None):
        super().__init__(vocab, regularizer)

        self.config = config
        
        if config.pretrain in WORD2VECS:
            embedding_path = WORD2VECS[config.pretrain]
            cache_embed_path = hashlib.md5(embedding_path.encode()).hexdigest()
            weight = auto_create(
                cache_embed_path, lambda: _read_pretrained_embeddings_file(
                    embedding_path, embedding_dim=EMBED_DIM[config.pretrain], vocab=vocab, namespace="tokens"), True, config.cache_path)
            token_embedder = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                       embedding_dim=EMBED_DIM[config.pretrain],
                                       weight=weight,
                                       sparse=True,
                                       trainable=config.is_embedding_trainable)
        elif config.pretrain == 'elmo':
            token_embedder = ElmoTokenEmbedder(ELMO_OPTION, ELMO_WEIGHT)
        else:
            token_embedder = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                       embedding_dim=EMBED_DIM[config.pretrain])

        self.word_embedders = BasicTextFieldEmbedder({"tokens": token_embedder})

        self.encoder = PytorchSeq2VecWrapper(
            torch.nn.LSTM(EMBED_DIM[config.pretrain], hidden_size=config.hidden_dim, num_layers=config.num_layers, batch_first=True))

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.encoder.get_output_dim(),
                            out_features=vocab.get_vocab_size('label')))
        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()
        
        self.word_dropout = EmbeddingDropout(config.dropout_rate)

    def forward(self, tokens, label=None):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embedders(tokens)
        if self.training:
            embeddings = self.noise(embeddings, self.config.embed_noise)
        embeddings = self.word_dropout(embeddings)
        encoder_out = self.encoder(embeddings, mask)
        if self.training:
            encoder_out = self.noise(encoder_out, self.config.lstm_noise)
        logits = self.linear(encoder_out)
        #         print(encoder_out.size(), logits.size())
        output = {"logits": logits, "probs": F.softmax(logits, dim=1)}
        #         print(output)
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}
    
    def noise(self, tsr: torch.Tensor, scale=1.0):
        return tsr + torch.normal(0., tsr.std().item() * scale, tsr.size()).to(tsr.device)