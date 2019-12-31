# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'

import hashlib
import logging
import pathlib
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BasicIterator, BucketIterator
from allennlp.data.iterators.multiprocess_iterator import MultiprocessIterator
from allennlp.data.token_indexers.single_id_token_indexer import \
    SingleIdTokenIndexer
from allennlp.data.token_indexers.token_characters_indexer import \
    TokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import (PytorchSeq2VecWrapper, Seq2VecEncoder)
from allennlp.modules.text_field_embedders import (BasicTextFieldEmbedder, TextFieldEmbedder)
from allennlp.modules.token_embedders import Embedding, TokenEmbedder
from allennlp.modules.token_embedders.embedding import \
    _read_pretrained_embeddings_file
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.optimizers import DenseSparseAdam
from tabulate import tabulate
from torch.nn.utils.rnn import pad_packed_sequence
from tqdm import tqdm

from allennlpx import allenutil
from allennlpx.interpret.attackers.attacker import DEFAULT_IGNORE_TOKENS
from allennlpx.interpret.attackers.bruteforce import BruteForce
from allennlpx.interpret.attackers.hotflip import HotFlip
from allennlpx.interpret.attackers.pgd import PGD
from allennlpx.predictors.text_classifier import TextClassifierPredictor
# from allennlp.training.trainer import Trainer
from allennlpx.training.callback_trainer import CallbackTrainer
from allennlpx.training.callbacks.evaluate_callback import EvaluateCallback
from luna import (auto_create, flt2str, log, log_config, ram_read, ram_reset, ram_write)
from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from collections import defaultdict

from awesome_sst.freq_util import analyze_frequency, frequency_analysis

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
    "sgns":
    "/disks/sdb/zjiehang/frequency/pretrained_embedding/word2vec/GoogleNews-vectors-negative300.txt",
    "glove": "/disks/sdb/zjiehang/embeddings/glove/glove.42B.300d.txt",
    "fasttext_ol": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
}

ELMO_OPTION = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
ELMO_WEIGHT = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

EMBED_DIM = defaultdict(lambda: 300, {"elmo": 256})


class LstmClassifier(Model):
    def __init__(self, vocab, pretrain="fasttext_ol", fix_embed=False):
        super().__init__(vocab)

        if pretrain in WORD2VECS:
            embedding_path = WORD2VECS[pretrain]
            cache_embed_path = hashlib.md5(embedding_path.encode()).hexdigest()
            weight = auto_create(
                cache_embed_path,
                lambda: _read_pretrained_embeddings_file(embedding_path,
                                                         embedding_dim=EMBED_DIM[pretrain],
                                                         vocab=vocab,
                                                         namespace="tokens"), True)
            token_embedder = Embedding(
                num_embeddings=vocab.get_vocab_size('tokens'),
                embedding_dim=EMBED_DIM[pretrain],
                weight=weight,
                #    scale_grad_by_freq=True,
                sparse=True,
                trainable=fix_embed)
        elif pretrain == 'elmo':
            token_embedder = ElmoTokenEmbedder(ELMO_OPTION, ELMO_WEIGHT)
        else:
            token_embedder = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                       embedding_dim=EMBED_DIM[pretrain])

        self.word_embedders = BasicTextFieldEmbedder({"tokens": token_embedder})

        self.encoder = PytorchSeq2VecWrapper(
            torch.nn.LSTM(EMBED_DIM[pretrain], hidden_size=512, num_layers=2, batch_first=True))

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.encoder.get_output_dim(),
                            out_features=vocab.get_vocab_size('label')))
        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, label=None):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embedders(tokens)
        if self.training:
            embeddings = noise(embeddings, ram_read("config").embed_noise)
        encoder_out = self.encoder(embeddings, mask)
        if self.training:
            encoder_out = noise(encoder_out, ram_read("config").lstm_noise)
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


def noise(tsr: torch.Tensor, scale=1.0):
    return tsr + torch.normal(0., tsr.std().item() * scale, tsr.size()).to(tsr.device)
