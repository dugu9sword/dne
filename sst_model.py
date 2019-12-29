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
from freq_util import analyze_frequency, frequency_analysis
from luna import (auto_create, flt2str, log, log_config, ram_read, ram_reset, ram_write)

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


class LstmClassifier(Model):
    def __init__(self, vocab):
        super().__init__(vocab)
        if pathlib.Path("/disks/sdb/zjiehang").exists():
            print("Code running in china.")
            # embedding_path = "/disks/sdb/zjiehang/frequency/pretrained_embedding/word2vec/GoogleNews-vectors-negative300.txt"
            embedding_path = "/disks/sdb/zjiehang/embeddings/fasttext/crawl-300d-2M.vec"
            # embedding_path = "/disks/sdb/zjiehang/embeddings/gensim_sgns_gnews/model.txt"
            # embedding_path = "/disks/sdb/zjiehang/embeddings/glove/glove.42B.300d.txt"
        else:
            embedding_path = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"

        if embedding_path:
            cache_embed_path = hashlib.md5(embedding_path.encode()).hexdigest()
            weight = auto_create(
                cache_embed_path, lambda: _read_pretrained_embeddings_file(
                    embedding_path, embedding_dim=300, vocab=vocab, namespace="tokens"), True)
            token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                        embedding_dim=300,
                                        weight=weight,
                                        sparse=True,
                                        trainable=False)
        else:
            token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                        embedding_dim=300)

        self.word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

        self.encoder = PytorchSeq2VecWrapper(
            torch.nn.LSTM(300, hidden_size=512, num_layers=2, batch_first=True))

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.encoder.get_output_dim(),
                            out_features=vocab.get_vocab_size('label')))
        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, label=None):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
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
