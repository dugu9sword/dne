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
from luna import (auto_create, flt2str, log, log_config, ram_read, ram_reset, ram_write,
                  ram_globalize)
from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from collections import defaultdict
from allennlpx.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from awesome_sst.freq_util import analyze_frequency, frequency_analysis
from allennlpx.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper

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
    def __init__(self, vocab, pretrain, finetunable=False):
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
            torch.nn.LSTM(EMBED_DIM[pretrain], hidden_size=512, num_layers=2, batch_first=True))
        self.linear = torch.nn.Linear(in_features=self.encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('label'))

        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

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
