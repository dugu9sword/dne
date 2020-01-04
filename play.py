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
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
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
from awesome_sst.freq_util import analyze_frequency, frequency_analysis
from luna import (auto_create, flt2str, log, log_config, ram_read, ram_reset, ram_write)
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from luna.pytorch import load_model, save_model, exist_model
from awesome_sst.sst_model import LstmClassifier, WORD2VECS
from awesome_sst.config import Config
from awesome_sst.task import Task
from allennlpx.training.util import evaluate
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from allennlp.common.tqdm import Tqdm

config = Config()._parse_args()

if config.alchemist:
    Tqdm.set_slower_interval(True)

ram_write("config", config)

log_config("log", "cf")
log(config)

task = Task(config)


# 
if config.mode == 'train':
    task.train()
elif config.mode == 'eval':
    task.from_trained()
    task.evaluate()
elif config.mode == 'attack':
    task.from_trained()
    task.attack()
