from allennlpx.data.dataset_readers.berty_tsv import BertyTSVReader
from allennlpx.data.dataset_readers.spacy_tsv import SpacyTSVReader
from allennlp.data.vocabulary import Vocabulary
from torch.optim.adam import Adam
from allennlp.training.optimizers import DenseSparseAdam
from allennlp.data.iterators.bucket_iterator import BucketIterator
from allennlpx.training.callback_trainer import CallbackTrainer
from allennlpx.training.callbacks.evaluate_callback import EvaluateCallback, evaluate
from luna.pytorch import save_model
from allennlp.data.iterators.basic_iterator import BasicIterator
import torch
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from luna.public import auto_create
from luna.pytorch import load_model
from allennlp.training.learning_rate_schedulers.slanted_triangular import SlantedTriangular
from pytorch_pretrained_bert.optimization import BertAdam
from luna.logging import log
from luna import flt2str
from allennlpx import allenutil
from allennlpx.interpret.attackers.attacker import DEFAULT_IGNORE_TOKENS
from allennlpx.interpret.attackers.bruteforce import BruteForce
from allennlpx.interpret.attackers.hotflip import HotFlip
from allennlpx.interpret.attackers.pgd import PGD
from allennlpx.predictors.text_classifier import TextClassifierPredictor
from tqdm import tqdm
import pandas
import csv
from torch.optim import AdamW


class KNNTask:
    def __init__(self):
        super().__init__()
        self.reader = SpacyTSVReader(sent1_col=spec['sent1_col'],
                                     sent2_col=spec['sent2_col'],
                                     label_col=spec['label_col'],
                                     skip_label_indexing=spec['skip_label_indexing'])

        def __load_data():
            train_data = self.reader.read('glue_data/SST-2/train.tsv')
            dev_data = self.reader.read('glue_data/SST-2/dev.tsv')
            test_data = self.reader.read('glue_data/SST-2/test.tsv')
            vocab = Vocabulary.from_instances(train_data + dev_data + test_data)
            return train_data, dev_data, test_data, vocab

        # The cache name is {task}-{tokenizer}
        self.train_data, self.dev_data, self.test_data, self.vocab = auto_create(
            "SST-spacy", __load_data, True)

    def train(self):
        pass


if __name__ == "__main__":
    task = KNNTask()
    task.train()