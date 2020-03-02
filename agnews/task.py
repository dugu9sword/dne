from typing import Dict
from luna.public import auto_create
from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from allennlp.common.params import Params
from allennlp.common.util import prepare_environment
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator,BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper
from allennlpx.modules.seq2vec_encoders.seq_max_pooler import PytorchSeqMaxPooler
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.training.optimizers import DenseSparseAdam
from allennlp.predictors.text_classifier import TextClassifierPredictor
from allennlp.interpret.attackers.hotflip import Hotflip
from allennlpx.data.dataset_readers.imdb import ImdbDatasetReader
from allennlpx.data.dataset_readers.yelp import YelpDatasetReader
from allennlpx.data.dataset_readers.agnews import AGNewsDataReader
from allennlpx.training.util import evaluate
from allennlpx import allenutil

import logging
# disable logging from allennlp
logging.getLogger('allennlp').setLevel(logging.CRITICAL)

from agnews.args import ProgramArgs
from agnews.model import RnnClassifier

class Task(object):
    def __init__(self, config: ProgramArgs):
        # get token indexer
        self.token_indexer = SingleIdTokenIndexer(lowercase_tokens=True)

        # get data reader
        self.data_reader_clean = AGNewsDataReader(token_indexers={"tokens": self.token_indexer})
        self.data_reader_crop = AGNewsDataReader(token_indexers={"tokens": self.token_indexer},
                                            crop=True,
                                            crop_batch_size=config.crop_batch_size,
                                            crop_window_size_rate=config.crop_window_size_rate,
                                            crop_min_window_size=config.crop_min_window_size)

        # get dataset 
        self.train_data_clean, \
        self.valid_data_clean, \
        self.test_data_clean = auto_create("agnews",
                                           lambda: self.read_dataset(self.data_reader_clean, config.train_data_file, config.valid_data_file, config.test_data_file),
                                           True,
                                           config.cache_path)
        self.test_data_clean = self.data_reader_clean.read(config.test_data_file)
        # get vocab
        self.vocab = auto_create('agnews_vocab',
                                 lambda: Vocabulary.from_instances(self.train_data_clean + self.test_data_clean),
                                 True,
                                 config.cache_path)

        # build word embedding 
        weight = auto_create('agnews_glove',
                             lambda: _read_pretrained_embeddings_file(config.embedding_path, embedding_dim=config.embedding_dim, vocab=self.vocab, namespace="tokens"),
                             True,
                             config.cache_path)
        weight = F.normalize(weight, p=2, dim=1)
        token_embedder = Embedding(num_embeddings=self.vocab.get_vocab_size('tokens'),
                                   embedding_dim=config.embedding_dim,
                                   weight=weight,
                                   trainable=False
                                   )
        word_embedders = BasicTextFieldEmbedder({"tokens": token_embedder})
        
        # build encoder for classifier
        encoder = PytorchSeqMaxPooler(torch.nn.LSTM(config.embedding_dim, hidden_size=config.hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True))

        self.model = RnnClassifier(self.vocab, word_embedders, encoder, dropout=0.3).cuda()

        # get iterator
        self.iterator = BucketIterator(batch_size=config.batch_size, sorting_keys=[("tokens", "num_tokens")])
        # iterator = BasicIterator(batch_size=batch_size)
        self.iterator.index_with(self.vocab)

    def read_dataset(self, data_reader, train_file, valid_file, test_file):
        train_data = data_reader.read(train_file)
        valid_data = data_reader.read(valid_file)
        test_data = data_reader.read(test_file)
        return train_data, valid_data, test_data

    def get_best_checkpoint(self, path: str):
        # The author is lazy ,so it is the most easy way in allennlp to find the best
        assert os.path.exists(path) and os.path.isdir(path)
        return "{}/best.th".format(path)

    def run(self, config: ProgramArgs):
        getattr(self, config.mode)(config)

    def train(self, config: ProgramArgs):
        if config.train_data_source == 'crop':
            self.train_data_crop = auto_create("agnews_train_crop",
                                               lambda: self.data_reader_crop.read(config.train_data_file),
                                               True,
                                               config.cache_path)
        need_grad = lambda x: x.requires_grad
        optimizer = torch.optim.Adam(filter(need_grad, self.model.parameters()), lr=config.learning_rate)

        # training 
        if config.train_data_source == 'crop':
            train_data = self.train_data_crop
        else:
            train_data = self.train_data_clean
        if config.test_data_source == 'crop':
            test_data = self.test_data_crop
        else:
            test_data = self.test_data_clean
        trainer = Trainer(model=self.model,
                          optimizer=optimizer,
                          iterator=self.iterator,
                          train_dataset=train_data,
                          validation_dataset=test_data,
                          validation_metric='+accuracy',
                          shuffle=True,
                          patience=None,
                          cuda_device=0,
                          num_epochs=config.epochs,
                          serialization_dir=config.save_path,
                          num_serialized_models_to_keep=5
                          )
        trainer.train()

    def evaluate(self, config: ProgramArgs):
        best_checkpoint_path = self.get_best_checkpoint(config.save_path)
        print('load model from {}'.format(best_checkpoint_path))
        self.model.load_state_dict(torch.load(best_checkpoint_path))
        if config.test_data_source == 'crop':
            self.test_data_crop = self.data_reader_crop.read(config.test_data_file)
            ensemble_iterator = BasicIterator(batch_size=config.crop_batch_size)
            ensemble_iterator.index_with(self.vocab)
            evaluate(self.model, self.test_data_crop, ensemble_iterator, cuda_device=0, ensemble=True, batch_weight_key=None)
        else:
            evaluate(self.model, self.test_data_clean, self.iterator, cuda_device=0, batch_weight_key=None)
        # best_checkpoint_path = get_best_checkpoint(config.saved_path)
        # print('load model from {}'.format(best_checkpoint_path))
        # self.model.load_state_dict(torch.load(get_best_checkpoint(config.saved_path)))
        # 
        # evaluate(self.model, self.test_data, self.iterator, cuda_device=0, batch_weight_key=None)

    def attack(self, config: ProgramArgs):
        best_checkpoint_path = self.get_best_checkpoint(config.save_path)
        print('load model from {}'.format(best_checkpoint_path))
        self.model.load_state_dict(torch.load(best_checkpoint_path))

        evaluate(self.model, self.test_data_clean, self.iterator, cuda_device=0, batch_weight_key=None)

        for module in self.model.modules():
            if isinstance(module, TextFieldEmbedder):
                for embed in module._token_embedders.keys():
                    module._token_embedders[embed].weight.requires_grad = True

        predictor = TextClassifierPredictor(self.model, self.data_reader_clean)
        # attacker = HotFlip(predictor)
        attacker = Hotflip(predictor)
        attacker.initialize()

        data_to_attack = self.test_data_clean
        total_num = len(data_to_attack)
        # total_num = 20
        succ_num = 0
        att_text_list = []
        for i in tqdm(range(total_num)):
            raw_text = allenutil.as_sentence(data_to_attack[i])
            result = attacker.attack_from_json({"sentence": raw_text})
            raw_text = " ".join(result['original'])
            att_text = " ".join(result['final'][0] if isinstance(result['final'],list) else result['final'])
            
            raw_probs = predictor.predict(raw_text)['probs']
            att_probs = predictor.predict(att_text)['probs']
            if np.argmax(raw_probs) != np.argmax(att_probs):
                succ_num += 1
                att_text_list.append(att_text)
            else:
                att_text_list.append(raw_text)

        print('Succ rate {:.2f}%'.format(succ_num / total_num * 100))
        gold_label = [data.fields['label'].label for data in self.test_data_clean]
        
        import csv
        with open("{}/data/agnews/attack.csv".format(config.workspace), 'w') as file:
            writer = csv.writer(file)
            for label, sentence in zip(gold_label, att_text_list):
                writer.writerow([label,"",sentence])

    def augmentation(self, config: ProgramArgs):
        pass


if __name__ == '__main__':
    config = ProgramArgs.parse(True)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_device

    # random seed
    prepare_environment(Params({}))

    task = Task(config)
    task.run(config)