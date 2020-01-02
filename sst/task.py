import os
import torch
from tqdm import tqdm
from allennlp.common.params import Params
from allennlp.common.util import prepare_environment
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.regularizers.regularizers import L2Regularizer
from allennlp.nn.regularizers.regularizer_applicator import RegularizerApplicator
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlpx import allenutil
from allennlpx.training.util import evaluate
from allennlpx.training.callback_trainer import CallbackTrainer
from allennlpx.training.callbacks.evaluate_callback import EvaluateCallback
from allennlpx.predictors.text_classifier import TextClassifierPredictor
from allennlpx.interpret.attackers.bruteforce import BruteForce
from allennlpx.interpret.attackers.attacker import DEFAULT_IGNORE_TOKENS
from luna import (auto_create, flt2str, log, log_config)
from sst.model import LstmClassifier
from sst.args import ProgramArgs
from sst.utils import (get_token_indexer, get_data_reader, get_data, get_iterator, get_optimizer, get_best_checkpoint)

# for logging
log_config("log", "cf")

class Task(object):
    def __init__(self, config: ProgramArgs):
        # get token indexer
        self.token_indexer = get_token_indexer(config.pretrain)

        # get data reader
        self.sub_reader = get_data_reader(self.token_indexer, subtree=True)
        self.reader = get_data_reader(self.token_indexer, subtree=False)

        # get dataset 
        self.sub_train_data, \
        self.train_data, \
        self.dev_data, \
        self.test_data = auto_create(f"{config.cache_prefix}_data",
                                     lambda: get_data(self.sub_reader, self.reader, config.train_data_file,
                                                      config.dev_data_file, config.test_data_file),
                                     True,
                                     config.cache_path)

        # get vocab
        self.vocab = auto_create(f"{config.cache_prefix}_vocab",
                                 lambda: Vocabulary.from_instances(self.sub_train_data + self.dev_data + self.test_data),
                                 True,
                                 config.cache_path)

        regularizer = RegularizerApplicator([("", L2Regularizer(config.weight_decay))])
        # get model
        self.model = LstmClassifier(self.vocab, config, regularizer).cuda()

        # get iterator
        self.iterator = get_iterator(config.batch_size)
        self.iterator.index_with(self.vocab)
                
    def run(self, config: ProgramArgs):
        getattr(self, config.mode)(config)    
                        
    def train(self, config: ProgramArgs):
        # get optimizer
        optimizer = get_optimizer(self.model, config.optimizer, config.learning_rate, config.weight_decay)
        # training 
        trainer = CallbackTrainer(model=self.model,
                                  optimizer=optimizer,
                                  iterator=self.iterator,
                                  train_dataset=self.sub_train_data,
                                  validation_dataset=self.dev_data,
                                  validation_metric='+accuracy',
                                  num_epochs=8,
                                  shuffle=True,
                                  patience=None,
                                  cuda_device=0,
                                  callbacks=[EvaluateCallback(self.test_data)])
        trainer.train()
        
    def evaluate(self, config: ProgramArgs):
        best_checkpoint_path = get_best_checkpoint(config.saved_path)
        print('load model from {}'.format(best_checkpoint_path))
        self.model.load_state_dict(torch.load(get_best_checkpoint(config.saved_path)))

        evaluate(self.model, self.test_data, self.iterator, cuda_device=0, batch_weight_key=None)
        
    def attack(self, config: ProgramArgs):
        best_checkpoint_path = get_best_checkpoint(config.saved_path)
        print('load model from {}'.format(best_checkpoint_path))
        self.model.load_state_dict(torch.load(get_best_checkpoint(config.saved_path)))

        evaluate(self.model, self.test_data, self.iterator, cuda_device=0, batch_weight_key=None)

        for module in self.model.modules():
            if isinstance(module, TextFieldEmbedder):
                for embed in module._token_embedders.keys():
                    module._token_embedders[embed].weight.requires_grad = True

        pos_words = [line.rstrip('\n') for line in open("../sentiment-words/positive-words.txt")]
        neg_words = [line.rstrip('\n') for line in open("../sentiment-words/negative-words.txt")]
        not_words = [line.rstrip('\n') for line in open("../sentiment-words/negation-words.txt")]
        forbidden_words = pos_words + neg_words + not_words + DEFAULT_IGNORE_TOKENS

        predictor = TextClassifierPredictor(self.model.cpu(), self.reader)
        # attacker = HotFlip(predictor)
        attacker = BruteForce(predictor)
        attacker.initialize()

        data_to_attack = self.test_data
        total_num = len(data_to_attack)
        # total_num = 20
        succ_num = 0
        src_words = []
        tgt_words = []
        for i in tqdm(range(total_num)):
            raw_text = allenutil.as_sentence(data_to_attack[i])

            result = attacker.attack_from_json({"sentence": raw_text},
                                               ignore_tokens=forbidden_words,
                                               forbidden_tokens=forbidden_words,
                                               max_change_num=5,
                                               search_num=256)

            att_text = allenutil.as_sentence(result['att'])

            if result["success"] == 1:
                succ_num += 1
                log("[raw]", raw_text)
                log("\t", flt2str(predictor.predict(raw_text)['probs']))
                log("[att]", att_text)
                log('\t', flt2str(predictor.predict(att_text)['probs']))
                log()

            raw_tokens = result['raw']
            att_tokens = result['att']
            for i in range(len(raw_tokens)):
                if raw_tokens[i] != att_tokens[i]:
                    src_words.append(raw_tokens[i])
                    tgt_words.append(att_tokens[i])

        print(f'Succ rate {succ_num / total_num * 100}')
        
        
    def augmentation(self, config: ProgramArgs):
        pass
    
    
if __name__ == '__main__':
    config = ProgramArgs.parse(True)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_device

    # random seed
    prepare_environment(Params({}))
    
    task = Task(config)
    task.run(config)