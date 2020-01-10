from awesome_glue.config import Config
from allennlpx.data.dataset_readers.berty_tsv import BertyTSVReader
from allennlpx.data.dataset_readers.spacy_tsv import SpacyTSVReader
from awesome_glue.task_specs import TASK_SPECS
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
from awesome_glue.models.bert_classifier import BertClassifier
from awesome_glue.models.lstm_classifier import LstmClassifier
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


class Task:
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        spec = TASK_SPECS[config.task_id]

        if config.tokenizer in ['spacy', 'xspacy']:
            self.reader = SpacyTSVReader(sent1_col=spec['sent1_col'],
                                         sent2_col=spec['sent2_col'],
                                         label_col=spec['label_col'],
                                         skip_label_indexing=spec['skip_label_indexing'],
                                         add_cls=config.tokenizer == 'xspacy')
        else:
            self.reader = BertyTSVReader(sent1_col=spec['sent1_col'],
                                         sent2_col=spec['sent2_col'],
                                         label_col=spec['label_col'],
                                         skip_label_indexing=spec['skip_label_indexing'])

        def __load_data():
            train_data = self.reader.read(f'{spec["path"]}/train.tsv')
            dev_data = self.reader.read(f'{spec["path"]}/dev.tsv')
            test_data = self.reader.read(f'{spec["path"]}/test.tsv')
            vocab = Vocabulary.from_instances(train_data + dev_data + test_data)
            return train_data, dev_data, test_data, vocab

        # The cache name is {task}-{tokenizer}
        self.train_data, self.dev_data, self.test_data, self.vocab = auto_create(
            f"{config.task_id}-{config.tokenizer}", __load_data, True)

        if config.arch == 'bert':
            self.model = BertClassifier(self.vocab, config.finetunable).cuda()
        elif config.arch == 'lstm':
            self.model = LstmClassifier(self.vocab, config.pretrain, config.finetunable).cuda()
        else:
            raise Exception

        self.predictor = TextClassifierPredictor(self.model, self.reader)

    def train(self):
        num_epochs = 4
        pseudo_batch_size = 16
        accumulate_num = 2
        batch_size = pseudo_batch_size * accumulate_num

        if isinstance(self.model, BertClassifier):
            optimizer = self.model.get_optimizer(
                    total_steps = (len(self.train_data) // batch_size + 1) * num_epochs)
        elif isinstance(self.model, LstmClassifier):
            optimizer = self.model.get_optimizer()

        if self.config.tokenizer in ['spacy', 'spacyx']:
            sorting_keys = [("sent", "num_tokens")]
        else:
            sorting_keys = [("berty_tokens", "num_tokens")]

        iterator = BucketIterator(batch_size=pseudo_batch_size, sorting_keys=sorting_keys)
        iterator.index_with(self.vocab)

        trainer = CallbackTrainer(
            model=self.model,
            optimizer=optimizer,
            iterator=iterator,
            train_dataset=self.train_data,
            validation_dataset=self.dev_data,
            num_epochs=num_epochs,
            shuffle=True,
            patience=None,
            grad_clipping=1.,
            cuda_device=0,
            num_gradient_accumulation_steps=accumulate_num,
            # serialization_dir='saved/allenmodels',
            # num_serialized_models_to_keep=1
            #   callbacks=[EvaluateCallback(self.dev_data)],
        )
        trainer.train()
        log(evaluate(self.model, self.dev_data, iterator, 0, None))
        save_model(self.model, self.config.model_name)

    def from_pretrained(self):
        load_model(self.model, self.config.model_name)

    @torch.no_grad()
    def evaluate(self):
        self.from_pretrained()
        self.model.eval()
        iterator = BasicIterator(batch_size=32)
        iterator.index_with(self.vocab)
        evaluate(self.model, self.test_data, iterator, 0, None)

    @torch.no_grad()
    def transfer_attack(self):
        self.from_pretrained()
        self.model.eval()
        df = pandas.read_csv(self.config.attack_tsv, sep='\t', quoting=csv.QUOTE_NONE)
        flip_num = 0
        for rid in range(1, df.shape[0]):
            raw = df.iloc[rid]['raw']
            att = df.iloc[rid]['att']
            raw_instance = self.reader.text_to_instance(raw)
            att_instance = self.reader.text_to_instance(att)
            results = self.model.forward_on_instances([raw_instance, att_instance])
            if (results[0]['probs'][0] - results[0]['probs'][1]) \
                * (results[1]['probs'][0] - results[1]['probs'][1]) < 0:
                flip_num += 1
        print(f'flipped {flip_num/df.shape[0]}')

    def attack(self):
        self.from_pretrained()
        self.model.eval()
        f_tsv = open(f"nogit/{self.config.model_name}.attack.tsv", 'w')
        f_tsv.write("raw\tatt\n")
        for module in self.model.modules():
            if isinstance(module, TextFieldEmbedder):
                for embed in module._token_embedders.keys():
                    module._token_embedders[embed].weight.requires_grad = True

        pos_words = [line.rstrip('\n') for line in open("sentiment-words/positive-words.txt")]
        neg_words = [line.rstrip('\n') for line in open("sentiment-words/negative-words.txt")]
        not_words = [line.rstrip('\n') for line in open("sentiment-words/negation-words.txt")]
        forbidden_words = pos_words + neg_words + not_words + DEFAULT_IGNORE_TOKENS

        predictor = TextClassifierPredictor(self.model.cpu(), self.reader)

        # attacker = HotFlip(predictor)
        attacker = BruteForce(predictor)
        attacker.initialize()

        data_to_attack = self.test_data[:200]
        total_num = len(data_to_attack)
        succ_num = 0
        for i in tqdm(range(total_num)):
            raw_text = allenutil.as_sentence(data_to_attack[i], 'sent')

            # result = attacker.attack_from_json({"sentence": raw_text},
            #                                    ignore_tokens=forbidden_words,
            #                                    forbidden_tokens=forbidden_words,
            #                                    step_size=100,
            #                                    max_change_num=3,
            #                                    iter_change_num=2)

            result = attacker.attack_from_json({"sentence": raw_text},
                                               field_to_change="sent",
                                               ignore_tokens=forbidden_words,
                                               forbidden_tokens=forbidden_words,
                                               max_change_num=5,
                                               search_num=256)

            att_text = allenutil.as_sentence(result['att'], 'sent')

            if result["success"] == 1:
                succ_num += 1
                log("[raw]", raw_text)
                log("\t", flt2str(predictor.predict(raw_text)['probs']))
                log("[att]", att_text)
                log('\t', flt2str(predictor.predict(att_text)['probs']))
                log()

            f_tsv.write(f"{raw_text}\t{att_text}\n")
        f_tsv.close()
        print(f'Succ rate {succ_num/total_num*100}')
