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
from allennlpx.interpret.attackers.bert_bruteforce import BertBruteForce
from allennlpx.interpret.attackers.hotflip import HotFlip
from allennlpx.interpret.attackers.pgd import PGD
from allennlpx.predictors.predictor import Predictor
from allennlpx.predictors.text_classifier import TextClassifierPredictor
from tqdm import tqdm
import pandas
import csv
from torch.optim import AdamW
import numpy as np
from awesome_glue.utils import AttackMetric, WORD2VECS
from typing import List
from allennlp.data import Instance
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from awesome_glue.utils import EMBED_DIM, WORD2VECS
from luna import ram_read, ram_write, ram_append, ram_reset, ram_has
from allennlpx.modules.knn_utils import build_faiss_index
import faiss


def load_data(task_id: str, tokenizer: str):
    spec = TASK_SPECS[task_id]

    if tokenizer in ['spacy', 'xspacy']:
        reader = SpacyTSVReader(sent1_col=spec['sent1_col'],
                                sent2_col=spec['sent2_col'],
                                label_col=spec['label_col'],
                                skip_label_indexing=spec['skip_label_indexing'],
                                add_cls=tokenizer == 'xspacy')
    else:
        reader = BertyTSVReader(sent1_col=spec['sent1_col'],
                                sent2_col=spec['sent2_col'],
                                label_col=spec['label_col'],
                                skip_label_indexing=spec['skip_label_indexing'])

    def __load_data():
        train_data = reader.read(f'{spec["path"]}/train.tsv')
        dev_data = reader.read(f'{spec["path"]}/dev.tsv')
        test_data = reader.read(f'{spec["path"]}/test.tsv')
        vocab = Vocabulary.from_instances(train_data + dev_data + test_data)
        return train_data, dev_data, test_data, vocab

    # The cache name is {task}-{tokenizer}
    train_data, dev_data, test_data, vocab = auto_create(f"{task_id}-{tokenizer}", __load_data,
                                                         True)

    return {"reader": reader, "data": (train_data, dev_data, test_data), "vocab": vocab}


class Task:
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        loaded_data = load_data(config.task_id, config.tokenizer)
        self.reader = loaded_data['reader']
        self.train_data, self.dev_data, self.test_data = loaded_data['data']
        self.vocab: Vocabulary = loaded_data['vocab']

        if config.arch == 'bert':
            self.model = BertClassifier(self.vocab, config.finetunable).cuda()
        elif config.arch == 'lstm':
            self.model = LstmClassifier(self.vocab, config.pretrain, config.finetunable).cuda()
        else:
            raise Exception

        self.predictor = TextClassifierPredictor(
            self.model, self.reader, key='sent' if self.config.arch != 'bert' else 'berty_tokens')

    def train(self):
        num_epochs = 2
        pseudo_batch_size = 32
        accumulate_num = 1
        batch_size = pseudo_batch_size * accumulate_num

        optimizer = self.model.get_optimizer()

        if self.config.tokenizer in ['spacy', 'spacyx']:
            sorting_keys = [("sent", "num_tokens")]
        else:
            sorting_keys = [("berty_tokens", "berty_tokens___input_ids")]

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
            # callbacks=[EvaluateCallback(self.dev_data)],
        )
        trainer.train()
        log(evaluate(self.model, self.dev_data, iterator, 0, None))
        save_model(self.model, self.config.model_name)

    def from_pretrained(self):
        load_model(self.model, self.config.model_name)

    def knn_build_index(self):
        self.from_pretrained()
        self.model.eval()
        iterator = BasicIterator(batch_size=32)
        iterator.index_with(self.vocab)

        ram_write("knn_flag", "collect")
        filtered = list(filter(lambda x: len(x.fields['berty_tokens'].tokens) > 10, self.train_data))
        evaluate(self.model, filtered, iterator, 0, None)

    def knn_evaluate(self):
        ram_write("knn_flag", "infer")
        self.evaluate()

    def knn_attack(self):
        ram_write("knn_flag", "infer")
        self.attack()

    @torch.no_grad()
    def evaluate(self):
        self.from_pretrained()
        self.model.eval()
        iterator = BasicIterator(batch_size=32)
        iterator.index_with(self.vocab)
        evaluate(self.model, self.dev_data, iterator, 0, None)

    @torch.no_grad()
    def transfer_attack(self):
        self.from_pretrained()
        self.model.eval()
        df = pandas.read_csv(self.config.attack_tsv, sep='\t', quoting=csv.QUOTE_NONE)
        attack_metric = AttackMetric()
        for rid in range(df.shape[0]):
            raw = df.iloc[rid]['raw']
            att = df.iloc[rid]['att']
            raw_instance = self.reader.text_to_instance(raw)
            att_instance = self.reader.text_to_instance(att)
            results = self.model.forward_on_instances([raw_instance, att_instance])
            raw_pred = np.argmax(results[0]['probs'])
            att_pred = np.argmax(results[1]['probs'])
            label = df.iloc[rid]['label']
            if raw_pred == label:
                if att_pred != raw_pred:
                    attack_metric.succeed()
                else:
                    attack_metric.fail()
            else:
                attack_metric.escape()
        print(attack_metric)

    def attack(self):
        self.from_pretrained()
        self.model.eval()

        if self.config.arch == 'bert':
            spacy_data = load_data(self.config.task_id, "spacy")
            spacy_vocab: Vocabulary = spacy_data['vocab']
            spacy_weight = auto_create(
                f"{self.config.task_id}-{self.config.tokenizer}-{self.config.attack_vectors}",
                lambda: _read_pretrained_embeddings_file(WORD2VECS[self.config.attack_vectors],
                                                         embedding_dim=EMBED_DIM[self.config.
                                                                                 attack_vectors],
                                                         vocab=spacy_vocab,
                                                         namespace="tokens"), True)

        f_tsv = open(f"nogit/{self.config.model_name}.attack.tsv", 'w')
        f_tsv.write("raw\tatt\tlabel\n")

        for module in self.model.modules():
            if isinstance(module, TextFieldEmbedder):
                for embed in module._token_embedders.keys():
                    module._token_embedders[embed].weight.requires_grad = True

        pos_words = [line.rstrip('\n') for line in open("sentiment-words/positive-words.txt")]
        neg_words = [line.rstrip('\n') for line in open("sentiment-words/negative-words.txt")]
        not_words = [line.rstrip('\n') for line in open("sentiment-words/negation-words.txt")]
        forbidden_words = pos_words + neg_words + not_words + DEFAULT_IGNORE_TOKENS

        # self.predictor._model.cpu()
        if self.config.arch == 'bert':
            attacker = BertBruteForce(self.predictor)
            attacker.initialize(vocab=spacy_vocab, token_embedding=spacy_weight)
        else:
            # attacker = HotFlip(predictor)
            attacker = BruteForce(self.predictor)
            attacker.initialize()

        data_to_attack = self.dev_data[:200]
        total_num = len(data_to_attack)
        attack_metric = AttackMetric()
        for i in tqdm(range(total_num)):
            raw_text = allenutil.as_sentence(data_to_attack[i])
            raw_pred = np.argmax(self.predictor.predict(raw_text)['probs'])
            raw_label = data_to_attack[i]['label'].label
            # Only attack correct instance
            if raw_pred == raw_label:
                # result = attacker.attack_from_json({"sentence": raw_text},
                #                                    ignore_tokens=forbidden_words,
                #                                    forbidden_tokens=forbidden_words,
                #                                    step_size=100,
                #                                    max_change_num=3,
                #                                    iter_change_num=2)
                if self.config.arch == 'bert':
                    field_to_change = 'berty_tokens'
                elif self.config.arch == 'lstm':
                    field_to_change = 'sent'

                result = attacker.attack_from_json({field_to_change: raw_text},
                                                   field_to_change=field_to_change,
                                                   ignore_tokens=forbidden_words,
                                                   forbidden_tokens=forbidden_words,
                                                   max_change_num=3,
                                                   search_num=128)

                att_text = allenutil.as_sentence(result['att'])

                if result["success"] == 1:
                    attack_metric.succeed()
                    log("[raw]", raw_text)
                    log("\t", flt2str(self.predictor.predict(raw_text)['probs']))
                    log("[att]", att_text)
                    log('\t', flt2str(self.predictor.predict(att_text)['probs']))
                    log()
                else:
                    attack_metric.fail()
            else:
                attack_metric.escape()
                att_text = raw_text

            f_tsv.write(f"{raw_text}\t{att_text}\t{raw_label}\n")

        f_tsv.close()

        print(attack_metric)
