import csv
import random
from collections import Counter
from statistics import mode

import faiss
import nlpaug.augmenter.word as naw
import numpy as np
import pandas
import torch
from allennlp.data import Instance
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.iterators.bucket_iterator import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders.embedding import \
    _read_pretrained_embeddings_file
from allennlp.training.learning_rate_schedulers.slanted_triangular import \
    SlantedTriangular
from allennlp.training.optimizers import DenseSparseAdam
from pytorch_pretrained_bert.optimization import BertAdam
from torch.optim import AdamW
from torch.optim.adam import Adam
from tqdm import tqdm

from allennlpx import allenutil
from allennlpx.data.dataset_readers.berty_tsv import BertyTSVReader
from allennlpx.data.dataset_readers.spacy_tsv import SpacyTSVReader
from allennlpx.interpret.attackers.attacker import DEFAULT_IGNORE_TOKENS
from allennlpx.interpret.attackers.bruteforce import BruteForce
from allennlpx.interpret.attackers.hotflip import HotFlip
from allennlpx.interpret.attackers.pgd import PGD
from allennlpx.interpret.attackers.pwws import PWWS
from allennlpx.interpret.attackers.genetic import Genetic
from allennlpx.interpret.attackers.policies import (CandidatePolicy,
                                                    EmbeddingPolicy,
                                                    SynonymPolicy)
from allennlpx.modules.knn_utils import build_faiss_index
from allennlpx.predictors.predictor import Predictor
from allennlpx.predictors.text_classifier import TextClassifierPredictor
from allennlpx.training.callback_trainer import CallbackTrainer
from allennlpx.training.callbacks.evaluate_callback import (EvaluateCallback,
                                                            evaluate)
from awesome_glue.config import Config
from awesome_glue.models.bert_classifier import BertClassifier
from awesome_glue.models.lstm_classifier import LstmClassifier
from awesome_glue.task_specs import TASK_SPECS
from awesome_glue.utils import EMBED_DIM, WORD2VECS, AttackMetric
from luna import flt2str, ram_append, ram_has, ram_read, ram_reset, ram_write
from luna.logging import log
from luna.public import auto_create, Aggregator
from luna.pytorch import load_model, save_model, set_seed


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
        pseudo_batch_size * accumulate_num

        optimizer = self.model.get_optimizer()

        if self.config.tokenizer in ['spacy', 'spacyx']:
            sorting_keys = [("sent", "num_tokens")]
        else:
            sorting_keys = [("berty_tokens", "num_tokens")]

        iterator = BucketIterator(batch_size=pseudo_batch_size, sorting_keys=sorting_keys)
        iterator.index_with(self.vocab)

        if self.config.aug_data != '':
            log(f'Augment data from {self.config.aug_data}')
            self.train_data.extend(self.reader.read(aug_data))

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
        iterator = BasicIterator(batch_size=32)
        iterator.index_with(self.vocab)

        ram_write("knn_flag", "collect")
        filtered = list(filter(lambda x: len(x.fields['berty_tokens'].tokens) > 10,
                               self.train_data))
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
        iterator = BasicIterator(batch_size=32)
        iterator.index_with(self.vocab)
        evaluate(self.model, self.dev_data, iterator, 0, None)

    @torch.no_grad()
    def transfer_attack(self):
        self.from_pretrained()
        # self.model.eval()
        set_seed(11221)
        df = pandas.read_csv(self.config.adv_data, sep='\t', quoting=csv.QUOTE_NONE)
        attack_metric = AttackMetric()
        
        def identity(x):
            return x
        
        def rand_drop(x):
            x_split = x.split(" ")
            for i in range(min(3, len(x_split) - 1)):
                x_split.pop(random.randrange(len(x_split)))
            return " ".join(x_split)
        
        def rand_stop(x):
            x_split = x.split(" ")
            idxs = random.choices(range(len(x_split)), k=5)
            for i in idxs:
                x_split[i] = 'the'
            return " ".join(x_split)
        
        def embed_aug(x):
            aug_num = 5
            if ram_has("emb_aug"):
                aug = ram_read("emb_aug")
            else:
                aug = naw.WordEmbsAug(  
                    model_type = 'glove',
                    top_k=30,
                    model_path = '/home/zhouyi/counter-fitting/word_vectors/counter-fitted-vectors.txt',
#                     model_path = '/home/zhouyi/counter-fitting/word_vectors/glove.txt',
                    aug_min = aug_num,
                    aug_max = aug_num,
                    stopwords = ['a', 'the'],
                    stopwords_regex = '@',
                )
                ram_write("emb_aug", aug)
            if len(x.split(' ')) < aug_num:
                aug.aug_min = 1
            ret = aug.substitute(x)
            aug.aug_min = aug_num
            return ret
        
        def syn_aug(x):
            aug_num = 5
            if ram_has("syn_aug"):
                aug = ram_read("syn_aug")
            else:
                while True:
                    try:
                        aug = naw.SynonymAug(
                            aug_min = aug_num,
                            aug_max = aug_num,
                            stopwords = ['a', 'the'],
                            stopwords_regex = '@',
                        )
                        break
                    except:
                        import nltk
                        nltk.download('wordnet')
                        nltk.download('averaged_perceptron_tagger')
                ram_write("syn_aug", aug)
            if len(x.split(' ')) < aug_num:
                aug.aug_min = 1
            ret = aug.substitute(x)
            aug.aug_min = aug_num
            return ret
            
        def bert_aug(x):
            aug_num = 5
            if ram_has("bert_aug"):
                aug = ram_read("bert_aug")
            else:
                aug = naw.ContextualWordEmbsAug(
                    model_path='bert-base-uncased', 
                    top_k = 30,
                    aug_min = aug_num,
                    aug_max = aug_num,
                    stopwords = ['a', 'the'],
                    stopwords_regex = '@',
                    action="substitute")
                ram_write("bert_aug", aug)
            if len(x.split(' ')) < aug_num:
                aug.aug_min = 1
            ret = aug.augment(x)
            aug.aug_min = aug_num
            return ret    
                
        transform = bert_aug
        ensemble_num = 9
            
        for rid in tqdm(range(df.shape[0])):
            raw = df.iloc[rid]['raw']
            adv = df.iloc[rid]['adv']
            
#             new_adv = []
#             for wr, wa in zip(raw.split(" "), att.split(" ")):
#                 if wr == wa:
#                     new_att.append(wa)
#                 else:
#                     pass
# #                     new_att.append()
#             adv = " ".join(new_att)

            raw_instances, adv_instances = [], []
            for i in range(ensemble_num):
                raw_instances.append(self.reader.text_to_instance(transform(raw)))
                adv_instances.append(self.reader.text_to_instance(transform(adv)))
            results = self.model.forward_on_instances(raw_instances + adv_instances)
            
            raw_preds, adv_preds = [], []
            for i in range(ensemble_num):
                raw_preds.append(np.argmax(results[i]['probs']))
                adv_preds.append(np.argmax(results[i + ensemble_num]['probs']))
            raw_pred = mode(raw_preds)
            adv_pred = mode(adv_preds)
            
            label = df.iloc[rid]['label']
            
            if raw_pred == label:
                if adv_pred != raw_pred:
                    attack_metric.succeed()
                else:
                    attack_metric.fail()
            else:
                attack_metric.escape()
            print('Agg metric', attack_metric)
        print(Counter(df["label"].tolist()))
        print(attack_metric)

    def attack(self):
        self.from_pretrained()
        self.model.eval()

        spacy_data = load_data(self.config.task_id, "spacy")
        spacy_vocab: Vocabulary = spacy_data['vocab']
        spacy_weight = auto_create(
            f"{self.config.task_id}-{self.config.tokenizer}-{self.config.attack_vectors}",
            lambda: _read_pretrained_embeddings_file(WORD2VECS[self.config.attack_vectors],
                                                     embedding_dim=EMBED_DIM[self.config.
                                                                             attack_vectors],
                                                     vocab=spacy_vocab,
                                                     namespace="tokens"), True)

        if self.config.attack_gen_adv:
            f_adv = open(f"nogit/{self.config.model_name}.adv.tsv", 'w')
            f_adv.write("raw\tadv\tlabel\n")

        if self.config.attack_gen_aug:
            f_aug = open(f"nogit/{self.config.model_name}.aug.tsv", 'w')
            f_aug.write("sentence\tlabel\n")

        for module in self.model.modules():
            if isinstance(module, TextFieldEmbedder):
                for embed in module._token_embedders.keys():
                    module._token_embedders[embed].weight.requires_grad = True

        pos_words = [line.rstrip('\n') for line in open("sentiment-words/positive-words.txt")]
        neg_words = [line.rstrip('\n') for line in open("sentiment-words/negative-words.txt")]
        not_words = [line.rstrip('\n') for line in open("sentiment-words/negation-words.txt")]
        forbidden_words = pos_words + neg_words + not_words + DEFAULT_IGNORE_TOKENS

        # self.predictor._model.cpu()
        kwargs = { 
            "ignore_tokens": forbidden_words,
            "forbidden_tokens": forbidden_words,
            "max_change_num_or_ratio": 0.25
        }
        attacker = HotFlip(self.predictor)
        attacker.initialize()
#         attacker = BruteForce(self.predictor, **kwargs)
#         attacker = PWWS(self.predictor, **kwargs)
#         attacker = Genetic(self.predictor, **kwargs)
#         attacker.initialize(vocab=spacy_vocab, token_embedding=spacy_weight)

        if self.config.attack_data_split == 'train':
            data_to_attack = self.train_data
            if self.config.task_id == 'SST':
                if self.config.arch == 'bert':
                    field_name = 'berty_tokens'
                else:
                    field_name = 'sent'
                data_to_attack = list(
                    filter(lambda x: len(x[field_name].tokens) > 20, data_to_attack))
        elif self.config.attack_data_split == 'dev':
            data_to_attack = self.dev_data
        if self.config.attack_size == -1:
            adv_number = len(data_to_attack)
        else:
            adv_number = self.config.attack_size
        data_to_attack = data_to_attack[:adv_number]

        attack_metric = AttackMetric()
        agg = Aggregator()
        for i in tqdm(range(adv_number)):
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
#                                                    policy=EmbeddingPolicy(measure='euc', topk=10, rho=None),
#                                                    policy=SynonymPolicy(), 
#                                                    search_num=256
                                                  )
    

                if result["success"] == 1:
                    changed_num = 0
                    for wr, wa in zip(result['raw'], result['adv']):
                        if wr != wa:
                            changed_num += 1
                    to_aggregate = [('changed', changed_num)]
                    if "generation" in result:
                        to_aggregate.append(('generation', result['generation']))
                    agg.aggregate(*to_aggregate)
                    attack_metric.succeed()
                    adv_text = allenutil.as_sentence(result['adv'])
                    log("[raw]", raw_text)
                    log("\t", flt2str(self.predictor.predict(raw_text)['probs']))
                    log("[adv]", adv_text)
                    log('\t', flt2str(self.predictor.predict(adv_text)['probs']))
                    if "changed" in result:
                        log("[changed]", result['changed'])
                    log()
                    log("Changed", agg.mean("changed"))
                    if "generation" in result:
                        log("Generation", agg.mean("generation"))
                    log("Aggregated metric:", attack_metric)
                else:
                    attack_metric.fail()
            else:
                attack_metric.escape()
                adv_text = raw_text

            if self.config.attack_gen_adv:
                f_adv.write(f"{raw_text}\t{adv_text}\t{raw_label}\n")
            if self.config.attack_gen_aug:
                f_aug.write(f"{adv_text}\t{raw_label}\n")

        if self.config.attack_gen_adv: 
            f_adv.close()
        if self.config.attack_gen_aug:
            f_aug.close()

        print(attack_metric)
