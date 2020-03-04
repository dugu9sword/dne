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
from allennlp.training.learning_rate_schedulers.slanted_triangular import \
    SlantedTriangular
from allennlp.training.optimizers import DenseSparseAdam
from pytorch_pretrained_bert.optimization import BertAdam
from torch.optim import AdamW
from torch.optim.adam import Adam
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from allennlpx import allenutil
from allennlpx.data.dataset_readers.berty_tsv import BertyTSVReader
from allennlpx.data.dataset_readers.spacy_tsv import SpacyTSVReader
from allennlpx.interpret.attackers.attacker import DEFAULT_IGNORE_TOKENS
from allennlpx.interpret.attackers.bruteforce import BruteForce
from allennlp.interpret.attackers.hotflip import Hotflip as RawHotFlip
from allennlpx.interpret.attackers.hotflip import HotFlip
from allennlpx.interpret.attackers.pgd import PGD
from allennlpx.interpret.attackers.pwws import PWWS
from allennlpx.interpret.attackers.genetic import Genetic
from allennlpx.interpret.attackers.policies import (CandidatePolicy,
                                                    EmbeddingPolicy,
                                                    SynonymPolicy)
from allennlpx.modules.token_embedders.embedding import \
    _read_pretrained_embeddings_file
from allennlpx.modules.knn_utils import build_faiss_index, H5pyCollector
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
from awesome_glue.transforms import identity, rand_drop, embed_aug, syn_aug, bert_aug
from luna import flt2str, ram_append, ram_has, ram_read, ram_reset, ram_write
from luna.logging import log
from luna.public import auto_create, Aggregator
from luna.pytorch import load_model, save_model, set_seed


def load_data(task_id: str, tokenizer: str):
    spec = TASK_SPECS[task_id]

    reader = {
        'spacy': SpacyTSVReader, 'bert': BertyTSVReader
    }[tokenizer](sent1_col=spec['sent1_col'],
                 sent2_col=spec['sent2_col'],
                 label_col=spec['label_col'],
                 skip_label_indexing=spec['skip_label_indexing']
    )

    def __load_data():
        train_data = reader.read(f'{spec["path"]}/train.tsv')
        dev_data = reader.read(f'{spec["path"]}/dev.tsv')
        test_data = reader.read(f'{spec["path"]}/test.tsv')
        vocab = Vocabulary.from_instances(train_data + dev_data + test_data)
        return train_data, dev_data, test_data, vocab

    # The cache name is {task}-{tokenizer}
    train_data, dev_data, test_data, vocab = auto_create(f"{task_id}-{tokenizer}.data", __load_data,
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
            self.model = LstmClassifier(self.vocab, 
                                        TASK_SPECS[config.task_id]['num_labels'],
                                        config.pretrain, 
                                        config.finetunable,
                                        f"{config.task_id}-{config.pretrain}.vec").cuda()
        else:
            raise Exception

        self.predictor = TextClassifierPredictor(
            self.model, self.reader, key='sent' if self.config.arch != 'bert' else 'berty_tokens')

    def train(self):
        num_epochs = 5
        pseudo_batch_size = 32
        accumulate_num = 1
        pseudo_batch_size * accumulate_num

        optimizer = self.model.get_optimizer()

        if self.config.tokenizer == 'spacy':
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
        
    def build_manifold(self):
        spacy_data = load_data(self.config.task_id, "spacy")
        train_data, dev_data, _ = spacy_data['data']
        if self.config.task_id == 'SST':
            train_data = list(filter(lambda x: len(x["sent"].tokens) > 15, train_data))
        spacy_vocab: Vocabulary = spacy_data['vocab']
            
        embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

        collector = H5pyCollector(f'{self.config.task_id}.train.h5py', 768)

        batch_size = 32
        total_size = len(train_data)
        for i in range(0, total_size, batch_size):
            sents = []
            for j in range(i, min(i + batch_size, total_size)):
                sents.append(allenutil.as_sentence(train_data[j]))
            collector.collect(np.array(embedder.encode(sents)))
        collector.close()
        
    def test_distance(self):
        embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
        index = build_faiss_index(f'{self.config.task_id}.train.h5py')
        
        df = pandas.read_csv(self.config.adv_data, sep='\t', quoting=csv.QUOTE_NONE)
        agg_D = []
        for rid in tqdm(range(df.shape[0])):
            raw = df.iloc[rid]['raw']
            adv = df.iloc[rid]['adv']
            if raw != adv:
                sent_embed = embedder.encode([raw, adv])
                D, _ = index.search(np.array(sent_embed), 3)
                agg_D.append(D.mean(axis=1))
        agg_D = np.array(agg_D)
        print(agg_D.mean(axis=0), agg_D.std(axis=0))
        print(sum(agg_D[:, 0] < agg_D[:, 1]), 'of', agg_D.shape[0])
        
    def test_ppl(self):
        en_lm = torch.hub.load('pytorch/fairseq', 
                               'transformer_lm.wmt19.en', 
                               tokenizer='moses', 
                               bpe='fastbpe')
        en_lm.eval()
        en_lm.cuda()
        
        df = pandas.read_csv(self.config.adv_data, sep='\t', quoting=csv.QUOTE_NONE)
        agg_ppls = []
        for rid in tqdm(range(df.shape[0])):
            raw = df.iloc[rid]['raw']
            adv = df.iloc[rid]['adv']
            if raw != adv:
                scores = en_lm.score([raw, adv])
                ppls = np.array([ele['positional_scores'].mean().neg().exp().item() for ele in scores])
                agg_ppls.append(ppls)
        agg_ppls = np.array(agg_ppls)
        print(agg_ppls.mean(axis=0), agg_ppls.std(axis=0))
        print(sum(agg_ppls[:, 0] < agg_ppls[:, 1]), 'of', agg_ppls.shape[0])

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
            f"{self.config.task_id}-{self.config.attack_vectors}.vec",
            lambda: _read_pretrained_embeddings_file(WORD2VECS[self.config.attack_vectors],
                                                     embedding_dim=EMBED_DIM[self.config.
                                                                             attack_vectors],
                                                     vocab=spacy_vocab,
                                                     namespace="tokens"), True)

        if self.config.attack_gen_adv:
            f_adv = open(f"nogit/{self.config.model_name}.{self.config.attack_method}.adv.tsv", 'w')
            f_adv.write("raw\tadv\tlabel\n")

        if self.config.attack_gen_aug:
            f_aug = open(f"nogit/{self.config.model_name}.{self.config.attack_method}.aug.tsv", 'w')
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
        general_kwargs = { 
            "ignore_tokens": forbidden_words,
            "forbidden_tokens": forbidden_words,
            "max_change_num_or_ratio": 0.25
        }
        blackbox_kwargs = {
            "vocab": spacy_vocab,
            "token_embedding": spacy_weight
        }
        if self.config.attack_method == 'pgd':
            attacker = PGD(self.predictor,
                           step_size = 100.,
                           max_step = 20,
                           iter_change_num = 1,
                           **general_kwargs)
        elif self.config.attack_method == 'hotflip':
            attacker = HotFlip(self.predictor, **general_kwargs)
        elif self.config.attack_method == 'bruteforce':
            attacker = BruteForce(self.predictor, 
                                  policy=EmbeddingPolicy(measure='euc', topk=10, rho=None),
                                  **general_kwargs, **blackbox_kwargs)
        elif self.config.attack_method == 'pwws':
            attacker = PWWS(self.predictor,
                            policy=EmbeddingPolicy(measure='euc', topk=10, rho=None),
#                             policy=SynonymPolicy(),
                            **general_kwargs, **blackbox_kwargs)
        elif self.config.attack_method == 'genetic':
            attacker = Genetic(self.predictor, 
                               num_generation = 10,
                               num_population = 20,
                               policy=EmbeddingPolicy(measure='euc', topk=10, rho=None),
                               lm_topk = 4,
                               **general_kwargs, **blackbox_kwargs)
        else:
            raise Exception()
            
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
            adv_text = raw_text = allenutil.as_sentence(data_to_attack[i])
            raw_pred = np.argmax(self.predictor.predict(raw_text)['probs'])
            raw_label = data_to_attack[i]['label'].label
            # Only attack correct instance
            if raw_pred == raw_label:
                if self.config.arch == 'bert':
                    field_to_change = 'berty_tokens'
                elif self.config.arch == 'lstm':
                    field_to_change = 'sent'

                result = attacker.attack_from_json({field_to_change: raw_text},
                                                   field_to_change=field_to_change)

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
                        log("Aggregated generation", agg.mean("generation"))
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
