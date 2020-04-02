import csv
import sys
from collections import Counter
from functools import partial
from copy import deepcopy

import numpy as np
import pandas
import torch
from typing import Dict, List, Any
from allennlp.data.dataloader import DataLoader, allennlp_collate
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlpx.training.adv_trainer import AdvTrainer, EpochCallback, BatchCallback
from allennlp.training.util import evaluate
from nltk.corpus import stopwords
from tqdm import tqdm
import pathlib
import shutil
from allennlp.training.checkpointer import Checkpointer

from allennlpx import allenutil
from allennlpx.data.dataset_readers import BertyTSVReader, SpacyTSVReader
from allennlpx.interpret.attackers.attacker import DEFAULT_IGNORE_TOKENS
from allennlpx.interpret.attackers import BruteForce, Genetic, HotFlip, PGD, PWWS
from allennlpx.interpret.attackers.policies import (CandidatePolicy,
                                                    EmbeddingPolicy,
                                                    SpecifiedPolicy,
                                                    SynonymPolicy,
                                                    UnconstrainedPolicy)
from allennlpx.modules.knn_utils import H5pyCollector, build_faiss_index
from allennlpx.modules.token_embedders.embedding import \
    _read_pretrained_embeddings_file
from allennlpx.modules.token_embedders.graph_funcs import MeanAggregator, PoolingAggregator
from allennlpx.predictors import TextClassifierPredictor, BiTextClassifierPredictor
from awesome_glue.config import Config
from awesome_glue.vanilla_classifier import Classifier
from awesome_glue.esim import ESIM
from awesome_glue import embed_util
from awesome_glue.bert_classifier import BertClassifier
from awesome_glue.task_specs import TASK_SPECS, is_sentence_pair, is_str_label
from awesome_glue.transforms import (BackTrans, DAE, BertAug, Crop, EmbedAug,
                                     Identity, RandDrop, SynAug,
                                     transform_collate,
                                     parse_transform_fn_from_args)
from awesome_glue.utils import (AttackMetric, FreqUtil, set_environments,
                                text_diff, get_neighbours, DirichletAnnealing, 
                                AnnealingTemperature, read_hyper)
from luna import flt2str, ram_write, ram_read
from luna.logging import log
from luna.public import Aggregator, auto_create
from luna.pytorch import set_seed
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from allennlpx.training import adv_utils
from allennlpx.interpret.attackers.searchers import EmbeddingSearcher
import logging
from awesome_glue.data_loader import load_data, load_banned_words
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlpx.modules.token_embedders.dirichlet_embedding import DirichletEmbedding
from luna import shutdown_logging

logging.getLogger('transformers').setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

set_environments()


class Task:
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        loaded_data = load_data(config.task_id, config.tokenizer)
        self.reader = loaded_data['reader']
        self.train_data, self.dev_data, self.test_data = loaded_data['data']
        self.vocab: Vocabulary = loaded_data['vocab']

        # Build the model
        embed_args = {
            "vocab": self.vocab,
            "pretrain": config.pretrain,
            "cache_embed_path": f"{config.task_id}-{config.pretrain}.vec"
        }
        if config.arch != 'bert':
            if config.embed == "":
                token_embedder = embed_util.build_embedding(**embed_args)
            elif config.embed == "d":
                _, spacy_vec = self.get_spacy_vocab_and_vec()
                neighbours, nbr_mask = get_neighbours(spacy_vec,
                                                      return_edges=False)
                token_embedder = embed_util.build_dirichlet_embedding(
                    **embed_args,
                    temperature=config.dir_temp,
                    neighbours=neighbours.cuda(),
                    nbr_mask=nbr_mask.cuda())
            elif config.embed == "g":
                _, spacy_vec = self.get_spacy_vocab_and_vec()
                edges = get_neighbours(spacy_vec, return_edges=True)
                token_embedder = embed_util.build_graph_embedding(
                    **embed_args,
                    gnn={
                        "mean": MeanAggregator(300),
                        "pool": PoolingAggregator(300)
                    }[config.gnn_type],
                    edges=edges.cuda(),
                    hop=config.gnn_hop)
            else:
                raise Exception()

        if config.arch in ['boe', 'cnn', 'lstm']:
            self.model = Classifier(
                vocab=self.vocab,
                token_embedder=token_embedder,
                arch=config.arch,
                num_labels=TASK_SPECS[config.task_id]['num_labels'])
        elif config.arch == 'esim':
            self.model = ESIM(
                vocab=self.vocab,
                token_embedder=token_embedder,
                num_labels=TASK_SPECS[config.task_id]['num_labels'])
        elif config.arch == 'bert':
            self.model = BertClassifier(
                self.vocab,
                num_labels=TASK_SPECS[config.task_id]['num_labels'])
            if config.embed == "d":
                bert_embeddings = self.model.bert_embedder.transformer_model.embeddings
                bert_vocab, bert_vec = embed_util.build_bert_vocab_and_vec(
                    "counter")
                neighbours, nbr_mask = get_neighbours(bert_vec,
                                                      return_edges=False)
                token_embedder = DirichletEmbedding(
                    num_embeddings=bert_vocab.get_vocab_size('tokens'),
                    embedding_dim=768,
                    weight=bert_embeddings.word_embeddings.weight,
                    temperature=config.dir_temp,
                    neighbours=neighbours.cuda(),
                    nbr_mask=nbr_mask.cuda(),
                    sparse=False,
                    trainable=True)
                bert_embeddings.word_embeddings = token_embedder
        self.model.cuda()

        # The predictor is a wrapper of the model.
        # It is slightly different from the predictor provided by AllenNLP.
        # With the predictor, we can do some tricky things before/after feeding
        # instances into a model, such as:
        # - do some input transformation (random drop, word augmentation, etc.)
        # - ensemble several models
        # Note that the attacker receives a predictor as the proxy of the model,
        # which allows for test-time attacks.
        if is_sentence_pair(self.config.task_id):
            self.predictor = BiTextClassifierPredictor(self.model,
                                                       self.reader,
                                                       key1='sent1',
                                                       key2='sent2')
        else:
            self.predictor = TextClassifierPredictor(self.model,
                                                     self.reader,
                                                     key='sent')

        # list[str] -> list[str]
        transform_fn = parse_transform_fn_from_args(
            self.config.pred_transform, self.config.pred_transform_args)

        self.predictor.set_ensemble_num(self.config.pred_ensemble)
        self.predictor.set_transform_fn(transform_fn)
        if is_sentence_pair(self.config.task_id):
            self.predictor.set_transform_field("sent2")
        else:
            self.predictor.set_transform_field("sent")

    def train(self):
        read_hyper_ = partial(read_hyper, self.config.task_id, self.config.arch)
        num_epochs = int(read_hyper_("num_epochs"))
        batch_size = int(read_hyper_("batch_size"))
        if self.config.arch == 'bert' and self.config.embed == 'd':
            num_epochs = 8
        logger.info(f"num_epochs: {num_epochs}, batch_size: {batch_size}")

        if self.config.model_name == 'tmp':
            p = pathlib.Path('saved/models/tmp')
            if p.exists():
                shutil.rmtree(p)

        # Maybe we will do some data augmentation here.
        if self.config.aug_data != '':
            log(f'Augment data from {self.config.aug_data}')
            aug_data = auto_create(
                f"{self.config.task_id}.{self.config.arch}.aug",
                lambda: self.reader.read(self.config.aug_data),
                cache=True)
            self.train_data.instances.extend(aug_data.instances)

        # Set up the adversarial training policy
        if self.config.adv_constraint:
            # We use an external weight to generate candidates for a word to replace.
            # The external weight must be corresponding to the internal weight!
            if self.config.arch == 'bert':
                adv_vocab, adv_weight = embed_util.build_bert_vocab_and_vec('counter')
            else:
                adv_vocab, adv_weight = self.get_spacy_vocab_and_vec()
            searcher = EmbeddingSearcher(
                embed=adv_weight,
                idx2word=adv_vocab.get_token_from_index,
                word2idx=adv_vocab.get_token_index)
            searcher.pre_search('euc', 10)
        else:
            searcher = None
        # yapf: disable
        policy_args = {
            "adv_iteration": self.config.adv_iter,
            "replace_num": self.config.adv_replace_num,
            "searcher": searcher,
            'adv_field': 'sent2' if is_sentence_pair(self.config.task_id) else 'sent'
        }
        # yapf: enable
        if self.config.adv_policy == 'hot':
            if is_sentence_pair(self.config.task_id):
                policy_args['forward_order'] = 1
            adv_policy = adv_utils.HotFlipPolicy(**policy_args)
        elif self.config.adv_policy == 'rad':
            adv_policy = adv_utils.RandomNeighbourPolicy(**policy_args)
        else:
            adv_policy = adv_utils.NoPolicy

        # A collate_fn will do some transformation an instance before
        # fed into a model. If we want to train a model with some transformations
        # such as cropping/DAE, we can modify code here. e.g.,
        # collate_fn = partial(transform_collate, self.vocab, self.reader, Crop(0.3))
        collate_fn = allennlp_collate
        train_data_sampler = BucketBatchSampler(
            # HIGHLIGHT: 
            data_source=self.train_data,
            batch_size=batch_size,
        )
        # Set callbacks
        epoch_callbacks = []
        # if self.config.embed == 'd':
        #     epoch_callbacks.append(AnnealingTemperature(anneal_epoch_num=5))
        batch_callbacks = []
        if self.config.embed == 'd':
            batch_callbacks.append(DirichletAnnealing(
                anneal_epoch_num=4, batch_per_epoch=len(train_data_sampler)))
        trainer = AdvTrainer(
            model=self.model,
            optimizer=self.model.get_optimizer(),
            validation_metric='+accuracy',
            adv_policy=adv_policy,
            data_loader=DataLoader(
                self.train_data,
                batch_sampler=train_data_sampler,
                collate_fn=collate_fn,
            ),
            validation_data_loader=DataLoader(
                self.dev_data,
                batch_size=batch_size,
            ),
            num_epochs=num_epochs,
            patience=None,
            grad_clipping=1.,
            cuda_device=0,
            epoch_callbacks=epoch_callbacks,
            batch_callbacks=batch_callbacks,
            serialization_dir=f'saved/models/{self.config.model_name}',
            num_serialized_models_to_keep=20)
        trainer.train()

    def from_pretrained(self):
        # ckpter = Checkpointer(f'saved/models/{self.config.model_name}'')
        # model_path = ckpter.find_latest_checkpoint()[0]
        model_path = f'saved/models/{self.config.model_name}/best.th'
        print(f'Load model from {model_path}')
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    @torch.no_grad()
    def evaluate_model(self):
        self.from_pretrained()
        print(evaluate(self.model, DataLoader(self.dev_data, 32), 0, None))

    @torch.no_grad()
    def evaluate_predictor(self):
        self.from_pretrained()
        metric = CategoricalAccuracy()
        batch_size = 32
        total_size = len(self.dev_data)
        for bid in tqdm(range(0, total_size, batch_size)):
            instances = [
                self.dev_data[i]
                for i in range(bid, min(bid + batch_size, total_size))
            ]
            outputs = self.predictor.predict_batch_instance(instances)
            preds, labels = [], []
            for inst, outp in zip(instances, outputs):
                preds.append([outp['probs']])
                label_idx = inst.fields['label'].label
                if isinstance(inst.fields['label'].label, str):
                    label_idx = self.vocab.get_token_index(label_idx, 'labels')
                labels.append([label_idx])
                metric(predictions=torch.tensor(preds),
                       gold_labels=torch.tensor(labels))
        print(metric.get_metric())

    @torch.no_grad()
    def transfer_attack(self):
        self.from_pretrained()
        set_seed(11221)
        df = pandas.read_csv(self.config.adv_data,
                             sep='\t',
                             quoting=csv.QUOTE_NONE)
        attack_metric = AttackMetric()

        for rid in tqdm(range(df.shape[0])):
            raw = df.iloc[rid]['raw']
            adv = df.iloc[rid]['adv']

            results = self.predictor.predict_batch_instance([
                self.reader.text_to_instance(raw),
                self.reader.text_to_instance(adv)
            ])

            raw_pred = np.argmax(results[0]['probs'])
            adv_pred = np.argmax(results[1]['probs'])

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

        # Set up the data to attack
        if self.config.attack_data_split == 'train':
            data_to_attack = self.train_data
        elif self.config.attack_data_split == 'dev':
            data_to_attack = self.dev_data
        if is_sentence_pair(self.config.task_id):
            field_to_change = 'sent2'
        else:
            field_to_change = 'sent'
        data_to_attack = list(
            filter(lambda x: len(x[field_to_change].tokens) < 300,
                   data_to_attack))

        if self.config.attack_size != -1:
            data_to_attack = data_to_attack[:self.config.attack_size]

        # Set up the attacker
        forbidden_words = load_banned_words(self.config.task_id)
        # forbidden_words += stopwords.words("english")
        general_kwargs = {
            "ignore_tokens": forbidden_words,
            "forbidden_tokens": forbidden_words,
            "max_change_num_or_ratio": 0.15,
            "field_to_change": field_to_change,
            "field_to_attack": 'label',
            "use_bert": self.config.arch == 'bert',
            "policy": EmbeddingPolicy(measure='euc', topk=10, rho=None),
        }
        spacy_vocab, spacy_weight = self.get_spacy_vocab_and_vec()
        blackbox_kwargs = {
            "vocab": spacy_vocab,
            "token_embedding": spacy_weight
        }
        if self.config.attack_method == 'hotflip':
            attacker = HotFlip(self.predictor, **general_kwargs)
        elif self.config.attack_method == 'bruteforce':
            attacker = BruteForce(self.predictor, **general_kwargs, **blackbox_kwargs)
        elif self.config.attack_method == 'pwws':
            attacker = PWWS(self.predictor, **general_kwargs, **blackbox_kwargs)
        elif self.config.attack_method == 'genetic':
            attacker = Genetic(self.predictor,
                               num_generation=20,
                               num_population=40,
                               lm_topk=-1,
                               **general_kwargs,
                               **blackbox_kwargs)
        else:
            raise Exception()

        # Start attacking
        if self.config.attack_gen_adv:
            f_adv = open(f"nogit/{self.config.model_name}.{self.config.attack_method}.adv.tsv", 'w')
            f_adv.write("raw\tadv\tlabel\n")
        strict_metric = AttackMetric()
        loose_metric = AttackMetric()
        agg = Aggregator()
        raw_counter = Counter()
        adv_counter = Counter()
        for i in tqdm(range(len(data_to_attack))):
            raw_json = allenutil.as_json(data_to_attack[i])
            adv_json = raw_json.copy()

            raw_probs = self.predictor.predict_json(raw_json)['probs']
            raw_pred = np.argmax(raw_probs)
            raw_label = data_to_attack[i]['label'].label
            if isinstance(raw_label, str):
                raw_label = self.vocab.get_token_index(raw_label, 'labels')

            # Only attack correct instance
            if raw_pred == raw_label:
                # yapf:disable
                result = attacker.attack_from_json(raw_json)
                adv_json[field_to_change] = allenutil.as_sentence(result['adv'])

                # Count
                if result['success']:
                    diff = text_diff(result['raw'], result['adv'])
                    raw_counter.update(diff['a_changes'])
                    adv_counter.update(diff['b_changes'])
                    to_aggregate = [('change_num', diff['change_num']),
                                    ('change_ratio', diff['change_ratio'])]
                    if "generation" in result:
                        to_aggregate.append(('generation', result['generation']))
                    agg.aggregate(*to_aggregate)
                    log("[raw]", raw_json, "\n[prob]", flt2str(raw_probs, cat=', '))
                    log("[adv]", adv_json, '\n[prob]', flt2str(result['outputs']['probs'], cat=', '))
                    if "changed" in result:
                        log("[changed]", result['changed'])
                    log()

                    log("Avg.change#", round(agg.mean("change_num"), 2),
                        "Avg.change%", round(100 * agg.mean("change_ratio"), 2))
                    if "generation" in result:
                        log("Avg.gen#", agg.mean("generation"))

                # Strict metric: An attacker thinks that it attacks successfully.
                if result['success']:
                    strict_metric.succeed()
                else:
                    strict_metric.fail()

                # Loose metric: Passing the adversarial example to the model will
                # have a different result. (Since there may be randomness in the model.)
                adv_probs = self.predictor.predict_json(adv_json)['probs']
                adv_pred = np.argmax(adv_probs)
                if raw_pred != adv_pred:
                    loose_metric.succeed()
                else:
                    loose_metric.fail()
                log(f"Aggregated metric: [loose] {loose_metric} [strict] {strict_metric}")
                # yapf:enable
            else:
                loose_metric.escape()
                strict_metric.escape()

            if self.config.attack_gen_adv:
                f_adv.write(
                    f"{raw_json[field_to_change]}\t{adv_json[field_to_change]}\t{raw_label}\n"
                )
            sys.stdout.flush()

        if self.config.attack_gen_adv:
            f_adv.close()

        # yapf:disable
        print("Statistics of changed words:")
        print(">> [raw] ", raw_counter.most_common())
        print(">> [adv] ", adv_counter.most_common())
        print("Overall:")
        print("Avg.change#:", round(agg.mean("change_num"), 2) if agg.has_key("change_num") else '-',
              "Avg.change%:", round(100 * agg.mean("change_ratio"), 2) if agg.has_key("change_ratio") else '-',
              "[loose] Accu.before%:", round(loose_metric.accuracy_before_attack, 2),
              "Accu.after%:", round(loose_metric.accuracy_after_attack, 2),
              "[strict] Accu.before%:", round(strict_metric.accuracy_before_attack, 2),
              "Accu.after%:", round(strict_metric.accuracy_after_attack, 2))
        # yapf:enable

    def get_spacy_vocab_and_vec(self):
        if self.config.tokenizer != 'spacy':
            spacy_data = load_data(self.config.task_id, "spacy")
            spacy_vocab: Vocabulary = spacy_data['vocab']
        else:
            spacy_vocab = self.vocab
        spacy_weight = embed_util.read_weight(
            spacy_vocab, self.config.attack_vectors,
            f"{self.config.task_id}-{self.config.attack_vectors}.vec")
        return spacy_vocab, spacy_weight

#     def build_manifold(self):
#         spacy_data = load_data(self.config.task_id, "spacy")
#         train_data, dev_data, _ = spacy_data['data']
#         if self.config.task_id == 'SST':
#             train_data = list(filter(lambda x: len(x["sent"].tokens) > 15, train_data))
#         spacy_vocab: Vocabulary = spacy_data['vocab']

#         embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

#         collector = H5pyCollector(f'{self.config.task_id}.train.h5py', 768)

#         batch_size = 32
#         total_size = len(train_data)
#         for i in range(0, total_size, batch_size):
#             sents = []
#             for j in range(i, min(i + batch_size, total_size)):
#                 sents.append(allenutil.as_sentence(train_data[j]))
#             collector.collect(np.array(embedder.encode(sents)))
#         collector.close()

#     def test_distance(self):
#         embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
#         index = build_faiss_index(f'{self.config.task_id}.train.h5py')

#         df = pandas.read_csv(self.config.adv_data, sep='\t', quoting=csv.QUOTE_NONE)
#         agg_D = []
#         for rid in tqdm(range(df.shape[0])):
#             raw = df.iloc[rid]['raw']
#             adv = df.iloc[rid]['adv']
#             if raw != adv:
#                 sent_embed = embedder.encode([raw, adv])
#                 D, _ = index.search(np.array(sent_embed), 3)
#                 agg_D.append(D.mean(axis=1))
#         agg_D = np.array(agg_D)
#         print(agg_D.mean(axis=0), agg_D.std(axis=0))
#         print(sum(agg_D[:, 0] < agg_D[:, 1]), 'of', agg_D.shape[0])

#     def test_ppl(self):
#         en_lm = torch.hub.load('pytorch/fairseq',
#                                'transformer_lm.wmt19.en',
#                                tokenizer='moses',
#                                bpe='fastbpe')
#         en_lm.eval()
#         en_lm.cuda()

#         df = pandas.read_csv(self.config.adv_data, sep='\t', quoting=csv.QUOTE_NONE)
#         agg_ppls = []
#         for rid in tqdm(range(df.shape[0])):
#             raw = df.iloc[rid]['raw']
#             adv = df.iloc[rid]['adv']
#             if raw != adv:
#                 scores = en_lm.score([raw, adv])
#                 ppls = np.array(
#                     [ele['positional_scores'].mean().neg().exp().item() for ele in scores])
#                 agg_ppls.append(ppls)
#         agg_ppls = np.array(agg_ppls)
#         print(agg_ppls.mean(axis=0), agg_ppls.std(axis=0))
#         print(sum(agg_ppls[:, 0] < agg_ppls[:, 1]), 'of', agg_ppls.shape[0])
