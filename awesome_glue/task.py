import csv
import sys
from collections import Counter
from functools import partial
import json

import numpy as np
import pandas
import torch
from allennlp.data.dataloader import DataLoader, allennlp_collate
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.vocabulary import Vocabulary
from allennlpx.training.adv_trainer import AdvTrainer
from allennlp.training.util import evaluate
from tqdm import tqdm
import pathlib
import shutil

from allennlpx import allenutil
from allennlpx.interpret.attackers import BruteForce, Genetic, HotFlip, PWWS
from allennlpx.interpret.attackers.searchers import EmbeddingSearcher, CachedWordSearcher, EmbeddingNbrUtil, WordIndexSearcher
from allennlpx.predictors import TextClassifierPredictor, BiTextClassifierPredictor
from awesome_glue.config import Config
from awesome_glue.vanilla_classifier import Classifier
from awesome_glue.esim import ESIM
from awesome_glue import embed_util
from awesome_glue.bert_classifier import BertClassifier
from awesome_glue.task_specs import TASK_SPECS, is_sentence_pair
from awesome_glue.transforms import (transform_collate,
                                     parse_transform_fn_from_args)
from awesome_glue.utils import (AttackMetric, FreqUtil, set_environments, WarmupCallback,
                                text_diff, get_neighbour_matrix, read_hyper)
from luna import flt2str, ram_write, ram_set_flag, ram_reset_flag
from luna.logging import log
from luna.public import Aggregator, auto_create, numpy_seed
from luna.pytorch import set_seed
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from allennlpx.training.checkpointer import CheckpointerX
from allennlpx.training import adv_utils
import logging
from awesome_glue.data_loader import load_banned_words, load_data
from awesome_glue.weighted_util import WeightedHull, SameAlphaHull, DecayAlphaHull
from awesome_glue.biboe import BiBOE
from awesome_glue.decom_att import DecomposableAttention
from awesome_glue.weighted_embedding import WeightedEmbedding
from typing import Dict, Any
from allennlpx.training.adv_trainer import EpochCallback


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
            
        if self.config.arch == "bert":
            self.vocab = embed_util.get_bert_vocab()

        # Build the model
        embed_args = {
            "vocab": self.vocab,
            "pretrain": config.pretrain,
            "finetune": config.finetune,
            "cache_embed_path": f"{config.task_id}-{config.pretrain}.vec"
        }
        
        if config.embed == "":
            if self.config.arch != "bert":
                token_embedder = embed_util.build_embedding(**embed_args)
        elif config.embed == "w":
            hull_args = {
                "alpha": config.dir_alpha,
                "nbr_file": "external_data/ibp-nbrs.json",
                "vocab": self.vocab,
                "nbr_num": config.nbr_num,
                "second_order": config.second_order
            }
            if config.second_order and 0.0 < config.dir_decay < 1.0:
                hull = DecayAlphaHull.build(
                    **hull_args,
                    decay=config.dir_decay,
                )
            else:
                hull = SameAlphaHull.build(**hull_args)
            print(f'Using {type(hull)} with second_order={config.second_order}')             
            
            if self.config.arch != "bert":
                token_embedder = embed_util.build_weighted_embedding(
                    **embed_args,
                    hull=hull
                )
        else:
            raise Exception()
        
        arch_args = {
            "vocab": self.vocab,
            "token_embedder": token_embedder,
            "num_labels": TASK_SPECS[config.task_id]['num_labels']
        }
        if config.arch in ['boe', 'cnn', 'lstm']:
            self.model = Classifier(arch=config.arch, pool=config.pool, **arch_args)
        elif config.arch == 'biboe':
            self.model = BiBOE(**arch_args, pool=config.pool)
        elif config.arch == 'datt':
            self.model = DecomposableAttention(**arch_args)
        elif config.arch == 'esim':
            self.model = ESIM(**arch_args)
        elif config.arch == 'bert':
            self.model = BertClassifier(
                self.vocab,
                num_labels=TASK_SPECS[config.task_id]['num_labels'])
            if config.embed == "w":
                bert_embeddings = self.model.bert_embedder.transformer_model.embeddings.word_embeddings
                bert_embeddings.word_embeddings = WeightedEmbedding(
                    num_embeddings=self.vocab.get_vocab_size('tokens'),
                    embedding_dim=768,
                    weight=bert_embeddings.weight,
                    hull=hull,
                    sparse=False,
                    trainable=True
                )
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

        if self.config.task_id == 'SST':
            self.train_data = self.dev_data
            self.config.attack_data_split = 'dev'

    def train(self):
        ram_write("adjust_point", self.config.adjust_point)
        # ram_write('dist_reg', self.config.dist_reg)
        read_hyper_ = partial(read_hyper, self.config.task_id, self.config.arch)
        num_epochs = int(read_hyper_("num_epochs"))
        batch_size = int(read_hyper_("batch_size"))
        if self.config.arch == 'bert' and self.config.embed == 'w':
            num_epochs = 6
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
        if self.config.arch == 'bert':
            model_vocab = embed_util.get_bert_vocab()
        else:
            model_vocab = self.vocab
        # yapf: disable
        adv_field = 'sent2' if is_sentence_pair(self.config.task_id) else 'sent'
        policy_args = {
            "adv_iteration": self.config.adv_iter,
            "replace_num": self.config.adv_replace_num,
            "searcher": WordIndexSearcher(
                CachedWordSearcher(
                    "external_data/ibp-nbrs.json",
                    model_vocab.get_token_to_index_vocabulary("tokens"),
                    second_order=False
                ),
                word2idx=model_vocab.get_token_index,
                idx2word=model_vocab.get_token_from_index,
            ),
            'adv_field': adv_field
        }
        # yapf: enable
        if self.config.adv_policy == 'hot':
            if is_sentence_pair(self.config.task_id):
                policy_args['forward_order'] = 1
            adv_policy = adv_utils.HotFlipPolicy(**policy_args)
        elif self.config.adv_policy == 'rad':
            adv_policy = adv_utils.RandomNeighbourPolicy(**policy_args)
        elif self.config.adv_policy == 'diy':
            adv_policy = adv_utils.DoItYourselfPolicy(
                self.config.adv_iter, adv_field, self.config.adv_step)
        else:
            adv_policy = adv_utils.NoPolicy

        # A collate_fn will do some transformation an instance before
        # fed into a model. If we want to train a model with some transformations
        # such as cropping/DAE, we can modify code here. e.g.,
        # collate_fn = partial(transform_collate, self.vocab, self.reader, Crop(0.3))
        collate_fn = allennlp_collate
        train_data_sampler = BucketBatchSampler(
            data_source=self.train_data,
            batch_size=batch_size,
        )
        # Set callbacks
        

        if self.config.task_id == 'SNLI':
            epoch_callbacks = [WarmupCallback(2)]
            # epoch_callbacks = []
            if self.config.model_pretrain != "":
                if self.config.model_pretrain == 'auto':
                    self.config.model_pretrain = {
                        "biboe": "SNLI-fix-biboe-sum",
                        "datt": "SNLI-fix-datt"
                    }[self.config.arch]
                logger.warning(f"Try loading weights from pretrained model {self.config.model_pretrain}")
                pretrain_ckpter = CheckpointerX(f"saved/models/{self.config.model_pretrain}")
                self.model.load_state_dict(pretrain_ckpter.best_model_state())
        else:
            epoch_callbacks = []
        # epoch_callbacks = []
        batch_callbacks = []

        opt = self.model.get_optimizer()
        # from allennlp.training.learning_rate_schedulers import SlantedTriangular
        # scl = SlantedTriangular(opt, num_epochs, len(self.train_data) // batch_size)

        trainer = AdvTrainer(
            model=self.model,
            optimizer=opt,
            # learning_rate_scheduler=scl,
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
        ckpter = CheckpointerX(f'saved/models/{self.config.model_name}')
        latest_epoch = 3 if self.config.arch == 'bert' else 10
        model_path = ckpter.find_latest_best_checkpoint(latest_epoch, 'validation_accuracy')[0]
        # model_path = f'saved/models/{self.config.model_name}/best.th'
        print(f'Load model from {model_path}')
        sys.stdout.flush()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    @torch.no_grad()
    def evaluate_model(self):
        self.from_pretrained()
        print(evaluate(self.model, DataLoader(self.dev_data, 32), 0, None))

    @torch.no_grad()
    def evaluate_predictor(self):
        self.from_pretrained()
        if self.config.eval_data_split == 'dev':
            eval_data = self.dev_data
        elif self.config.eval_data_split == 'test':
            eval_data = self.test_data
        metric = CategoricalAccuracy()
        batch_size = 32
        if self.config.eval_size == -1:
            total_size = len(eval_data)
        else:
            total_size = min(len(eval_data), self.config.eval_size)
        bar = tqdm(range(0, total_size, batch_size))
        for bid in bar:
            instances = [
                eval_data[i]
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
            bar.set_description("{:5.2f}".format(metric.get_metric()))
        print(f"Evaluate on {self.config.eval_data_split}, the result is ", metric.get_metric())

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

    def downsample_data_to_attack(self):
        # Set up the data to attack
        if self.config.attack_data_split == 'train':
            data_to_attack = self.train_data
        elif self.config.attack_data_split == 'dev':
            data_to_attack = self.dev_data
        elif self.config.attack_data_split == 'test':
            data_to_attack = self.test_data
        if is_sentence_pair(self.config.task_id):
            field_to_change = 'sent2'
        else:
            field_to_change = 'sent'
        data_to_attack = list(
            filter(lambda x: len(x[field_to_change].tokens) < 300,
                   data_to_attack))

        if self.config.attack_size != -1:
            with numpy_seed(19491001):
                idxes = np.random.permutation(len(data_to_attack))
                data_to_attack = [data_to_attack[i] for i in idxes[:self.config.attack_size]]
        return data_to_attack

    def attack(self):
        self.from_pretrained()

        data_to_attack = self.downsample_data_to_attack()
        if is_sentence_pair(self.config.task_id):
            field_to_change = 'sent2'
        else:
            field_to_change = 'sent'

        # Speed up the predictor
        if self.config.arch != 'bert':
            self.predictor.set_max_tokens(360000)
            if self.config.nbr_2nd[1] == '2':
                if self.config.nbr_num <= 12:
                    self.predictor.set_max_tokens(360000)
                elif self.config.nbr_num <= 24:
                    self.predictor.set_max_tokens(120000)
                else:
                    self.predictor.set_max_tokens(90000)
            if self.config.poor_gpu:
                self.predictor.set_max_tokens(30000)
        else:
            self.predictor.set_max_tokens(60000)
            
        # Set up the attacker
        # Whatever bert/non-bert model, we use the spacy vocab
        spacy_vocab = load_data(self.config.task_id, "spacy")['vocab']
        searcher = CachedWordSearcher(
            "external_data/ibp-nbrs.json",
            spacy_vocab.get_token_to_index_vocabulary("tokens")
        )
        forbidden_words = load_banned_words(self.config.task_id)
        # forbidden_words += stopwords.words("english")
        general_kwargs = {
            "ignore_tokens": forbidden_words,
            "forbidden_tokens": forbidden_words,
            "max_change_num_or_ratio": 0.15,
            "field_to_change": field_to_change,
            "field_to_attack": 'label',
            "use_bert": self.config.arch == 'bert',
            "searcher": searcher
        }
        if self.config.attack_method == 'hotflip':
            attacker = HotFlip(self.predictor, **general_kwargs)
        elif self.config.attack_method == 'bruteforce':
            attacker = BruteForce(self.predictor, **general_kwargs)
        elif self.config.attack_method == 'pwws':
            attacker = PWWS(self.predictor, **general_kwargs)
        elif self.config.attack_method in ['genetic', 'genetic_nolm']:
            if self.config.attack_method == "genetic":
                lm_constraints = json.load(open(f"external_data/ibp-nbrs.{self.config.task_id}.{self.config.attack_data_split}.lm.json"))
            else:
                lm_constraints = None
            attacker = Genetic(self.predictor,
                               num_generation=40,
                               num_population=60,
                               lm_topk=-1,
                               lm_constraints=lm_constraints,
                               **general_kwargs)
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
            log(f"Attacking instance {i}...")
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
                
                if "generation" in result:
                    print("stop at generation", result['generation'])
                # sanity check: in case of failure, the changed num should be close to
                # the max change num.
                if not result['success']:
                    diff = text_diff(result['raw'], result['adv'])
                    print('[Fail statistics]', diff)

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

                    log("Avg.change#", ":5.2f".format(agg.mean("change_num")),
                        "Avg.change%", ":5.2f".format(100 * agg.mean("change_ratio")))
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
                log(f"Aggregated metric: " +  
                    # "[loose] {loose_metric}" +
                    f"[strict] {strict_metric}")
                # yapf:enable
            else:
                log("Skipping the current instance since the predicted label is wrong.")
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
