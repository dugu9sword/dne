import csv
from collections import Counter, defaultdict
from functools import partial

import numpy as np
import pandas
import torch
from allennlp.data.dataloader import DataLoader, allennlp_collate
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.training.learning_rate_schedulers.slanted_triangular import \
    SlantedTriangular
from allennlp.training.trainer import Trainer
from allennlpx.training.adv_trainer import AdvTrainer
from allennlpx.training.vanilla_trainer import VanTrainer
from allennlp.training.util import evaluate
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from allennlpx import allenutil
from allennlpx.data.dataset_readers.berty_tsv import BertyTSVReader
from allennlpx.data.dataset_readers.spacy_tsv import SpacyTSVReader
from allennlpx.interpret.attackers.attacker import DEFAULT_IGNORE_TOKENS
from allennlpx.interpret.attackers.bruteforce import BruteForce
from allennlpx.interpret.attackers.genetic import Genetic
from allennlpx.interpret.attackers.hotflip import HotFlip
from allennlpx.interpret.attackers.pgd import PGD
from allennlpx.interpret.attackers.policies import (CandidatePolicy, EmbeddingPolicy,
                                                    SpecifiedPolicy, SynonymPolicy,
                                                    UnconstrainedPolicy)
from allennlpx.interpret.attackers.pwws import PWWS
from allennlpx.modules.knn_utils import H5pyCollector, build_faiss_index
from allennlpx.modules.token_embedders.embedding import \
    _read_pretrained_embeddings_file
from allennlpx.predictors.text_classifier import TextClassifierPredictor
from awesome_glue.config import Config
from awesome_glue.models.bert_classifier import BertClassifier
from awesome_glue.models.lstm_classifier import LstmClassifier
from awesome_glue.models.graph_lstm_classifier import GraphLstmClassifier
from awesome_glue.task_specs import TASK_SPECS
from awesome_glue.transforms import (BackTrans, DAE, BertAug, Crop, EmbedAug, Identity, RandDrop, SynAug,
                                     transform_collate)
from awesome_glue.utils import (EMBED_DIM, WORD2VECS, AttackMetric, FreqUtil, set_environments,
                                text_diff)
from luna import flt2str, ram_write
from luna.logging import log
from luna.public import Aggregator, auto_create
from luna.pytorch import set_seed
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from allennlpx.training import adv_utils
from allennlpx.interpret.attackers.cached_searcher import CachedIndexSearcher
from allennlpx.interpret.attackers.embedding_searcher import EmbeddingSearcher
import faiss


set_environments()


def load_data(task_id: str, tokenizer: str):
    spec = TASK_SPECS[task_id]

    reader = {
        'spacy': SpacyTSVReader,
        'bert': BertyTSVReader
    }[tokenizer](sent1_col=spec['sent1_col'],
                 sent2_col=spec['sent2_col'],
                 label_col=spec['label_col'],
                 skip_label_indexing=spec['skip_label_indexing'])

    def __load_data():
        train_data = reader.read(f'{spec["path"]}/train.tsv')
        dev_data = reader.read(f'{spec["path"]}/dev.tsv')
        test_data = reader.read(f'{spec["path"]}/test.tsv')
        _MIN_COUNT = 3
        vocab = Vocabulary.from_instances(
            train_data,
            min_count = {"tokens": _MIN_COUNT, "sent": _MIN_COUNT,
                        "sent1": _MIN_COUNT, "sent2": _MIN_COUNT},
        )
        train_data.index_with(vocab)
        dev_data.index_with(vocab)
        test_data.index_with(vocab)
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
            self.model = LstmClassifier(self.vocab, TASK_SPECS[config.task_id]['num_labels'],
                                        config.pretrain,
                                        0.0 if config.task_id in ['SST', 'TOY'] else 0.3,
                                        config.finetunable,
                                        f"{config.task_id}-{config.pretrain}.vec").cuda()
        elif config.arch == 'glstm':
            _, spacy_vec = self.get_spacy_vocab_and_vec()
            index = faiss.IndexFlatL2(spacy_vec.shape[1])
            res = faiss.StandardGpuResources()  # use a single GPU
            index = faiss.index_cpu_to_gpu(res, 0, index)
            embed =  spacy_vec.cpu().numpy()
            index.add(embed)
            _, I = index.search(embed, k=10)
#             import ipdb; ipdb.set_trace()
            
            self.model = GraphLstmClassifier(vocab=self.vocab, 
                                             num_labels=TASK_SPECS[config.task_id]['num_labels'],
                                             pretrain=config.pretrain, 
                                             neighbours=torch.tensor(I).cuda(),
                                             word_drop_rate=0.0 if config.task_id in ['SST', 'TOY'] else 0.3,
                                             finetunable=config.finetunable,
                                             cache_embed_path=f"{config.task_id}-{config.pretrain}.vec").cuda()
        else:
            raise Exception

        self.predictor = TextClassifierPredictor(
            self.model, self.reader, key='sent' if self.config.arch != 'bert' else 'berty_tokens')

        # the code is a bullshit.
        _transform_fn = self.reader.transform_instances
        targ = self.config.pred_transform_args
        transform_fn = {
            "": lambda: lambda x: x,
            "identity": lambda: lambda x: x,
            "bt": lambda: partial(_transform_fn, BackTrans()),
            "dae": lambda: partial(_transform_fn, DAE()),
            "rand_drop": lambda: partial(_transform_fn, RandDrop(targ)),
            "embed_aug": lambda: partial(_transform_fn, EmbedAug(targ)),
            "syn_aug": lambda: partial(_transform_fn, SynAug(targ)),
            "bert_aug": lambda: partial(_transform_fn, BertAug(targ)),
            "crop": lambda: partial(_transform_fn, Crop(targ)),
        }[self.config.pred_transform]()
        self.predictor.set_ensemble_num(self.config.pred_ensemble)
        self.predictor.set_transform_fn(transform_fn)

    def train(self):
        num_epochs = 10
        pseudo_batch_size = 32
        accumulate_num = 1

        optimizer = self.model.get_optimizer()

        if self.config.aug_data != '':
            log(f'Augment data from {self.config.aug_data}')
            self.train_data.extend(self.reader.read(self.config.aug_data))


#         collate_fn = partial(transform_collate, self.vocab, self.reader, Crop(0.3))
        collate_fn = allennlp_collate
        if self.config.adv_constraint:
            # VERY IMPORTANT!
            # we use the spacy_weight here since during attacks we use an external weight.
            # but it is also possible to use the model's internal weight.
            # one thing is important: the weight must be corresponding to the vocab!
            _, spacy_weight = self.get_spacy_vocab_and_vec()
            searcher = EmbeddingSearcher(
                embed=spacy_weight,
                idx2word=self.vocab.get_token_from_index,
                word2idx=self.vocab.get_token_index
            )
        else:
            searcher = None
        adv_policy = adv_utils.HotFlipPolicy(
            normal_iteration=1,
            adv_iteration=self.config.adv_iter,
            replace_num=15,
            searcher=searcher,
        )
        trainer = AdvTrainer(
            model=self.model,
            optimizer=optimizer,
            validation_metric='+accuracy',
            adv_policy=adv_policy,
            # adv_policy = None,
            data_loader=DataLoader(
                self.train_data,
                batch_sampler=BucketBatchSampler(
                    data_source=self.train_data,
                    batch_size=pseudo_batch_size,
                ),
                collate_fn=collate_fn,
            ),
            validation_data_loader=DataLoader(
                self.dev_data,
                batch_size=pseudo_batch_size,
            ),
            num_epochs=num_epochs,
            patience=None,
            grad_clipping=1.,
            cuda_device=0,
            #   num_gradient_accumulation_steps=accumulate_num,
            serialization_dir=f'saved/models/{self.config.model_name}',
            num_serialized_models_to_keep=3)
        trainer.train()

    def from_pretrained(self):
        self.model.load_state_dict(torch.load(f'saved/models/{self.config.model_name}/best.th'))

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
                ppls = np.array(
                    [ele['positional_scores'].mean().neg().exp().item() for ele in scores])
                agg_ppls.append(ppls)
        agg_ppls = np.array(agg_ppls)
        print(agg_ppls.mean(axis=0), agg_ppls.std(axis=0))
        print(sum(agg_ppls[:, 0] < agg_ppls[:, 1]), 'of', agg_ppls.shape[0])

    @torch.no_grad()
    def evaluate_model(self):
        self.from_pretrained()
        self.model.eval()
        evaluate(self.model, DataLoader(self.dev_data, 32), 0, None)

    @torch.no_grad()
    def evaluate_predictor(self):
        self.from_pretrained()
        self.model.eval()
        metric = CategoricalAccuracy()
        batch_size = 32
        total_size = len(self.dev_data)
        for bid in tqdm(range(0, total_size, batch_size)):
            instances = [self.dev_data[i] for i in range(bid, min(bid + batch_size, total_size))]
            outputs = self.predictor.predict_batch_instance(instances)
            preds, labels = [], []
            for inst, outp in zip(instances, outputs):
                preds.append([outp['probs']])
                labels.append([inst.fields['label'].label])
                metric(predictions=torch.tensor(preds), gold_labels=torch.tensor(labels))
        print(metric.get_metric())

    @torch.no_grad()
    def transfer_attack(self):
        self.from_pretrained()
        self.model.eval()
        set_seed(11221)
        df = pandas.read_csv(self.config.adv_data, sep='\t', quoting=csv.QUOTE_NONE)
        attack_metric = AttackMetric()

        for rid in tqdm(range(df.shape[0])):
            raw = df.iloc[rid]['raw']
            adv = df.iloc[rid]['adv']

            #             print(text_diff(raw, adv))
            #             new_adv = []
            #             for wr, wa in zip(raw.split(" "), att.split(" ")):
            #                 if wr == wa:
            #                     new_att.append(wa)
            #                 else:
            #                     pass
            # #                     new_att.append()
            #             adv = " ".join(new_att)

            results = self.predictor.predict_batch_instance(
                [self.reader.text_to_instance(raw),
                 self.reader.text_to_instance(adv)])

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
                                         
    def get_spacy_vocab_and_vec(self):
        if self.config.tokenizer != 'spacy':
            spacy_data = load_data(self.config.task_id, "spacy")
            spacy_vocab: Vocabulary = spacy_data['vocab']
        else:
            spacy_vocab = self.vocab
        spacy_weight = auto_create(
            f"{self.config.task_id}-{self.config.attack_vectors}.vec", lambda:
            _read_pretrained_embeddings_file(WORD2VECS[self.config.attack_vectors],
                                             embedding_dim=EMBED_DIM[self.config.attack_vectors],
                                             vocab=spacy_vocab,
                                             namespace="tokens"), True)
        return spacy_vocab, spacy_weight

    def attack(self):
        self.from_pretrained()
        self.model.eval()

        spacy_vocab, spacy_weight = self.get_spacy_vocab_and_vec()

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

        forbidden_words = DEFAULT_IGNORE_TOKENS
        if 'banned_words' in TASK_SPECS[self.config.task_id]:
            forbidden_words.extend([
                line.rstrip('\n') for line in open(TASK_SPECS[self.config.task_id]['banned_words'])
            ])

        list(self.vocab.get_index_to_token_vocabulary().values())
        forbidden_words += stopwords.words("english")
        #         STOP_WORDS = stopwords.words("english")
        #         for ele in ['nor', 'above']:
        #             STOP_WORDS.remove(ele)
        #         for ele in STOP_WORDS:
        #             if "'" in ele:
        #                 STOP_WORDS.remove(ele)
        FreqUtil.topk_frequency(self.vocab, 100, 'least', forbidden_words)
        # self.predictor._model.cpu()
        general_kwargs = {
            "ignore_tokens": forbidden_words,
            "forbidden_tokens": forbidden_words,
            "max_change_num_or_ratio": 0.15
        }
        blackbox_kwargs = {"vocab": spacy_vocab, "token_embedding": spacy_weight}
        if self.config.attack_method == 'pgd':
            attacker = PGD(self.predictor,
                           step_size=100.,
                           max_step=20,
                           iter_change_num=1,
                           **general_kwargs)
        elif self.config.attack_method == 'hotflip':
            attacker = HotFlip(
                self.predictor,
                policy=EmbeddingPolicy(measure='euc', topk=10, rho=None),
                #                                policy=UnconstrainedPolicy(),
                **general_kwargs)
        elif self.config.attack_method == 'bruteforce':
            attacker = BruteForce(self.predictor,
                                  policy=EmbeddingPolicy(measure='euc', topk=10, rho=None),
                                  **general_kwargs,
                                  **blackbox_kwargs)
        elif self.config.attack_method == 'pwws':
            attacker = PWWS(
                self.predictor,
                #                 policy=SpecifiedPolicy(words=STOP_WORDS),
                policy=EmbeddingPolicy(measure='euc', topk=10, rho=None),
                #                             policy=SynonymPolicy(),
                **general_kwargs,
                **blackbox_kwargs)
        elif self.config.attack_method == 'genetic':
            attacker = Genetic(self.predictor,
                               num_generation=10,
                               num_population=20,
                               policy=EmbeddingPolicy(measure='euc', topk=10, rho=None),
                               lm_topk=4,
                               **general_kwargs,
                               **blackbox_kwargs)
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

        if self.config.arch == 'bert':
            field_name = 'berty_tokens'
        else:
            field_name = 'sent'
        data_to_attack = list(filter(lambda x: len(x[field_name].tokens) < 300, data_to_attack))

        if self.config.attack_size == -1:
            adv_number = len(data_to_attack)
        else:
            adv_number = self.config.attack_size
        data_to_attack = data_to_attack[:adv_number]

        attack_metric = AttackMetric()
        agg = Aggregator()
        raw_counter = Counter()
        adv_counter = Counter()
        for i in tqdm(range(adv_number)):
            raw_text = allenutil.as_sentence(data_to_attack[i])
            adv_text = None

            raw_pred = np.argmax(self.predictor.predict_instance(data_to_attack[i])['probs'])
            raw_label = data_to_attack[i]['label'].label
            # Only attack correct instance
            if raw_pred == raw_label:
                if self.config.arch == 'bert':
                    field_to_change = 'berty_tokens'
                elif self.config.arch in ['lstm', 'glstm' ]:
                    field_to_change = 'sent'

                result = attacker.attack_from_json({field_to_change: raw_text},
                                                   field_to_change=field_to_change)

                if result["success"] == 1:
                    diff = text_diff(result['raw'], result['adv'])
                    raw_counter.update(diff['a_changes'])
                    adv_counter.update(diff['b_changes'])
                    to_aggregate = [('change_num', diff['change_num']),
                                    ('change_ratio', diff['change_ratio'])]
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
                    log("Avg.change#", round(agg.mean("change_num"), 2), "Avg.change%",
                        round(100 * agg.mean("change_ratio"), 2))
                    if "generation" in result:
                        log("Aggregated generation", agg.mean("generation"))
                    log("Aggregated metric:", attack_metric)
                else:
                    attack_metric.fail()
            else:
                attack_metric.escape()

            if adv_text is None:
                adv_text = raw_text

            if self.config.attack_gen_adv:
                f_adv.write(f"{raw_text}\t{adv_text}\t{raw_label}\n")
            if self.config.attack_gen_aug:
                f_aug.write(f"{adv_text}\t{raw_label}\n")

        if self.config.attack_gen_adv:
            f_adv.close()
        if self.config.attack_gen_aug:
            f_aug.close()

        print("raw\t", raw_counter.most_common())
        print("adv\t", adv_counter.most_common())
        print("Avg.change#", round(agg.mean("change_num"), 2), "Avg.change%",
              round(100 * agg.mean("change_ratio"), 2))
        print(attack_metric)
