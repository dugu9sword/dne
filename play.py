# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'

import hashlib
import logging
import pathlib
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BasicIterator, BucketIterator
from allennlp.data.iterators.multiprocess_iterator import MultiprocessIterator
from allennlp.data.token_indexers.single_id_token_indexer import \
    SingleIdTokenIndexer
from allennlp.data.token_indexers.token_characters_indexer import \
    TokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import (PytorchSeq2VecWrapper, Seq2VecEncoder)
from allennlp.modules.text_field_embedders import (BasicTextFieldEmbedder, TextFieldEmbedder)
from allennlp.modules.token_embedders import Embedding, TokenEmbedder
from allennlp.modules.token_embedders.embedding import \
    _read_pretrained_embeddings_file
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.optimizers import DenseSparseAdam
from tabulate import tabulate
from torch.nn.utils.rnn import pad_packed_sequence
from tqdm import tqdm

from allennlpx import allenutil
from allennlpx.interpret.attackers.attacker import DEFAULT_IGNORE_TOKENS
from allennlpx.interpret.attackers.bruteforce import BruteForce
from allennlpx.interpret.attackers.hotflip import HotFlip
from allennlpx.interpret.attackers.pgd import PGD
from allennlpx.predictors.text_classifier import TextClassifierPredictor
# from allennlp.training.trainer import Trainer
from allennlpx.training.callback_trainer import CallbackTrainer
from allennlpx.training.callbacks.evaluate_callback import EvaluateCallback
from freq_util import analyze_frequency, frequency_analysis
from luna import (auto_create, flt2str, log, log_config, ram_read, ram_reset, ram_write)
from sst_model import LstmClassifier

log_config("log", "cf")
FORMAT = '%(asctime)-15s %(message)s'
# logging.basicConfig(format=FORMAT, level=logging.INFO)

sub_reader = StanfordSentimentTreeBankDatasetReader(
    token_indexers={"tokens": SingleIdTokenIndexer(lowercase_tokens=True)},
    granularity='2-class',
    use_subtrees=True)
reader = StanfordSentimentTreeBankDatasetReader(
    token_indexers={"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}, granularity='2-class')


def load_data():
    sub_train_data = sub_reader.read(
        'https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
    train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')
    test_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/test.txt')
    return sub_train_data, train_data, dev_data, test_data


sub_train_data, train_data, dev_data, test_data = auto_create("sst", load_data, True)

vocab = auto_create("sst_vocab",
                    lambda: Vocabulary.from_instances(sub_train_data + dev_data + test_data))

counter = Counter(dict(vocab._retained_counter['tokens']))
freq_threshold = 1000
high_freq_words, high_freq_counts = list(zip(*counter.most_common()[:freq_threshold]))
high_freq_words = list(high_freq_words)
low_freq_words, low_freq_counts = list(zip(*counter.most_common()[:freq_threshold - 1:-1]))
low_freq_words = list(low_freq_words)
print("Threshold is set to {}, #high_freq_words={}, #low_freq_words={}".format(
    freq_threshold, sum(high_freq_counts), sum(low_freq_counts)))
ram_write("high_freq_words", high_freq_words)
ram_write("low_freq_words", low_freq_words)
# analyze_frequency(vocab)
# exit()

model = LstmClassifier(vocab).cuda()

model_path = 'sst_model.pt'
if pathlib.Path(model_path).exists():
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
else:
    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
    # iterator = BasicIterator(batch_size=32)
    iterator.index_with(vocab)
    optimizer = DenseSparseAdam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-5)

    trainer = CallbackTrainer(model=model,
                              optimizer=optimizer,
                              iterator=iterator,
                              train_dataset=sub_train_data,
                              validation_dataset=dev_data,
                              num_epochs=8,
                              shuffle=True,
                              patience=None,
                              cuda_device=0,
                              callbacks=[EvaluateCallback(test_data)])
    trainer.train()

    with open(model_path, 'wb') as f:
        torch.save(model.state_dict(), f)

for module in model.modules():
    if isinstance(module, TextFieldEmbedder):
        for embed in module._token_embedders.keys():
            module._token_embedders[embed].weight.requires_grad = True

pos_words = [line.rstrip('\n') for line in open("sentiment-words/positive-words.txt")]
neg_words = [line.rstrip('\n') for line in open("sentiment-words/negative-words.txt")]
not_words = [line.rstrip('\n') for line in open("sentiment-words/negation-words.txt")]
forbidden_words = pos_words + neg_words + not_words + DEFAULT_IGNORE_TOKENS

predictor = TextClassifierPredictor(model.cpu(), reader)
# attacker = HotFlip(predictor)
attacker = BruteForce(predictor)
attacker.initialize()

# total_num = len(test_data) // 4
data_to_attack = train_data
total_num = len(data_to_attack)
# total_num = 20
succ_num = 0
src_words = []
tgt_words = []
for i in tqdm(range(total_num)):
    raw_text = allenutil.as_sentence(data_to_attack[i])

    # result = attacker.attack_from_json({"sentence": raw_text},
    #                                    ignore_tokens=forbidden_words,
    #                                    forbidden_tokens=forbidden_words,
    #                                    step_size=100,
    #                                    max_change_num=3,
    #                                    iter_change_num=2)

    result = attacker.attack_from_json({"sentence": raw_text},
                                       ignore_tokens=forbidden_words,
                                       forbidden_tokens=forbidden_words,
                                       max_change_num=5,
                                       search_num=256)

    # raw_inc_sents = []
    # for ti in range(1, len(result['raw'])):
    #     raw_inc_sents.append({"sentence": allenutil.as_sentence(result['raw'][:ti])})
    # raw_inc_results = predictor.predict_batch_json(raw_inc_sents)
    # raw_inc_probs = flt2str([x['probs'][0] for x in raw_inc_results], fmt=":.2f")
    # att_inc_sents = []
    # for ti in range(1, len(result['att'])):
    #     att_inc_sents.append({"sentence": allenutil.as_sentence(result['att'][:ti])})
    # att_inc_results = predictor.predict_batch_json(att_inc_sents)
    # att_inc_probs = flt2str([x['probs'][0] for x in att_inc_results], fmt=":.2f")

    # log(i)
    # table = []
    # table.append(result['raw'])
    # table.append(raw_inc_probs)
    # table.append(result['att'])
    # table.append(att_inc_probs)
    # table = list(zip(*table))
    # log(tabulate(table, floatfmt=".2f"))
    # log()

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

print(f'Succ rate {succ_num/total_num*100}')

# print("#num of words that can be changed: [mean] {:.2f} [median] {:.2f}".format(
#     np.mean(ram_read("can_change")), np.median(ram_read("can_change"))))
# print("#num of neighbours of each word: [mean] {:.2f} [median] {:.2f}".format(
#     np.mean(ram_read("nbrs")), np.median(ram_read("nbrs"))))
# print("distance of neighbours of each word: [mean] {:.2f} [median] {:.2f}".format(
#     np.mean(ram_read("vals")), np.median(ram_read("vals"))))

# print(src_words)
# print(tgt_words)
# frequency_analysis(Counter(dict(vocab._retained_counter['tokens'])), src_words, tgt_words)
