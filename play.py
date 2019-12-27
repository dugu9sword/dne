# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'

import torch

from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import StanfordSentimentTreeBankDatasetReader
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_characters_indexer import TokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder, TextFieldEmbedder
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.training.metrics import CategoricalAccuracy
# from allennlp.training.trainer import Trainer
from allennlpx.training.callback_trainer import CallbackTrainer
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, Seq2VecEncoder
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.nn.util import get_text_field_mask
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import pathlib
import hashlib
from allennlp.training.optimizers import DenseSparseAdam
from allennlp.data.iterators.multiprocess_iterator import MultiprocessIterator
from allennlpx.interpret.attackers.attacker import DEFAULT_IGNORE_TOKENS
from allennlpx.interpret.attackers.pgd import PGD
from allennlpx.interpret.attackers.hotflip import HotFlip
from allennlpx.interpret.attackers.bruteforce import BruteForce
from allennlpx.predictors.text_classifier import TextClassifierPredictor
from allennlpx.training.callbacks.evaluate_callback import EvaluateCallback
from allennlpx import allenutil
from luna import auto_create, flt2str
from freq_util import frequency_analysis, analyze_frequency
from collections import Counter
import logging
from luna import ram_write, ram_read, ram_reset
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from luna import log_config, log

log_config("log", "cf")
FORMAT = '%(asctime)-15s %(message)s'
# logging.basicConfig(format=FORMAT, level=logging.INFO)

train_reader = StanfordSentimentTreeBankDatasetReader(
    token_indexers={"tokens": SingleIdTokenIndexer(lowercase_tokens=True)},
    granularity='2-class',
    use_subtrees=True)
reader = StanfordSentimentTreeBankDatasetReader(
    token_indexers={"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}, granularity='2-class')


def load_data():
    train_data = train_reader.read(
        'https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')
    test_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/test.txt')
    return train_data, dev_data, test_data


train_data, dev_data, test_data = auto_create("sst", load_data, True)

vocab = Vocabulary.from_instances(train_data + dev_data + test_data)

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

# embedding_path = None
if pathlib.Path("/disks/sdb/zjiehang").exists():
    print("Code running in china.")
    # embedding_path = "/disks/sdb/zjiehang/frequency/pretrained_embedding/word2vec/GoogleNews-vectors-negative300.txt"
    embedding_path = "/disks/sdb/zjiehang/embeddings/fasttext/crawl-300d-2M.vec"
    # embedding_path = "/disks/sdb/zjiehang/embeddings/gensim_sgns_gnews/model.txt"
    # embedding_path = "/disks/sdb/zjiehang/embeddings/glove/glove.42B.300d.txt"
else:
    embedding_path = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"

if embedding_path:
    cache_embed_path = hashlib.md5(embedding_path.encode()).hexdigest()
    weight = auto_create(
        cache_embed_path, lambda: _read_pretrained_embeddings_file(
            embedding_path, embedding_dim=300, vocab=vocab, namespace="tokens"), True)
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=300,
                                weight=weight,
                                sparse=True,
                                trainable=False)

else:
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=300)

# from allennlpx.interpret.attackers.embedding_searcher import EmbeddingSearcher
# from luna import load_word2vec
# emb_searcher = EmbeddingSearcher(token_embedding.weight,
#                                  word2idx=vocab.get_token_index,
#                                  idx2word=vocab.get_token_from_index)

# cf_embedding = torch.nn.Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
#                                   embedding_dim=300)
# load_word2vec(cf_embedding, vocab._token_to_index["tokens"],
#               "../counter-fitting/results/counter_fitted_vectors.txt")
# cf_searcher = EmbeddingSearcher(cf_embedding.weight,
#                                 word2idx=vocab.get_token_index,
#                                 idx2word=vocab.get_token_from_index)

# emb_searcher.find_neighbours("happy", "euc", topk=20, verbose=True)

pass
# with open('sst_vocab.txt', 'w') as f:
#     for word in vocab._index_to_token['tokens'].values():
#         f.write(word + '\n')
# exit()

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
encoder = PytorchSeq2VecWrapper(
    torch.nn.LSTM(
        300,
        hidden_size=512,
        num_layers=2,
        #                                               bidirectional=True,
        batch_first=True))


class LstmClassifier(Model):
    def __init__(self, word_embeddings, encoder, vocab):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=encoder.get_output_dim(),
                            out_features=vocab.get_vocab_size('label')))
        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, label=None):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)
        #         print(encoder_out.size(), logits.size())
        output = {"logits": logits, "probs": F.softmax(logits, dim=1)}
        #         print(output)
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}


model = LstmClassifier(word_embeddings, encoder, vocab)
model.cuda()

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
                              train_dataset=train_data,
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

total_num = len(test_data) // 4
# total_num = 20
succ_num = 0
src_words = []
tgt_words = []
for i in tqdm(range(total_num)):
    raw_text = allenutil.as_sentence(test_data[i])
    # print(raw_text)
    # print("\t", flt2str(predictor.predict(raw_text)['probs']))

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

    raw_inc_sents = []
    for ti in range(1, len(result['raw'])):
        raw_inc_sents.append({"sentence": allenutil.as_sentence(result['raw'][:ti])})
    raw_inc_results = predictor.predict_batch_json(raw_inc_sents)
    raw_inc_probs = flt2str([x['probs'][0] for x in raw_inc_results], fmt=":.2f")
    att_inc_sents = []
    for ti in range(1, len(result['att'])):
        att_inc_sents.append({"sentence": allenutil.as_sentence(result['att'][:ti])})
    att_inc_results = predictor.predict_batch_json(att_inc_sents)
    att_inc_probs = flt2str([x['probs'][0] for x in att_inc_results], fmt=":.2f")

    log(i)
    table = []
    table.append(result['raw'])
    table.append(raw_inc_probs)
    table.append(result['att'])
    table.append(att_inc_probs)
    table = list(zip(*table))
    log(tabulate(table, floatfmt=".2f"))
    log()

    if result["success"] == 1:
        succ_num += 1

    att_text = allenutil.as_sentence(result['att'])
    # print(att_text)
    # print('\t', flt2str(predictor.predict(att_text)['probs']))
    # print()

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
