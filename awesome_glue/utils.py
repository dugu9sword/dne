import pathlib
from collections import defaultdict

from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.embedding import \
    _read_pretrained_embeddings_file

import numpy as np
from tabulate import tabulate
from collections import Counter
from luna import auto_create


class FreqUtil:
    @staticmethod
    def print_statistics(vocab: Vocabulary):
        tokens_with_counts = list(vocab._retained_counter['tokens'].items())
        tokens_with_counts.sort(key=lambda x: x[1], reverse=True)
        num = len(tokens_with_counts)
        
        tables = []
        for i in range(1, 11):
            before = min(num // 10 * i, num - 1)
            tables.append([before, tokens_with_counts[before][1]])
        print(tabulate(tables, headers=['top-k', 'min frequency']))
        
    @staticmethod
    def get_group_by_frequency(vocab: Vocabulary, group_id, num_groups):
        tokens_with_counts = list(vocab._retained_counter['tokens'].items())
        tokens_with_counts.sort(key=lambda x: x[1], reverse=True)
        num = len(tokens_with_counts)
        
        start_idx = int(num / num_groups * group_id)
        end_idx = min(int(num / num_groups * (group_id + 1)), num)
        selected = tokens_with_counts[start_idx: end_idx]
        return list(map(lambda x: x[0], selected))
    
    @staticmethod
    def topk_frequency(vocab: Vocabulary, number, order='most', exclude_words=[]):
        assert order in ['most', 'least']
        tokens_with_counts = list(vocab._retained_counter['tokens'].items())
        tokens_with_counts.sort(key=lambda x: x[1], reverse=order =='most')
        ret = []
        for tok, cou in tokens_with_counts:
            if tok not in exclude_words:
                ret.append(tok)
            if len(ret) == number:
                break
        return ret
        

class AttackMetric:
    def __init__(self):
        super().__init__()
        self._succ_num = 0
        self._fail_num = 0
        self._escape_num = 0

    def succeed(self):
        self._succ_num += 1

    def fail(self):
        self._fail_num += 1

    def escape(self):
        self._escape_num += 1

    def count_label(self, gold_label, raw_label, att_label):
        if raw_label == gold_label:
            if att_label == raw_label:
                self.fail()
            else:
                self.succeed()
        else:
            self.escape()

    @property
    def accuracy_before_attack(self):
        return (self._succ_num + self._fail_num) / (self._succ_num + self._fail_num +
                                                    self._escape_num) * 100

    @property
    def accuracy_after_attack(self):
        return self._fail_num / (self._succ_num + self._fail_num + self._escape_num) * 100

    @property
    def flip_ratio(self):
        return self._succ_num / (self._succ_num + self._fail_num + 1e-40) * 100

    def __repr__(self):
        return "Accu before: {:.2f}%, after: {:.2f}%, Flip ratio {:.2f}%".format(
            self.accuracy_before_attack, self.accuracy_after_attack, self.flip_ratio)


def maybe_path(*args):
    for arg in args:
        if pathlib.Path(arg).exists():
            break
    return arg


def text_diff(a_text, b_text):
    if isinstance(a_text, list):
        a_lst, b_lst = a_text, b_text
    else:
        a_lst = a_text.split(" ")
        b_lst = b_text.split(" ")
    assert len(a_lst) == len(b_lst), (a_lst, b_lst)
    a_changes = []
    b_changes = []
    for a_word, b_word in zip(a_lst, b_lst):
        if a_word != b_word:
            a_changes.append(a_word)
            b_changes.append(b_word)
    return {
        "a_changes": a_changes,
        "b_changes": b_changes,
        "change_num": len(a_changes),
        "change_ratio": len(a_changes)/len(a_lst)
    }


WORD2VECS = {
    "fasttext":
    maybe_path("/disks/sdb/zjiehang/embeddings/fasttext/crawl-300d-2M.vec",
               "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"),
    "glove":
    maybe_path("/disks/sdb/zjiehang/embeddings/glove/glove.42B.300d.txt",
               "/root/glove/glove.42B.300d.txt", "http://nlp.stanford.edu/data/glove.42B.300d.zip"),
    "counter":
    maybe_path("/disks/sdb/zjiehang/embeddings/counter/counter.txt", "https://raw.githubusercontent.com/nmrksic/counter-fitting/master/word_vectors/counter-fitted-vectors.txt.zip")
}

EMBED_DIM = defaultdict(lambda: 300, {"elmo": 256})
