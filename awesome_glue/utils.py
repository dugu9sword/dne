import pathlib
import hashlib
from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from luna import auto_create
from collections import defaultdict


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


WORD2VECS = {
    "fasttext":
    maybe_path("/disks/sdb/zjiehang/embeddings/fasttext/crawl-300d-2M.vec",
               "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"),
    "glove":
    maybe_path("/disks/sdb/zjiehang/embeddings/glove/glove.42B.300d.txt",
               "/root/glove/glove.42B.300d.txt", "http://nlp.stanford.edu/data/glove.42B.300d.zip"),
}

EMBED_DIM = defaultdict(lambda: 300, {"elmo": 256})
