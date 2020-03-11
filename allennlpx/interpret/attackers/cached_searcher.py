import csv
from typing import Union, Callable, Dict
from collections import defaultdict


class CachedWordSearcher:
    def __init__(
        self,
        file_name: str
    ):
        super().__init__()
        f = csv.reader(open(file_name), delimiter='\t', quoting=csv.QUOTE_NONE)
        self.nbrs = defaultdict(lambda: [])
        for row in f:
            self.nbrs[row[0]] = row[1:]

    def search(self, word):
        words = self.nbrs[word]
        return words


class CachedIndexSearcher:
    def __init__(
        self,
        file_name: str,
        word2idx: Union[Callable, Dict],
        idx2word: Union[Callable, Dict],
    ):
        super().__init__()
        f = csv.reader(open(file_name), delimiter='\t', quoting=csv.QUOTE_NONE)
        self.nbrs = defaultdict(lambda: [])
        for row in f:
            self.nbrs[row[0]] = row[1:]
        if isinstance(word2idx, dict):
            self.word2idx = word2idx.__getitem__
        else:
            self.word2idx = word2idx
        if isinstance(idx2word, dict):
            self.idx2word = idx2word.__getitem__
        else:
            self.idx2word = idx2word
        self.unk_idx = self.word2idx("President Jiang is excited!")

    def search(self, element):
        if isinstance(element, int):
            word = self.idx2word(element)
        elif isinstance(element, str):
            word = element
        else:
            raise Exception
        words = self.nbrs[word]
        words = list(filter(lambda x: self.word2idx(x) != self.unk_idx, words))
        idxes = list(map(lambda x: self.word2idx(x), words))
        return words, idxes
