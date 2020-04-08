import csv
from typing import Union, Callable, Dict
import json
from collections import defaultdict
from functools import lru_cache
from .searcher import Searcher


class CachedWordSearcher(Searcher):
    def __init__(
        self,
        file_name: str
    ):
        super().__init__()
        if file_name.endswith(".tsv"):
            f = csv.reader(open(file_name), delimiter='\t', quoting=csv.QUOTE_NONE)
            self.nbrs = defaultdict(lambda: [])
            for row in f:
                self.nbrs[row[0]] = row[1:]
        elif file_name.endswith(".json"):
            self.nbrs = json.load(open(file_name))

    def search(self, word):
        words = self.nbrs[word]
        return words


class CachedIndexSearcher(Searcher):
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

    @lru_cache(maxsize=None)
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
