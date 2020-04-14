import csv
from typing import Union, Callable, Dict
import json
from collections import defaultdict, Counter
from functools import lru_cache
from .searcher import Searcher
import numpy as np


class CachedWordSearcher(Searcher):
    """
        Load words from a json file
    """
    def __init__(
        self,
        file_name: str,
        vocab_list,
        verbose: bool = False,
    ):
        super().__init__()
        loaded = json.load(open(file_name))
        if verbose:
            print('Before: ')
            nbr_num = list(map(len, list(loaded.values())))
            print(f"total word: {len(loaded)}, ",
                  f"mean: {round(np.mean(nbr_num), 2)}, ",
                  f"median: {round(np.median(nbr_num), 2)}, "
                  f"max: {np.max(nbr_num)}, ")
            print(Counter(nbr_num))
        if vocab_list:
            self.nbrs = defaultdict(lambda: [], {})
            for k in loaded:
                if k in vocab_list:
                    for v in loaded[k]:
                        if v in vocab_list:
                            self.nbrs[k].append(v)
        else:
            self.nbrs = loaded

        if verbose:
            nbrs = dict(self.nbrs)
            print('After: ')
            nbr_num = list(map(len, list(nbrs.values())))
            print(f"total word: {len(nbrs)}, ",
                  f"mean: {round(np.mean(nbr_num), 2)}, ",
                  f"median: {round(np.median(nbr_num), 2)}, "
                  f"max: {np.max(nbr_num)}, ")
            print(Counter(nbr_num))

    def search(self, word):
        if word in self.nbrs:
            return self.nbrs[word]
        else:
            return []


