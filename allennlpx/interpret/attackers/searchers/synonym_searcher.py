from functools import lru_cache

from nltk.corpus import wordnet as wn
from .searcher import Searcher


class SynonymSearcher(Searcher):
    def __init__(self, vocab_list = None):
        self.vocab_list = vocab_list
    
    @staticmethod
    def _to_wordnet_pos(pos):
        if pos is None:
            return None
        pos = pos.lower()
        if pos in ['r', 'n', 'v']:  # ADV, NOUN, VERB
            return pos
        elif pos == 'j':
            return 'as' # ADJ/ADJ_SAT
        else:
            return None
    
    @lru_cache(maxsize=None)
    def search(self, word, penn_pos=None, only_unigram=True):
        pos = SynonymSearcher._to_wordnet_pos(penn_pos)
        synonyms = []
        for syn in wn.synsets(word): 
            if pos is None or syn.pos() in pos:
                for synonym in syn.lemma_names():
                    if "_" in synonym and only_unigram:
                        continue
                    else:
                        synonym = synonym.replace("_", " ")
                    if synonym in synonyms:
                        continue
                    legal = True
                    if self.vocab_list is not None:
                        split = synonym.split(" ")
                        for ele in split:
                            if ele not in self.vocab_list:
                                legal = False
                    if legal:
                        synonyms.append(synonym)
        return synonyms
