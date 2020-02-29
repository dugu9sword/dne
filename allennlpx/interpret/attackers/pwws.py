# pylint: disable=protected-access
from copy import deepcopy
from typing import List

import numpy
import torch
import numpy as np
from allennlp.common.util import JsonDict, sanitize
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from luna import cast_list, lazy_property
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN

from allennlpx.interpret.attackers.attacker import Attacker, DEFAULT_IGNORE_TOKENS
from allennlpx.interpret.attackers.embedding_searcher import EmbeddingSearcher
from allennlpx.interpret.attackers.synonym_searcher import SynonymSearcher
from allennlpx import allenutil
from itertools import product
from collections import defaultdict
import random

from functools import lru_cache
from allennlp.data.tokenizers import SpacyTokenizer
from luna import time_record


class PWWS(Attacker):
    def __init__(self, predictor):
        super().__init__(predictor)
        self.spacy = SpacyTokenizer()

    @torch.no_grad()
    def attack_from_json(self,
                         inputs: JsonDict = None,
                         field_to_change: str = 'tokens',
                         field_to_attack: str = 'label',
                         grad_input_field: str = 'grad_input_1',
                         ignore_tokens: List[str] = DEFAULT_IGNORE_TOKENS,
                         forbidden_tokens: List[str] = DEFAULT_IGNORE_TOKENS,
                         max_change_num_or_ratio: int = 5,
                         measure='euc',
                         topk=20,
                         rho=None) -> JsonDict:
        if self.vocab is None:
            raise Exception('initialize it first~')
        
        raw_instance = self.predictor.json_to_labeled_instances(inputs)[0]
        raw_tokens = list(map(lambda x: x.text, self.spacy.tokenize(inputs[field_to_change])))

        # Select words that can be changed
        sids_to_change = []
        nbr_dct = defaultdict(lambda: [])
        for i in range(len(raw_tokens)):
            if raw_tokens[i] not in ignore_tokens:
                word = raw_tokens[i]
#                 nbrs = self.neariest_neighbours(word, measure, topk, rho)
                nbrs = self.synom_searcher.search(word)
                nbrs = [nbr for nbr in nbrs if nbr not in forbidden_tokens]
                if len(nbrs) > 0:
                    sids_to_change.append(i)
                    nbr_dct[i] = nbrs
        
        # Compute the word saliency
        repl_dct = {}  # {idx: "the replaced word"}
        drop_dct = {}  # {idx: prob_current - prob_replaced}
        sali_dct = {}  # {idx: prob_current - prob_unk}

        for sid in sids_to_change:
            tmp_instances = []
            tmp_instances.append(self.predictor._dataset_reader.text_to_instance(" ".join(raw_tokens)))
            tmp_tokens = [ele for ele in raw_tokens]
            tmp_tokens[sid] = DEFAULT_OOV_TOKEN
            tmp_instances.append(self.predictor._dataset_reader.text_to_instance(" ".join(tmp_tokens)))
            for nbr in nbr_dct[sid]:
                tmp_tokens = [ele for ele in raw_tokens]
                tmp_tokens[sid] = nbr
                tmp_instance = self.predictor._dataset_reader.text_to_instance(" ".join(tmp_tokens))
                tmp_instances.append(tmp_instance)
            results = self.predictor._model.forward_on_instances(tmp_instances)
            probs = np.array([result['probs'] for result in results])
            true_idx = np.argmax(probs[0])
            true_probs = probs[:, true_idx]
            raw_prob = true_probs[0]
            oov_prob = true_probs[1]
            other_probs = true_probs[2:]
            repl_dct[sid] = nbr_dct[sid][np.argmin(other_probs)]
            drop_dct[sid] = np.max(raw_prob - other_probs)
            sali_dct[sid] = raw_prob - oov_prob
            
            print(sid, raw_tokens[sid], repl_dct[sid], drop_dct[sid], sali_dct[sid])
#             print(true_probs)

        pwws_dct = {}
        total_exp = 0
        for sid in sids:
            sali_dct[sid] = np.exp(sali_dct[sid])
            total_exp += sali_dct[sid]
        for sid in sids:
            pwws_dct[sid] = drop_dct[sid] * sali_dct[sid] / total_exp
        
        
        # Construct adversarial instances
        att_instances = []
        for i in range(search_num):
            att_tokens = [ele for ele in raw_tokens]
            word_sids = random.choices(sids_to_change, k=max_change_num)
            for word_sid in word_sids:
                att_tokens[word_sid] = random.choice(nbr_dct[word_sid])
            att_instances.append(
                self.predictor._dataset_reader.text_to_instance(" ".join(att_tokens)))

            
        # max number of tokens that can be changed
        if max_change_num_or_ratio < 1:
            max_change_num = int(len(raw_tokens) * max_change_num_or_ratio)
        else:
            max_change_num = max_change_num_or_ratio
        max_change_num = min(max_change_num, len(sids_to_change))
        
        # Checking attacking status, early stop
        successful = False
        results = self.predictor._model.forward_on_instances(att_instances)
        for i, result in enumerate(results):
            att_instance = self.predictor.predictions_to_labeled_instances(att_instances[i], result)[0]
            if att_instance[field_to_attack].label != raw_instance[field_to_attack].label:
                successful = True
                break
        att_tokens = att_instances[i][field_to_change].tokens
        outputs = result

        
        return sanitize({
            "att": att_tokens,
            "raw": raw_tokens,
            "outputs": outputs,
            "success": 1 if successful else 0
        })

    @lazy_property
    def embed_searcher(self) -> EmbeddingSearcher:
        return EmbeddingSearcher(embed=self.token_embedding,
                                 idx2word=lambda x: self.vocab.get_token_from_index(x),
                                 word2idx=lambda x: self.vocab.get_token_index(x))
    
    @lazy_property
    def synom_searcher(self) -> SynonymSearcher:
        return SynonymSearcher(vocab_list=self.vocab.get_index_to_token_vocabulary().values())

    @lru_cache(maxsize=None)
    def neariest_neighbours(self, word, measure, topk, rho):
        # May be accelerated by caching a the distance
        vals, idxs = self.embed_searcher.find_neighbours(word, measure=measure, topk=topk, rho=rho)
        return [self.vocab.get_token_from_index(idx) for idx in cast_list(idxs)]