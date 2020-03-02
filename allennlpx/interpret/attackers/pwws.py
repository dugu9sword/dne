# pylint: disable=protected-access
import copy
from collections import defaultdict
from functools import lru_cache
from itertools import product
from typing import List

import numpy as np
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import (ELMoTokenCharactersIndexer,
                                          TokenCharactersIndexer)
from allennlp.data.tokenizers import SpacyTokenizer, Token
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, Vocabulary
from allennlp.modules.text_field_embedders.text_field_embedder import \
    TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding

from allennlpx import allenutil
from allennlpx.interpret.attackers.attacker import (DEFAULT_IGNORE_TOKENS,
                                                    Attacker)
from allennlpx.interpret.attackers.embedding_searcher import EmbeddingSearcher
from allennlpx.interpret.attackers.policies import (CandidatePolicy,
                                                    EmbeddingPolicy,
                                                    SynonymPolicy)
from allennlpx.interpret.attackers.synonym_searcher import SynonymSearcher
from luna import cast_list, lazy_property, time_record


class PWWS(Attacker):
    def __init__(self, 
                 predictor, 
                 *, 
                 policy: CandidatePolicy = None,
                 **kwargs):
        super().__init__(predictor, **kwargs)
        self.policy = policy

    @torch.no_grad()
    def attack_from_json(self,
                         inputs: JsonDict = None,
                         field_to_change: str = 'tokens',
                         field_to_attack: str = 'label',
                         grad_input_field: str = 'grad_input_1',
                         ) -> JsonDict:
        raw_instance = self.predictor.json_to_labeled_instances(inputs)[0]
        raw_tokens = list(map(lambda x: x.text, self.spacy.tokenize(inputs[field_to_change])))

        # Select words that can be changed
        sids_to_change = []
        nbr_dct = defaultdict(lambda: [])
        for i in range(len(raw_tokens)):
            if raw_tokens[i] not in self.ignore_tokens:
                word = raw_tokens[i]
                if isinstance(self.policy, EmbeddingPolicy):
                    nbrs = self.neariest_neighbours(word, 
                                                    self.policy.measure, 
                                                    self.policy.topk, 
                                                    self.policy.rho)
                elif isinstance(self.policy, SynonymPolicy):
                    nbrs = self.synom_searcher.search(word)
                nbrs = [nbr for nbr in nbrs if nbr not in self.forbidden_tokens]
                if len(nbrs) > 0:
                    sids_to_change.append(i)
                    nbr_dct[i] = nbrs
        
        # Compute the word saliency
        repl_dct = {}  # {idx: "the replaced word"}
        drop_dct = {}  # {idx: prob_current - prob_replaced}
        sali_dct = {}  # {idx: prob_current - prob_unk}
        pwws_dct = {}
        
        for sid in sids_to_change:
            tmp_instances = []
            # first element is the raw sentence
            tmp_instances.append(self._tokens_to_instance(raw_tokens))
            # second element is the UNK sentence
            tmp_tokens = copy.copy(raw_tokens)
            tmp_tokens[sid] = DEFAULT_OOV_TOKEN
            tmp_instances.append(self._tokens_to_instance(tmp_tokens))
            # starting from the third one are modified sentences
            for nbr in nbr_dct[sid]:
                tmp_tokens = copy.copy(raw_tokens)
                tmp_tokens[sid] = nbr
                tmp_instances.append(self._tokens_to_instance(tmp_tokens))
            results = self.predictor._model.forward_on_instances(tmp_instances)
            
            probs = np.array([result['probs'] for result in results])
            true_probs = probs[:, np.argmax(probs[0])]
            raw_prob = true_probs[0]
            oov_prob = true_probs[1]
            other_probs = true_probs[2:]
            repl_dct[sid] = nbr_dct[sid][np.argmin(other_probs)]
            drop_dct[sid] = np.max(raw_prob - other_probs)
            sali_dct[sid] = raw_prob - oov_prob
            
            pwws_dct[sid] = drop_dct[sid] * np.exp(sali_dct[sid])
            
#             print(sid, raw_tokens[sid], repl_dct[sid], drop_dct[sid], sali_dct[sid])
#             print(true_probs)

        # Seems that PWWS is ... not that efficient?
#         total_exp = 0
#         for sid in sids_to_change:
#             sali_dct[sid] = np.exp(sali_dct[sid])
#             total_exp += sali_dct[sid]
#         for sid in sids_to_change:
#             pwws_dct[sid] = drop_dct[sid] * sali_dct[sid] / total_exp
        
        
        # max number of tokens that can be changed
        max_change_num = min(self.max_change_num(len(raw_tokens)), len(sids_to_change))
        
        final_tokens = [ele for ele in raw_tokens]
        sorted_pwws = sorted(pwws_dct.items(), key=lambda x:x[1], reverse=True)
        successful = False
        result = None
        for i in range(max_change_num):
            sid = sorted_pwws[i][0]
            final_tokens[sid] = repl_dct[sid]
            final_instance = self.predictor._dataset_reader.text_to_instance(" ".join(final_tokens))
            result = self.predictor._model.forward_on_instance(final_instance)
            final_instance = self.predictor.predictions_to_labeled_instances(final_instance, result)[0]
            if final_instance[field_to_attack].label != raw_instance[field_to_attack].label:
                successful = True
                break
        
        return sanitize({
            "adv": final_tokens,
            "raw": raw_tokens,
            "outputs": result,
            "changed": i + 1,
            "success": 1 if successful else 0
        })