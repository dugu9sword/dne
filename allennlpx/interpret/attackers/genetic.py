# pylint: disable=protected-access
import random
from collections import defaultdict
from functools import lru_cache
from itertools import product
from typing import List
import copy

import numpy as np
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import (ELMoTokenCharactersIndexer,
                                          TokenCharactersIndexer)
from allennlp.data.tokenizers import SpacyTokenizer, Token
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


class Genetic(Attacker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def perturb(self,
                raw_tokens,
                cur_tokens,
                target_idx = -1
               ):
        # randomly select a word
        legal_sids = []
        for i in range(len(raw_tokens)):
            if raw_tokens[i] not in self.ignore_tokens and self.embed_searcher.is_pretrained(raw_tokens[i]):
                legal_sids.append(i)
        sid = random.choice(legal_sids)
        lucky_dog = raw_tokens[sid]          # use the original word
#         print("Selected ", sid, lucky_dog)
        
        # find top-k neighbours using word vectors
        cands = self.neariest_neighbours(lucky_dog, 'cos', 10, None)
        cands = [ele for ele in cands if ele not in self.forbidden_tokens]
#         print(cands)
        
        # re-ranking words with language model
        cand_sents = []
        for cand in cands:
            tmp_tokens = copy.copy(cur_tokens)
            tmp_tokens[sid] = cand
            cand_sents.append(" ".join(tmp_tokens))
        scores = self.lang_model.score(cand_sents)
        ppls = np.array([ele['positional_scores'].mean().neg().exp().item() for ele in scores])
        cand_idxs = ppls.argsort()[:5]
        cands = [cands[i] for i in cand_idxs]
        
        # select the one that maximize the drop
        tmp_instances = [self._tokens_to_instance(raw_tokens)]
        for cand in cands:
            tmp_tokens = copy.copy(cur_tokens)
            tmp_tokens[sid] = cand
            tmp_instances.append(self._tokens_to_instance(tmp_tokens))
        results = self.predictor._model.forward_on_instances(tmp_instances)
        
        probs = np.array([result['probs'] for result in results])
        other_probs = probs[1:]
        true_idx = np.argmax(probs[0])
        
        if target_idx == -1:
            true_probs = probs[:, true_idx]
            target_probs = 1 - true_probs
        else:
            target_probs = probs[:, target_idx]
            
#         raw_prob = true_probs[0]
        fitnesses = target_probs[1:]
        cand_idx = np.argmax(fitnesses)
        cand_fitness = np.max(fitnesses)
        
        if target_idx == -1:
            success = 1 if np.argmax(other_probs[cand_idx]) != true_idx else 0
        else:
            success = 1 if np.argmax(other_probs[cand_idx]) == target_idx else 0 
            
        final_tokens = copy.copy(cur_tokens)
        final_tokens[sid] = cands[cand_idx]
        
#         print(final_tokens)
        return { 
            "individual": final_tokens,
            "fitness": cand_fitness,
            "success": success 
        }
    
    def crossover(self, a_tokens, b_tokens):
        coins = np.random.randint(0, 2, len(a_tokens))
        ret = a_tokens.copy()
        for i in range(len(ret)):
            if coins[i] == 1:
                ret[i] = b_tokens[i]
        return ret

    @torch.no_grad()
    def attack_from_json(self,
                         inputs: JsonDict = None,
                         field_to_change: str = 'tokens',
                         field_to_attack: str = 'label',
                         grad_input_field: str = 'grad_input_1',
                         num_generation = 5,
                         num_population = 20,
                         policy: CandidatePolicy= None) -> JsonDict:
        if self.token_embedding is None:
            raise Exception('initialize it first~')

#         raw_instance = self.predictor.json_to_labeled_instances(inputs)[0]
        raw_tokens = list(map(lambda x: x.text, self.spacy.tokenize(inputs[field_to_change])))
        
        # initialzie the population
        P = [self.perturb(raw_tokens, raw_tokens) for _ in range(num_population)] 
        
        adv_tokens = None
        for gid in range(num_generation):
            fitnesses = np.array([ele['fitness'] for ele in P])
            best = P[np.argmax(fitnesses)]
            if best['success']:
                adv_tokens = best['individual']
                break
            else:
                new_P = [best]
                for _ in range(num_population - 1):
                    select_prob = fitnesses / fitnesses.sum()
                    p1, p2 = np.random.choice(P, 2, False, select_prob)
                    child_tokens = self.crossover(p1['individual'], p2['individual'])
                    new_tokens = self.perturb(raw_tokens, child_tokens)
                    new_P.append(new_tokens)
                P = new_P
        
        if adv_tokens is not None:
            result = self.predictor._model.forward_on_instance(
                self._tokens_to_instance(adv_tokens)
            )
        else:
            result = None
            
        return sanitize({
            "adv": adv_tokens,
            "raw": raw_tokens,
            "outputs": result,
            "success": 1 if adv_tokens else 0,
            "generation": gid + 1
        })
        
    @lazy_property
    def lang_model(self):
        en_lm = torch.hub.load('pytorch/fairseq', 
                               'transformer_lm.wmt19.en', 
                               tokenizer='moses', 
                               bpe='fastbpe')
        en_lm.eval()
        en_lm.cuda()
        return en_lm

        
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
        if idxs is None:
            return []
        else:
            return [self.vocab.get_token_from_index(idx) for idx in cast_list(idxs)]
