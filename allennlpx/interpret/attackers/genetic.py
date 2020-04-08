# pylint: disable=protected-access
import random
import copy

import numpy as np
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.data.token_indexers import (ELMoTokenCharactersIndexer,
                                          TokenCharactersIndexer)
from allennlp.modules.text_field_embedders.text_field_embedder import \
    TextFieldEmbedder

from allennlpx.interpret.attackers.attacker import (DEFAULT_IGNORE_TOKENS,
                                                    Attacker)
from luna import lazy_property

class Genetic(Attacker):
    """
    EMNLP 2018 - Generating Natural Language Adversarial Examples
    """
    def __init__(
        self,
        predictor,
        *,
        num_generation=5,
        num_population=20,
        searcher = None,
        lm_topk=4,  # if lm_topk=-1, the lm re-ranking will be passed
        **kwargs):
        super().__init__(predictor, **kwargs)
        self.num_generation = num_generation
        self.num_population = num_population
        self.searcher = searcher
        self.lm_topk = lm_topk

        # during an attack, there may be many temp variables
        self.ram_pool = {}
        
    def evolve(self, P, target_idx = -1):
        _volatile_json_ = self.ram_pool['volatile_json']
        legal_sids = self.ram_pool['legal_sids']
        nbr_dct = self.ram_pool['nbr_dct']
        raw_tokens = self.ram_pool['raw_tokens']
        
        gen_fitnesses = np.array([ele['fitness'] for ele in P])
        best = P[np.argmax(gen_fitnesses)]
        
        new_P = [best]

        # Crossover
        children = []
        select_prob = gen_fitnesses / gen_fitnesses.sum()
        for cid in range(self.num_population - 1):
            p1, p2 = np.random.choice(P, 2, False, select_prob)
            c = [random.sample([w1, w2], 1)[0] \
                            for (w1, w2) in zip(p1['individual'], p2['individual'])]
            children.append(c)

        # Batch perturbation
        _perturbed_sids = {} # {cid: sid}
        _jsons = []  # concatenate all jsons
        _offsets = {}  # {cid: [start_offset, number_of_jsons]}
        for cid in range(self.num_population - 1):
            child_tokens = children[cid]
            # randomly select a word
            sid = random.choice(legal_sids)
            cands = nbr_dct[sid]
            # replace with candidate word
            tmp_jsons = []
            for cand in cands:
                tmp_tokens = copy.copy(child_tokens)
                tmp_tokens[sid] = cand
                _volatile_json_[self.f2c] = " ".join(tmp_tokens)
                tmp_jsons.append(_volatile_json_.copy())
            _perturbed_sids[cid] = sid
            _offsets[cid] = (len(_jsons), len(tmp_jsons))
            _jsons.extend(tmp_jsons)

        # Batch forward 
        _volatile_json_[self.f2c] = " ".join(raw_tokens)
        raw_result = self.predictor.predict_json(_volatile_json_)
        _results= self.predictor.predict_batch_json(_jsons, fast=True)

        # Stronger one will survive
        true_idx = np.argmax(raw_result['probs'])
        for cid in range(self.num_population - 1):
            child_tokens = children[cid]
            _start, _num = _offsets[cid]
            results = _results[_start:_start + _num]
            probs = np.array([result['probs'] for result in results])

            if target_idx == -1:
                true_probs = probs[:, true_idx]
                perturb_fitnesses = 1 - true_probs
            else:
                perturb_fitnesses = probs[:, target_idx]
            
            perturb_fitnesses += 1e-6
            cand_idx = np.argmax(perturb_fitnesses)
            cand_fitness = np.max(perturb_fitnesses)

            if target_idx == -1:
                success = 1 if np.argmax(probs[cand_idx]) != true_idx else 0
            else:
                success = 1 if np.argmax(probs[cand_idx]) == target_idx else 0

            final_tokens = copy.copy(child_tokens)
            sid = _perturbed_sids[cid]
            final_tokens[sid] = nbr_dct[sid][cand_idx]

            next_gen =  {
                "result": results[cand_idx],
                "individual": final_tokens,
                "fitness": cand_fitness,
                "success": success
            }
            new_P.append(next_gen)
        return new_P

    @torch.no_grad()
    def attack_from_json(self,
                         inputs: JsonDict = None) -> JsonDict:
        _volatile_json_ = inputs.copy()
        raw_tokens = list(map(lambda x: x.text, self.spacy.tokenize(inputs[self.f2c])))

        # pre-compute some variables for later operations
        self.ram_pool.clear()
        legal_sids = []
        nbr_dct = {}
        for i in range(len(raw_tokens)):
            if raw_tokens[i] not in self.ignore_tokens:
                lucky_dog = raw_tokens[i]  # use the original word
                cands = self.searcher.search(lucky_dog)
                cands = [
                    ele for ele in cands if ele not in self.forbidden_tokens
                ]
                if len(cands) > 0:
                    legal_sids.append(i)
                    nbr_dct[i] = cands
        self.ram_pool['legal_sids'] = legal_sids
        self.ram_pool['nbr_dct'] = nbr_dct
        self.ram_pool['volatile_json'] = _volatile_json_
        self.ram_pool['raw_tokens'] = raw_tokens

        adv_tokens = raw_tokens.copy()
        gid = -1
        success = False
        if len(legal_sids) != 0:
            # initialzie the population
            P = [
                {"individual": raw_tokens, "fitness": 1e-6, "success": 0}
                for _ in range(self.num_population)
            ]

            for gid in range(self.num_generation):
                fitnesses = np.array([ele['fitness'] for ele in P])
                best = P[np.argmax(fitnesses)]
                print("generation ", gid, ":", best['fitness'])
                if best['success']:
                    return sanitize({
                        "adv": best['individual'],
                        "raw": raw_tokens,
                        "outputs": best['result'],
                        "success": 1,
                        "generation": gid + 1
                    })
                else:
                    P = self.evolve(P)
        
        adv_tokens = best['individual']
        _volatile_json_[self.f2c] = " ".join(adv_tokens)
        result = self.predictor.predict_json(_volatile_json_)
        raw_instance = self.predictor._json_to_labeled_instance(inputs)
        final_instance = self.predictor._json_to_instance(_volatile_json_)
        final_instance = self.predictor.predictions_to_labeled_instances(final_instance, result)[0]
        success = raw_instance[self.f2a].label != final_instance[self.f2a].label

        return sanitize({
            "adv": adv_tokens,
            "raw": raw_tokens,
            "outputs": result,
            "success": success,
            "generation": gid + 1
        })
        