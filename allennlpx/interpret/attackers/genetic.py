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
from luna import ram_append, ram_read, flt2str
from allennlpx import allenutil

class Genetic(Attacker):
    """
    EMNLP 2018 - Generating Natural Language Adversarial Examples
    """
    def __init__(
        self,
        predictor,
        *,
        lm_constraints,
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
        self.lm_constraints = lm_constraints

        # during an attack, there may be many temp variables
        self.ram_pool = {}
        
    def evolve(self, P, target_idx = -1):
        _volatile_json_ = self.ram_pool['volatile_json']
        legal_sids = self.ram_pool['legal_sids']
        nbr_dct = self.ram_pool['nbr_dct']
        raw_tokens = self.ram_pool['raw_tokens']
        
        gen_fitnesses = np.array([ele['fitness'] for ele in P])
        gen_fitnesses += 1e-6
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
            
            if success:
                print('DETECTED SUCCESS with PROBS', results[cand_idx])
                pass_validation = 0 
                for _ in range(5):
                    validation = self.predictor.predict_json({"sent": " ".join(final_tokens)}, fast=True)
                    print("validation:", validation)
                    if np.argmax(validation['probs']) != true_idx:
                        pass_validation += 1
                if not pass_validation >= 3:
                    print("Validation not pass")
                    success = 0
                
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
        raw_tokens = inputs[self.f2c].split(" ")
        
#         print(list(self.lm_constraints.keys())[1])
#         print(allenutil.as_sentence(raw_tokens))
        lm_filters = self.lm_constraints[allenutil.as_sentence(raw_tokens)]
#         print(lm_filters)
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
                ram_append("before_lm", len(cands))
                cands = [
                    ele for ele in cands if ele in lm_filters[str(i)]
                ]
                ram_append("after_lm", len(cands))
                if len(cands) > 0:
                    legal_sids.append(i)
                    nbr_dct[i] = cands
        print("LM constraints:", round(100 * sum(ram_read("after_lm")) /sum(ram_read("before_lm")), 2))
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
                fitnesses = [ele['fitness'] for ele in P]
                best = P[np.argmax(fitnesses)]
                print("generation ", gid, ":", 
                      "topk: ", flt2str(sorted(fitnesses, reverse=True)[:3], ":4.3f", cat=", "),
                      "mean: ", round(np.mean(fitnesses), 3),
                      "median: ", round(np.median(fitnesses), 3),
                     )
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
        