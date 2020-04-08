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
from allennlpx.interpret.attackers.policies import (CandidatePolicy,
                                                    EmbeddingPolicy,
                                                    SpecifiedPolicy,
                                                    SynonymPolicy)
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
        policy: CandidatePolicy = None,
        lm_topk=4,  # if lm_topk=-1, the lm re-ranking will be passed
        **kwargs):
        super().__init__(predictor, **kwargs)
        self.num_generation = num_generation
        self.num_population = num_population
        self.policy = policy
        self.lm_topk = lm_topk

        # during an attack, there may be many temp variables
        self.ram_pool = {}

    def perturb(self, raw_tokens, cur_tokens, target_idx=-1):
        _volatile_json_ = self.ram_pool['volatile_json']

        # randomly select a word
        legal_sids = self.ram_pool['legal_sids']
        sid = random.choice(legal_sids)
        cands = self.ram_pool['nbr_dct'][sid]

        # re-ranking words with language model
        if self.lm_topk > 0:
            cand_sents = []
            for cand in cands:
                tmp_tokens = copy.copy(cur_tokens)
                tmp_tokens[sid] = cand
                cand_sents.append(" ".join(tmp_tokens))
            scores = self.lang_model.score(cand_sents)
            ppls = np.array([
                ele['positional_scores'].mean().neg().exp().item()
                for ele in scores
            ])
            cand_idxs = ppls.argsort()[:self.lm_topk]
            cands = [cands[i] for i in cand_idxs]

        # select the one that maximize the drop
        _volatile_json_[self.f2c] = " ".join(raw_tokens)
        tmp_jsons = [_volatile_json_.copy()]
        for cand in cands:
            tmp_tokens = copy.copy(cur_tokens)
            tmp_tokens[sid] = cand
            _volatile_json_[self.f2c] = " ".join(tmp_tokens)
            tmp_jsons.append(_volatile_json_.copy())
        results = self.predictor.predict_batch_json(tmp_jsons)
#         print(results)

        other_results = results[1:]
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
#             success = 1 if other_probs[cand_idx][true_idx] < 0.3 else 0
        else:
            success = 1 if np.argmax(
                other_probs[cand_idx]) == target_idx else 0

        final_tokens = copy.copy(cur_tokens)
        final_tokens[sid] = cands[cand_idx]

        #         print(final_tokens)
        return {
            "result": other_results[cand_idx],
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
                         inputs: JsonDict = None) -> JsonDict:
        _volatile_json_ = inputs.copy()
        raw_tokens = list(map(lambda x: x.text, self.spacy.tokenize(inputs[self.f2c])))

        # pre-compute some variables for later operations
        self.ram_pool.clear()
        legal_sids = []
        nbr_dct = {}
        for i in range(len(raw_tokens)):
            if raw_tokens[i] not in self.ignore_tokens and self.embed_searcher.is_pretrained(
                        raw_tokens[i]):
                lucky_dog = raw_tokens[i]  # use the original word
                if isinstance(self.policy, EmbeddingPolicy):
                    cands = self.neariest_neighbours(lucky_dog,
                                                     self.policy.measure,
                                                     self.policy.topk,
                                                     self.policy.rho)
                elif isinstance(self.policy, SynonymPolicy):
                    cands = self.synom_searcher.search(lucky_dog)
                elif isinstance(self.policy, SpecifiedPolicy):
                    cands = self.policy.nbrs[lucky_dog]
                cands = [
                    ele for ele in cands if ele not in self.forbidden_tokens
                ]
                if len(cands) > 0:
                    legal_sids.append(i)
                    nbr_dct[i] = cands
        self.ram_pool['legal_sids'] = legal_sids
        self.ram_pool['nbr_dct'] = nbr_dct
        self.ram_pool['volatile_json'] = _volatile_json_

        adv_tokens = raw_tokens.copy()
        gid = -1
        success = False
        if len(legal_sids) != 0:
            # initialzie the population
            P = [
                self.perturb(raw_tokens, raw_tokens)
                for _ in range(self.num_population)
            ]

            for gid in range(self.num_generation):
            # after G generation, the maximum change maybe 2^(G-1)
#             max_change_num = self.max_change_num(len(raw_tokens))
#             for gid in range(min(self.num_generation, max_change_num)):
                fitnesses = np.array([ele['fitness'] for ele in P])
                # print(fitnesses)
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
                    new_P = [best]
                    for _ in range(self.num_population - 1):
                        select_prob = fitnesses / fitnesses.sum()
                        p1, p2 = np.random.choice(P, 2, False, select_prob)
                        child_tokens = self.crossover(p1['individual'],
                                                      p2['individual'])
                        next_gen = self.perturb(raw_tokens, child_tokens)
                        new_P.append(next_gen)
                    P = new_P
        
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

    @lazy_property
    def lang_model(self):
        en_lm = torch.hub.load('pytorch/fairseq',
                               'transformer_lm.wmt19.en',
                               tokenizer='moses',
                               bpe='fastbpe')
        en_lm.eval()
        en_lm.cuda()
        return en_lm
