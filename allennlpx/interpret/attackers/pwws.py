# pylint: disable=protected-access
import copy
from collections import defaultdict

import numpy as np
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.data.token_indexers import (ELMoTokenCharactersIndexer, TokenCharactersIndexer)
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN
from allennlp.modules.text_field_embedders.text_field_embedder import \
    TextFieldEmbedder

from allennlpx.interpret.attackers.attacker import (DEFAULT_IGNORE_TOKENS, Attacker)
from allennlpx.interpret.attackers.policies import (CandidatePolicy, EmbeddingPolicy,
                                                    SpecifiedPolicy, SynonymPolicy)
from allennlpx import allenutil

class PWWS(Attacker):
    """
    ACL 2019 - Generating Natural Language Adversarial Examples through
    Probability Weighted Word Saliency
    """
    def __init__(self, predictor, *, policy: CandidatePolicy = None, **kwargs):
        super().__init__(predictor, **kwargs)
        self.policy = policy

    @torch.no_grad()
    def attack_from_json(
            self,
            inputs: JsonDict = None,
    ) -> JsonDict:
        # we reuse volatile_json to avoid deepcopy of the dict each time a new
        # instance is created, which is rather time consuming.
        # !!! MUST BE CAREFUL since volatile_json will change through the code.
        _volatile_json_ = inputs.copy()

        raw_instance = self.predictor.json_to_labeled_instances(inputs)[0]
        raw_tokens = list(map(lambda x: x.text, self.spacy.tokenize(inputs[self.f2c])))

        # Select words that can be changed
        sids_to_change = []
        nbr_dct = defaultdict(lambda: [])
        for i in range(len(raw_tokens)):
            if raw_tokens[i] not in self.ignore_tokens:
                word = raw_tokens[i]
                if isinstance(self.policy, EmbeddingPolicy):
                    nbrs = self.neariest_neighbours(word, self.policy.measure, self.policy.topk,
                                                    self.policy.rho)
                elif isinstance(self.policy, SynonymPolicy):
                    nbrs = self.synom_searcher.search(word)
                elif isinstance(self.policy, SpecifiedPolicy):
                    nbrs = self.policy.words
                nbrs = [nbr for nbr in nbrs if nbr not in self.forbidden_tokens]
                if len(nbrs) > 0:
                    sids_to_change.append(i)
                    nbr_dct[i] = nbrs

        # 1. Replace each word with <UNK> and other candidate words
        # 2. Generate all sentences, then concatenate them into a
        #    list for batch forwarding
        _jsons = []  # concatenate all jsons
        _offsets = {}  # {sid: [start_offset, number_of_jsons]}

        for sid in sids_to_change:
            tmp_jsons = []
            # first element is the raw sentence
            _volatile_json_[self.f2c] = " ".join(raw_tokens)
            tmp_jsons.append(_volatile_json_.copy())
            # second element is the UNK sentence
            tmp_tokens = copy.copy(raw_tokens)
            tmp_tokens[sid] = '[UNK]' if self.use_bert else DEFAULT_OOV_TOKEN
            _volatile_json_[self.f2c] = " ".join(tmp_tokens)
            tmp_jsons.append(_volatile_json_.copy())
            # starting from the third one are modified sentences
            for nbr in nbr_dct[sid]:
                tmp_tokens = copy.copy(raw_tokens)
                tmp_tokens[sid] = nbr
                _volatile_json_[self.f2c] = " ".join(tmp_tokens)
                tmp_jsons.append(_volatile_json_.copy())

            _offsets[sid] = (len(_jsons), len(tmp_jsons))
            _jsons.extend(tmp_jsons)

        # ugly
        if len(_jsons) == 0:
            return sanitize({
                "adv": raw_tokens,
                "raw": raw_tokens,
                "outputs": self.predictor.predict_json(inputs),
                "changed": 0,
                "success": 0
            })
        _results = self.predictor.predict_batch_json(_jsons)

        # Compute the word saliency
        repl_dct = {}  # {idx: "the replaced word"}
        pwws_dct = {}
        for sid in sids_to_change:
            _start, _num = _offsets[sid]
            results = _results[_start:_start + _num]
            probs = np.array([result['probs'] for result in results])
            true_probs = probs[:, np.argmax(probs[0])]
            raw_prob = true_probs[0]
            oov_prob = true_probs[1]
            other_probs = true_probs[2:]
            repl_dct[sid] = nbr_dct[sid][np.argmin(other_probs)]
            pwws_dct[sid] = np.max(raw_prob - other_probs) * np.exp(raw_prob - oov_prob)

        # max number of tokens that can be changed
        max_change_num = min(self.max_change_num(len(raw_tokens)), len(sids_to_change))

        final_tokens = [ele for ele in raw_tokens]
        sorted_pwws = sorted(pwws_dct.items(), key=lambda x: x[1], reverse=True)
        successful = False
        result = None
        for i in range(max_change_num):
            sid = sorted_pwws[i][0]
            final_tokens[sid] = repl_dct[sid]
            _volatile_json_[self.f2c] = " ".join(final_tokens)
            final_instance = self.predictor._json_to_instance(_volatile_json_)
            result = self.predictor.predict_instance(final_instance)
            final_instance = self.predictor.predictions_to_labeled_instances(
                final_instance, result)[0]
            if final_instance[self.f2a].label != raw_instance[self.f2a].label:
                successful = True
                break

        return sanitize({
            "adv": final_tokens,
            "raw": raw_tokens,
            "outputs": result,
            "changed": i + 1,
            "success": 1 if successful else 0
        })
