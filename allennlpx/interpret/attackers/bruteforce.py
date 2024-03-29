# pylint: disable=protected-access
import random
from collections import defaultdict

import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.data.token_indexers import (ELMoTokenCharactersIndexer, TokenCharactersIndexer)
from allennlp.modules.text_field_embedders.text_field_embedder import \
    TextFieldEmbedder

from allennlpx.interpret.attackers.attacker import (DEFAULT_IGNORE_TOKENS, Attacker)


class BruteForce(Attacker):
    """
    Replace k words randomly (each time), try n times.
    """
    def __init__(self,
                 predictor,
                 *,
                 searcher,
                 search_num: int = 256,
                 **kwargs):
        super().__init__(predictor, **kwargs)
        self.searcher = searcher
        self.search_num = search_num

    @torch.no_grad()
    def attack_from_json(self,
                         inputs: JsonDict = None) -> JsonDict:
        _volatile_json_ = inputs.copy()
        raw_instance = self.predictor.json_to_labeled_instances(inputs)[0]
        raw_tokens = list(map(lambda x: x.text, self.spacy.tokenize(inputs[self.f2c])))

        # Select words that can be changed
        sids_to_change = []
        nbr_dct = defaultdict(lambda: [])
        for i in range(len(raw_tokens)):
            if raw_tokens[i] not in self.ignore_tokens:
                word = raw_tokens[i]
                nbrs = self.searcher.search(word)
                nbrs = [nbr for nbr in nbrs if nbr not in self.forbidden_tokens]
                if len(nbrs) > 0:
                    sids_to_change.append(i)
                    nbr_dct[i] = nbrs

        # max number of tokens that can be changed
        max_change_num = min(self.max_change_num(len(raw_tokens)), len(sids_to_change))

        # Construct adversarial instances
        adv_jsons = []
        for i in range(self.search_num):
            adv_tokens = [ele for ele in raw_tokens]
            word_sids = random.choices(sids_to_change, k=max_change_num)
            for word_sid in word_sids:
                adv_tokens[word_sid] = random.choice(nbr_dct[word_sid])
            _volatile_json_[self.f2c] = " ".join(adv_tokens)
            adv_jsons.append(_volatile_json_.copy())

        # Checking attacking status, early stop
        successful = False
        results = self.predictor.predict_batch_json(adv_jsons)

        for i, result in enumerate(results):
            adv_instance = self.predictor._json_to_instance(adv_jsons[i])
            adv_instance = self.predictor.predictions_to_labeled_instances(
                adv_instance, result)[0]
            if adv_instance[self.f2a].label != raw_instance[self.f2a].label:
                successful = True
                break
        adv_tokens = adv_jsons[i][self.f2c].split(" ")
        outputs = result

        return sanitize({
            "adv": adv_tokens,
            "raw": raw_tokens,
            "outputs": outputs,
            "success": 1 if successful else 0
        })
