# pylint: disable=protected-access
from copy import deepcopy
from typing import List

import numpy
import torch

from allennlp.common.util import JsonDict, sanitize
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from luna import cast_list, lazy_property

from allennlpx.interpret.attackers.attacker import EmbedAttacker, DEFAULT_IGNORE_TOKENS
from allennlpx.interpret.attackers.embedding_searcher import EmbeddingSearcher
from allennlpx import allenutil
from itertools import product
from collections import defaultdict
import random

from functools import lru_cache
from luna import time_record



class BruteForce(EmbedAttacker):
    def attack_from_json(self,
                         inputs: JsonDict = None,
                         field_to_change: str = 'tokens',
                         field_to_attack: str = 'label',
                         grad_input_field: str = 'grad_input_1',
                         ignore_tokens: List[str] = DEFAULT_IGNORE_TOKENS,
                         forbidden_tokens: List[str] = DEFAULT_IGNORE_TOKENS,
                         max_change_num: int = 5,
                         measure = 'euc',
                         topk = 20,
                         rho = None,
                         search_num: int = 512) -> JsonDict:
        if self.token_embedding is None:
            self.initialize()
        raw_instance = self.predictor.json_to_labeled_instances(inputs)[0]
        raw_text_field: TextField = raw_instance[field_to_change]
        raw_tokens = raw_text_field.tokens

        raw_output = self.predictor.predict_instance(raw_instance)

        sids_to_change = []
        nbr_dct = defaultdict(lambda: [])
        for i in range(len(raw_tokens)):
            if raw_tokens[i].text not in ignore_tokens:
                word = raw_tokens[i].text
                nbrs = self.neariest_neighbours(word, measure, topk, rho)
                nbrs = [nbr for nbr in nbrs if nbr not in forbidden_tokens]
                if len(nbrs) > 0:
                    sids_to_change.append(i)
                    nbr_dct[i] = nbrs
        max_change_num = min(max_change_num, len(sids_to_change))

        instances = []
        for i in range(search_num):
            att_instance = deepcopy(raw_instance)
            word_sids = random.choices(sids_to_change, k=max_change_num)
            for word_sid in word_sids:
                att_instance[field_to_change].tokens[word_sid] = Token(
                    random.choice(nbr_dct[word_sid]))
            att_instance.indexed = False
            instances.append(att_instance)

        successful = False
        results = self.predictor._model.forward_on_instances(instances)
        for i, result in enumerate(results):
            # print(allenutil.as_sentence(instances[i]))
            # print(result['probs'][0] - result['probs'][1])
            att_instance = self.predictor.predictions_to_labeled_instances(instances[i], result)[0]
            if att_instance.fields[field_to_attack] != raw_instance.fields[field_to_attack]:
                successful = True
                break
        att_tokens = att_instance[field_to_change].tokens
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

    @lru_cache(maxsize=None)
    def neariest_neighbours(self, word, measure, topk, rho):
        # May be accelerated by caching a the distance
        vals, idxs = self.embed_searcher.find_neighbours(word, measure=measure, topk=topk, rho=rho)
        return [self.vocab.get_token_from_index(idx) for idx in cast_list(idxs)]