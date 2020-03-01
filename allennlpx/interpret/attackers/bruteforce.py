# pylint: disable=protected-access
import random
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


class BruteForce(Attacker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def attack_from_json(self,
                         inputs: JsonDict = None,
                         field_to_change: str = 'tokens',
                         field_to_attack: str = 'label',
                         grad_input_field: str = 'grad_input_1',
                         policy: CandidatePolicy= None,
                         search_num: int = 256) -> JsonDict:
        if self.token_embedding is None:
            raise Exception('initialize it first~')

        raw_instance = self.predictor.json_to_labeled_instances(inputs)[0]
        raw_tokens = list(map(lambda x: x.text, self.spacy.tokenize(inputs[field_to_change])))

        # Select words that can be changed
        sids_to_change = []
        nbr_dct = defaultdict(lambda: [])
        for i in range(len(raw_tokens)):
            if raw_tokens[i] not in self.ignore_tokens:
                word = raw_tokens[i]
                if isinstance(policy, EmbeddingPolicy):
                    nbrs = self.neariest_neighbours(word, policy.measure, policy.topk, policy.rho)
                elif isinstance(policy, SynonymPolicy):
                    nbrs = self.synom_searcher.search(word)
                    
                nbrs = [nbr for nbr in nbrs if nbr not in self.forbidden_tokens]
                if len(nbrs) > 0:
                    sids_to_change.append(i)
                    nbr_dct[i] = nbrs
                    
        # max number of tokens that can be changed
        max_change_num = min(self.max_change_num(len(raw_tokens)), len(sids_to_change))

        # Construct adversarial instances
        adv_instances = []
        for i in range(search_num):
            adv_tokens = [ele for ele in raw_tokens]
            word_sids = random.choices(sids_to_change, k=max_change_num)
            for word_sid in word_sids:
                adv_tokens[word_sid] = random.choice(nbr_dct[word_sid])
            adv_instances.append(
                self.predictor._dataset_reader.text_to_instance(" ".join(adv_tokens)))

        # Checking attacking status, early stop
        successful = False
        results = self.predictor._model.forward_on_instances(adv_instances)
        for i, result in enumerate(results):
            adv_instance = self.predictor.predictions_to_labeled_instances(adv_instances[i], result)[0]
            if adv_instance[field_to_attack].label != raw_instance[field_to_attack].label:
                successful = True
                break
        adv_tokens = adv_instances[i][field_to_change].tokens
        outputs = result

        
        return sanitize({
            "adv": adv_tokens,
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
