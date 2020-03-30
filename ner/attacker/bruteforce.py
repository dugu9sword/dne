import numpy as np
from overrides import overrides
from copy import deepcopy
from typing import Dict, Any
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.predictors.predictor import Predictor
from allennlp.modules.token_embedders.embedding import Embedding
from ner.attacker.attacker import Attacker
from ner.utils.spanaccuracy import SpanBasedAccuracyMeasure


class BruteForce(Attacker):
    def __init__(self,
                 predictor: Predictor,
                 vocab: Vocabulary,
                 ignores: Dict[str, int],
                 attack_rate: float = 0.3,
                 attack_number: int = 512, 
                 min_sentence_length: int = 0,
                 policy: str = 'random',
                 embedding: Embedding = None):
        # embeding : only used when policy == 'embedding'
        super().__init__(predictor, vocab, ignores, attack_rate, min_sentence_length, policy, embedding)
        self.attack_number = attack_number
        named_entity_index_to_token = {v: k for k,v in self.ignores.items()}
        # non named entity index, which can be replaced
        self.non_named_entity_index = [i for i in range(self.vocab.get_vocab_size('tokens')) if i not in named_entity_index_to_token]
         
    def brute_force_candidates(self, max_change_num, adv_index, adv_numbers):
        adv_word_indexes = np.concatenate(
            [np.random.choice(adv_index, size=[1, max_change_num], replace=False) 
             for _ in range(adv_numbers)], axis=0)
        adv_word_list = np.random.choice(self.non_named_entity_index, size=[adv_numbers, max_change_num])
        return adv_word_indexes, adv_word_list
    
    @overrides
    def attack_on_instance(self,
                           instance: Instance,
                           tokens_namespace: str = 'tokens',
                           label_namespace: str = 'tags') -> Dict[str, Any]:
        raw_tokens = [token.text for token in instance.fields[tokens_namespace]]
        gold_tags = [tag for tag in instance.fields[label_namespace]]
        # gold_tags_index = [self.vocab.get_token_index(tag, label_namespace) for tag in gold_tags]

        # raw metric on span-based accuracy
        raw_result = self.predictor.predict_instance(instance)
        raw_metric = SpanBasedAccuracyMeasure(self.vocab)
        raw_metric(raw_result['tags'], gold_tags, raw_result['mask'])

        ids_to_change = self.attackable_ids(gold_tags)
        if len(raw_tokens) < self.min_sentence_length or len(ids_to_change) == 0:
            return {'raw_tokens': instance,
                    'raw_result': raw_result,
                    'gold_tags': gold_tags,
                    'attack_flag': 0
                    }
        
        max_change_num = min(self.max_change_num(len(raw_tokens)), len(ids_to_change))
        adv_indexes, adv_words = self.brute_force_candidates(max_change_num, ids_to_change, self.attack_number)
        
        adv_instances = []
        label_to_token_vocab = self.vocab.get_index_to_token_vocabulary(tokens_namespace)
        for adv_index, adv_word in zip(adv_indexes, adv_words):
            adv_tokens = deepcopy(raw_tokens)
            for index, word in zip(adv_index, adv_word):
                adv_tokens[index] = label_to_token_vocab[word]
            adv_instance = self.predictor.text_to_instance(adv_tokens)
            adv_instances.append(adv_instance)

        adv_results = self.generate_results(adv_instances)
        
        adv_instance_min = None
        adv_result_min = None
        adv_metric_min = None
        successful = False
        for adv_instance, adv_result in zip(adv_instances, adv_results):
            adv_metric = SpanBasedAccuracyMeasure(self.vocab)
            adv_metric(adv_result['tags'], gold_tags, adv_result['mask'])
            if adv_metric < raw_metric:
                successful = True
                if adv_metric_min == None or adv_metric < adv_metric_min:
                    adv_instance_min = adv_instance
                    adv_metric_min = adv_metric
                    adv_result_min = adv_result
                    if adv_metric_min.accuracy == 0:
                        break

        return {"raw_tokens": instance,
                "raw_result": raw_result,
                "adv_tokens": adv_instance_min,
                "adv_result": adv_result_min,
                'gold_tags': gold_tags,
                'attack_flag': 1,
                "success": 1 if successful else 0,
                "changed": max_change_num}