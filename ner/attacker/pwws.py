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

class PWWS(Attacker):
    def __init__(self, 
                 predictor: Predictor,
                 vocab: Vocabulary, 
                 ignores: Dict[str, int],
                 attack_rate: float = 0.3,
                 min_sentence_length: int = 0,
                 policy: str = 'random',
                 embedding: Embedding = None):
        # embeding : only used when policy == 'embedding'
        super().__init__(predictor, vocab, ignores, attack_rate, min_sentence_length, policy, embedding)
        
        
    @overrides
    def attack_on_instance(self, 
                           instance: Instance,
                           tokens_namespace: str = 'tokens',
                           label_namespace: str = 'tags') -> Dict[str, Any]:
        raw_tokens = [token.text for token in instance.fields[tokens_namespace]]
        gold_tags = [tag for tag in instance.fields[label_namespace]]
        gold_tags_index = [self.vocab.get_token_index(tag, label_namespace) for tag in gold_tags]
        
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

        # # 1. Replace each word with <UNK> and other candidate words
        # # 2. Generate all sentences, then concatenate them into a
        # #    list for batch forwarding
        instances = [] # concatenate all instances
        offsets = {} # {sid: [start_offset, number_of_instances]}
        candidates_dict = {}
        for ids in ids_to_change:
            tmp_instances = []
            tmp_instances.append(self.predictor.text_to_instance(raw_tokens))
            tmp_tokens = deepcopy(raw_tokens)
            tmp_tokens[ids] = DEFAULT_OOV_TOKEN
            tmp_instances.append(self.predictor.text_to_instance(tmp_tokens))
            
            candidate_words = self.generate_candidate(raw_tokens[ids].lower(), number=30)
            candidates_dict[ids] = candidate_words
            for word in candidate_words:
                tmp_tokens[ids] = word
                tmp_instances.append(self.predictor.text_to_instance(tmp_tokens))
            offsets[ids] = (len(instances), len(tmp_instances))
            instances.extend(tmp_instances)

        results_batch = self.generate_results(instances)
        
        pwws_dict = {}
        replace_dict = {}
        for sid in ids_to_change:
            _start, _num = offsets[sid]
            results = results_batch[_start:_start + _num]
            # sentence_logit = sum(max(logit(non gold)) - logit(gold)) for each word
            probs = [self.predictor.calculate_sentence_logit(result['logits'], gold_tags_index, sid) for result in results]
            # the larger the probs is, the better.
            raw_prob = probs[0]
            oov_prob = probs[1]
            other_probs = probs[2:]
            
            replace_dict[sid] = candidates_dict[sid][np.argmax(other_probs)]
            pwws_dict[sid] = np.min(raw_prob - other_probs) * np.exp(raw_prob - oov_prob)

        max_change_num = min(self.max_change_num(len(raw_tokens)), len(ids_to_change))

        adv_tokens = deepcopy(raw_tokens)
        sorted_pwws = sorted(pwws_dict.items(), key=lambda x: x[1])
        successful = False
        for i in range(max_change_num):
            sid = sorted_pwws[i][0]
            adv_tokens[sid] = replace_dict[sid]
            adv_instance = self.predictor.text_to_instance(adv_tokens)
            adv_result = self.predictor.predict_instance(adv_instance)
            adv_metric = SpanBasedAccuracyMeasure(self.vocab)
            adv_metric(adv_result['tags'], gold_tags, adv_result['mask'])
            if adv_metric < raw_metric:
                successful = True
                break

        return {"raw_tokens": instance,
                "raw_result": raw_result,
                "adv_tokens": adv_instance,
                "adv_result": adv_result,
                'gold_tags': gold_tags,
                'attack_flag': 1,
                "success": 1 if successful else 0,
                "changed": i + 1}