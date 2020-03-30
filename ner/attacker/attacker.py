import math
import numpy as np
from typing import Dict
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.predictors.predictor import Predictor
from allennlp.modules.token_embedders.embedding import Embedding
from ner.attacker.policies.policy import get_policy_from_str
from ner.utils.utils import NON_NAMED_ENTITY_LABEL

class Attacker(object):
    def __init__(self, 
                 predictor: Predictor,
                 vocab: Vocabulary, 
                 ignores: Dict[str, int],
                 attack_rate: float = 0.3,
                 min_sentence_length: int = 0, 
                 policy='random',
                 embedding: Embedding = None):
        self.predictor = predictor
        self.vocab = vocab
        self.ignores = ignores # ignores and forbidden tokens(all are named entities)
        self.min_sentence_length = min_sentence_length # min sentence length to be attack
        # word substition policies, default is random
        # current policies including 
        self.attack_rate = attack_rate
        self.policy = get_policy_from_str(vocab, policy, embedding)
        
    def attackable_ids(self, tags, distance:int = 0):
        named_entity_index = set()
        sentence_length = len(tags)
        for index, tag in enumerate(tags):
            if tag != NON_NAMED_ENTITY_LABEL:
                # delete left word 
                for left_distance in range(1, distance + 1):
                    if index - left_distance >= 0:
                        named_entity_index.add(index - left_distance)
                named_entity_index.add(index)
                # delete right word
                for right_distance in range(1, distance + 1):
                    if index + right_distance < sentence_length:
                        named_entity_index.add(index + right_distance)
        attackable_index = [i for i in range(sentence_length) if i not in named_entity_index]
        return attackable_index

    def generate_candidate(self, token: str, number: int = 30):
        tmp_candidates = self.policy.generate(token, 100)
        candidates = []
        for tmp_candidate in tmp_candidates:
            if tmp_candidate not in self.ignores:
                candidates.append(tmp_candidate)
                if len(candidates) >= number:
                    break
        if len(candidates) == 0:
            random_int = np.random.randint(2, self.vocab.get_vocab_size('tokens'), size=(100,))
            index_to_token_vocab = self.vocab.get_index_to_token_vocabulary('tokens')
            random_candidate_words = [index_to_token_vocab[rand_int] for rand_int in random_int]
            for random_candidate in random_candidate_words:
                if random_candidate not in self.ignores:
                    candidates.append(random_candidate)
                    if len(candidates) >= number:
                        break
            return candidates
        else:
            return candidates
        
    def generate_results(self, instances, batch_size=300):
        batch_number = len(instances) // batch_size
        results = []
        for i in range(batch_number):
            results.extend(self.predictor.predict_batch_instance(instances[i*batch_size:(i+1)*batch_size]))
        if batch_number * batch_size < len(instances):
            results.extend(self.predictor.predict_batch_instance(instances[batch_number * batch_size:]))
        return results
    
    def max_change_num(self, sentence_length: int) -> int: 
        return int(math.ceil(sentence_length * self.attack_rate))
        
    def attack_on_instance(self, 
                           instance: Instance,
                           tokens_namespace: str = 'tokens',
                           label_namespace: str = 'tags'):
        raise NotImplementedError()