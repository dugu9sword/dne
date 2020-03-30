from typing import List
from overrides import overrides
import numpy as np
from allennlp.common.util import JsonDict
from allennlp.data.tokenizers.token import Token
from allennlp.data.instance import Instance
from allennlp.predictors.predictor import Predictor

from allennlp.models.model import Model
from allennlp.data.dataset_readers import DatasetReader


class NerPredictor(Predictor):
    def __init__(self, model: Model, data_reader: DatasetReader): 
        super().__init__(model, data_reader)
    
    def text_to_instance(self, tokens: List[str]):
        return self._dataset_reader.text_to_instance([Token(token) for token in tokens])
    
    # remove santitize
    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return outputs

    # remove santitize
    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return outputs
    
    # for each sentence: 
    # sentence_logit = sum(max(logit(non gold)) - logit(gold)) for each word
    # can be defined by other way
    def calculate_sentence_logit(self, logits, gold_labels_index):
        gold_label_logit = logits[np.arange(len(gold_labels_index)),gold_labels_index]
        logits[np.arange(len(gold_labels_index)),gold_labels_index] = -10000.0
        non_gold_label_logit = np.max(logits, axis=1)
        return np.sum(non_gold_label_logit - gold_label_logit)