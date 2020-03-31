from copy import deepcopy
from typing import Dict, List

import numpy
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.fields import LabelField
from allennlp.data.tokenizers import SpacyTokenizer
from overrides import overrides

from allennlpx.predictors.predictor import Predictor


class BiTextClassifierPredictor(Predictor):
    def __init__(self, model, dataset_reader, key1='sent1', key2='sent2'):
        super().__init__(model, dataset_reader)
        self.key1 = key1
        self.key2 = key2

    def predict(self, sentence1: str, sentence2: str) -> JsonDict:
        return self.predict_json({self.key1: sentence1, self.key2: sentence2})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(json_dict[self.key1], json_dict[self.key2])

    @overrides
    def predictions_to_labeled_instances(self,
                                         instance: Instance,
                                         outputs: Dict[str, numpy.ndarray]) -> List[Instance]:
        new_instance = deepcopy(instance)
        label = numpy.argmax(outputs['probs'])
        new_instance.add_field('label', LabelField(int(label), skip_indexing=True))
        return [new_instance]
