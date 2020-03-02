from copy import deepcopy
from typing import Dict, List

import numpy
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.fields import LabelField
from allennlp.data.tokenizers import SpacyTokenizer
from overrides import overrides

from allennlpx.predictors.predictor import Predictor


class TextClassifierPredictor(Predictor):
    def __init__(self, model, dataset_reader, key='sent'):
        super().__init__(model, dataset_reader)
        self.key=key

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({self.key: sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict[self.key]
        if not hasattr(self._dataset_reader, 'tokenizer') and not hasattr(self._dataset_reader, '_tokenizer'):
            tokenizer = SpacyTokenizer()
            sentence = [str(t) for t in tokenizer.tokenize(sentence)]
        return self._dataset_reader.text_to_instance(sentence)

    @overrides
    def predictions_to_labeled_instances(self,
                                         instance: Instance,
                                         outputs: Dict[str, numpy.ndarray]) -> List[Instance]:
        new_instance = deepcopy(instance)
        label = numpy.argmax(outputs['probs'])
        new_instance.add_field('label', LabelField(int(label), skip_indexing=True))
        return [new_instance]
