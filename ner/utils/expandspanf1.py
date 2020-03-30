from typing import List, Dict
from typing import Optional

from functools import lru_cache

from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.span_based_f1_measure import SpanBasedF1Measure, TAGS_TO_SPANS_FUNCTION_TYPE

class ExpandSpanBasedF1Measure(SpanBasedF1Measure):
    def __init__(self, 
                 vocabulary: Vocabulary,
                 tag_namespace: str = "tags",
                 ignore_classes: List[str] = None,
                 label_encoding: Optional[str] = "BIO",
                 tags_to_spans_function: Optional[TAGS_TO_SPANS_FUNCTION_TYPE] = None,):
        super(ExpandSpanBasedF1Measure, self).__init__(vocabulary,
                                                       tag_namespace,
                                                       ignore_classes,
                                                       label_encoding,
                                                       tags_to_spans_function)
        
    def __repr__(self):
        return " ".join("{}:{:.4f}".format(*i) for i in self.get_metric().items())
        
    def __lt__(self, other):
        return self.f1_score < other
    
    def __le__(self, other):
        return self.f1_score <= other
        
    def __ge__(self, other):
        return self.f1_score >= other
    
    def __gt__(self, other):
        return self.f1_score > other
    
    @property
    def f1_score(self):
        result = self.get_metric(reset=False)
        return result['f1-measure-overall']
    
    def __iadd__(self, other: "ExpandSpanBasedF1Measure"):
        for key, value in other._true_positives.items():
            if key in self._true_positives:
                self._true_positives[key] += value
            else:
                self._true_positives[key] = value
            
        for key, value in other._false_positives.items():
            if key in self._false_positives:
                self._false_positives[key] += value
            else:
                self._false_positives[key] = value
            
        for key, value in other._false_negatives.items():
            if key in self._false_negatives:
                self._false_negatives[key] += value
            else:
                self._false_negatives[key] = value
        return self