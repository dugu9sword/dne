from typing import Dict, List, Optional, Set, Union
from collections import defaultdict

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric
from allennlp.data.dataset_readers.dataset_utils.span_utils import (
    bio_tags_to_spans,
    bioul_tags_to_spans,
    iob1_tags_to_spans,
    bmes_tags_to_spans,
    TypedStringSpan,
)
NON_NAMED_ENTITY_LABEL = 'O'

@Metric.register("span_accuracy")
class SpanBasedAccuracyMeasure(Metric):
    def __init__(
        self,
        vocabulary: Vocabulary,
        tag_namespace: str = "tags",
        ignore_classes: List[str] = None,
        label_encoding: str = "BIOUL",
    ) -> None:
        """
        # Parameters

        vocabulary : `Vocabulary`, required.
            A vocabulary containing the tag namespace.
        tag_namespace : str, required.
            This metric assumes that a BIO format is used in which the
            labels are of the format: ["B-LABEL", "I-LABEL"].
        ignore_classes : List[str], optional.
            Span labels which will be ignored when computing span metrics.
            A "span label" is the part that comes after the BIO label, so it
            would be "ARG1" for the tag "B-ARG1". For example by passing:

             `ignore_classes=["V"]`
            the following sequence would not consider the "V" span at index (2, 3)
            when computing the precision, recall and F1 metrics.

            ["O", "O", "B-V", "I-V", "B-ARG1", "I-ARG1"]

            This is helpful for instance, to avoid computing metrics for "V"
            spans in a BIO tagging scheme which are typically not included.
        label_encoding : `str`, optional (default = "BIO")
            The encoding used to specify label span endpoints in the sequence.
            Valid options are "BIO", "IOB1", "BIOUL" or "BMES".
        tags_to_spans_function : `Callable`, optional (default = `None`)
            If `label_encoding` is `None`, `tags_to_spans_function` will be
            used to generate spans.
        """
        if label_encoding:
            if label_encoding not in ["BIO", "IOB1", "BIOUL", "BMES"]:
                raise ConfigurationError(
                    "Unknown label encoding - expected 'BIO', 'IOB1', 'BIOUL', 'BMES'."
                )
            
        self._label_encoding = label_encoding
        self._label_vocabulary = vocabulary.get_index_to_token_vocabulary(tag_namespace)
        self._ignore_classes: List[str] = ignore_classes or []

        # These will hold per label span counts.
        self._correct = 0
        self._total = 0

    def __call__(
        self,
        predictions: Union[torch.Tensor, List[str]],
        gold_labels: Union[torch.Tensor, List[str]],
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, sequence_length, num_classes).
        gold_labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, sequence_length). It must be the same
            shape as the `predictions` tensor without the `num_classes` dimension.
        mask : `torch.BoolTensor`, optional (default = None).
            A masking tensor the same size as `gold_labels`.
        prediction_map : `torch.Tensor`, optional (default = None).
            A tensor of size (batch_size, num_classes) which provides a mapping from the index of predictions
            to the indices of the label vocabulary. If provided, the output label at each timestep will be
            `vocabulary.get_index_to_token_vocabulary(prediction_map[batch, argmax(predictions[batch, t]))`,
            rather than simply `vocabulary.get_index_to_token_vocabulary(argmax(predictions[batch, t]))`.
            This is useful in cases where each Instance in the dataset is associated with a different possible
            subset of labels from a large label-space (IE FrameNet, where each frame has a different set of
            possible roles associated with it).
        """
        # if mask is None:
        #     mask = torch.ones_like(gold_labels).bool()
        

        # predictions, gold_labels, mask, prediction_map = self.detach_tensors(
        #     predictions, gold_labels, mask, prediction_map
        # )
        # 
        # sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        # argmax_predictions = predictions.max(-1)[1]
        # 
        # argmax_predictions = argmax_predictions.float()

        # Iterate over timesteps in batch.
        # batch_size = gold_labels.size(0)
        # for i in range(batch_size):
        #     sequence_prediction = argmax_predictions[i, :]
        #     sequence_gold_label = gold_labels[i, :]
        #     length = sequence_lengths[i]
        # 
        #     if length == 0:
        #         # It is possible to call this metric with sequences which are
        #         # completely padded. These contribute nothing, so we skip these rows.
        #         continue
        # 
        #     predicted_string_labels = [
        #         self._label_vocabulary[label_id]
        #         for label_id in sequence_prediction[:length].tolist()
        #     ]
        #     gold_string_labels = [
        #         self._label_vocabulary[label_id]
        #         for label_id in sequence_gold_label[:length].tolist()
        #     ]

        tags_to_spans_function = None
        # `label_encoding` is empty and `tags_to_spans_function` is provided.
        if self._label_encoding is None and self._tags_to_spans_function:
            tags_to_spans_function = self._tags_to_spans_function
        # Search by `label_encoding`.
        elif self._label_encoding == "BIO":
            tags_to_spans_function = bio_tags_to_spans
        elif self._label_encoding == "IOB1":
            tags_to_spans_function = iob1_tags_to_spans
        elif self._label_encoding == "BIOUL":
            tags_to_spans_function = bioul_tags_to_spans
        elif self._label_encoding == "BMES":
            tags_to_spans_function = bmes_tags_to_spans

        predicted_spans = tags_to_spans_function(predictions, self._ignore_classes)
        gold_spans = tags_to_spans_function(gold_labels, self._ignore_classes)

        predicted_spans = self.handle_with_non_named_entity(predictions, predicted_spans)
        gold_spans = self.handle_with_non_named_entity(gold_labels, gold_spans)

        for span in gold_spans:
            if span in predicted_spans:
                self._correct += 1
        self._total += len(gold_spans)
    
    def handle_with_non_named_entity(self, tags: List[str], spans: List[TypedStringSpan]):
        for index, tag in enumerate(tags):
            if tag == NON_NAMED_ENTITY_LABEL:
                spans.append((NON_NAMED_ENTITY_LABEL, (index, index)))
        return spans
            

    def get_metric(self, reset: bool = False):
        all_metric = {'accuracy': self._correct / self._total}
        if reset:
            self.reset()
        return all_metric

    def reset(self):
        self._correct = 0
        self._total = 0

    def __lt__(self, other):
        return self.accuracy < other

    def __le__(self, other):
        return self.accuracy <= other

    def __ge__(self, other):
        return self.accuracy >= other

    def __gt__(self, other):
        return self.accuracy > other

    @property
    def accuracy(self):
        return self._correct / self._total