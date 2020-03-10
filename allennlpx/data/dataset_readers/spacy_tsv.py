import csv
import logging
from typing import Dict, Optional, List

# from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
import pandas
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer
from overrides import overrides
from typing import Callable
from copy import deepcopy
from allennlpx import allenutil

logger = logging.getLogger(__name__)


class SpacyTSVReader(DatasetReader):
    def __init__(self,
                 sent1_col: str,
                 sent2_col: str = None,
                 label_col: str = 'label',
                 max_sequence_length: int = 512,
                 skip_label_indexing: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._sent1_col = sent1_col
        self._sent2_col = sent2_col
        self._label_col = label_col
        self._tokenizer = SpacyTokenizer()
        self._max_sequence_length = max_sequence_length
        self._skip_label_indexing = skip_label_indexing

        self._token_indexers = {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            # without the quoting arg, errors will occur with line having quoting characters "/'
            df = pandas.read_csv(data_file, sep='\t', quoting=csv.QUOTE_NONE)
            has_label = self._label_col in df.columns
            for rid in range(1, df.shape[0]):
                sent1 = df.iloc[rid][self._sent1_col]

                if self._sent2_col:
                    sent2 = df.iloc[rid][self._sent2_col]
                else:
                    sent2 = None

                if has_label:
                    label = df.iloc[rid][self._label_col]
                    if self._skip_label_indexing:
                        label = int(label)
                else:
                    label = None

                instance = self.text_to_instance(sent1=sent1, sent2=sent2, label=label)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self,
                         sent1: str,
                         sent2: str = None,
                         label: Optional[str] = None) -> Instance:  # type: ignore
        fields: Dict[str, Field] = {}

        tokens1 = self._tokenizer.tokenize(sent1)
        if sent2:
            tokens2 = self._tokenizer.tokenize(sent2)

        if sent2:
            fields['sent1'] = TextField(tokens1, self._token_indexers)
            fields['sent2'] = TextField(tokens2, self._token_indexers)
        else:
            fields['sent'] = TextField(tokens1, self._token_indexers)

        if label is not None:
            fields['label'] = LabelField(label, skip_indexing=self._skip_label_indexing)
        return Instance(fields)
    
    def transform_instances(self,
                            transform: Callable[[List[str]], List[str]],
                            instances: List[Instance],
                          ) -> List[Instance]:
        # For simple transformation, a single for-loop is enough.
        # However for complex transformation such as back-translation/DAE/SpanBERT,
        # a batch version is required.
        ret_instances = deepcopy(instances)
        sents = []
        for instance in ret_instances:
            sents.append(allenutil.as_sentence(instance.fields['sent']))
        new_sents = transform(sents)
        for i, instance in enumerate(ret_instances):
            instance.fields['sent'] = TextField(self._tokenizer.tokenize(new_sents[i]), self._token_indexers)
            instance.indexed = False
#             instance.fields['sent'].tokens = self._tokenizer.tokenize(new_sents[i])
#             instance.fields['sent'].indexed = False
#             instance.fields['sent']._indexed_tokens = None
#         import pdb
#         pdb.set_trace()
        return ret_instances
