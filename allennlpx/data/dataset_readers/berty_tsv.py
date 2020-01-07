from typing import Dict, List, Union, Optional
import logging
import json
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
import pandas
import csv
from pytorch_pretrained_bert import BertTokenizer
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.data.tokenizers.token import Token

logger = logging.getLogger(__name__)


class BertyTSVReader(DatasetReader):
    def __init__(
            self,
            sent1_col: str,
            sent2_col: str = None,
            label_col: str = 'label',
            bert_model: str = 'bert-base-uncased',
            max_sequence_length: int = 512,
            skip_label_indexing: bool = False,
            lazy: bool = False,
    ) -> None:
        super().__init__(lazy=lazy)
        self._sent1_col = sent1_col
        self._sent2_col = sent2_col
        self._label_col = label_col
        self._tokenizer = BertTokenizer.from_pretrained(bert_model)
        self._max_sequence_length = max_sequence_length
        self._skip_label_indexing = skip_label_indexing
        self._token_indexers = {
            "berty_tokens": PretrainedBertIndexer(pretrained_model=bert_model, do_lowercase=True)
        }

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

        tokens = self._tokenizer.tokenize(sent1)
        if sent2:
            tokens +=  ['[SEP]'] + self._tokenizer.tokenize(sent2) 
        # tokens = tokens[:self._max_sequence_length]
        tokens = [Token(x) for x in tokens]
        # tokens = tokens[:512]

        fields['berty_tokens'] = TextField(tokens, self._token_indexers)
        
        if label is not None:
            fields['label'] = LabelField(label, skip_indexing=self._skip_label_indexing)
        return Instance(fields)
