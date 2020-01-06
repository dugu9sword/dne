from typing import Dict, List, Union, Optional
import logging
import json
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
import pandas
import csv

logger = logging.getLogger(__name__)


class TSVReader(DatasetReader):
    """
    Reads tokens and their labels from a labeled text classification dataset.
    Expects a "text" field and a "label" field in JSON format.
    The output of ``read`` is a list of ``Instance`` s with the fields:
        tokens: ``TextField`` and
        label: ``LabelField``
    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.
        See :class:`TokenIndexer`.
    tokenizer : ``Tokenizer``, optional (default = ``{"tokens": SpacyTokenizer()}``)
        Tokenizer to use to split the input text into words or other kinds of tokens.
    segment_sentences: ``bool``, optional (default = ``False``)
        If True, we will first segment the text into sentences using SpaCy and then tokenize words.
        Necessary for some models that require pre-segmentation of sentences, like the Hierarchical
        Attention Network (https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf).
    max_sequence_length: ``int``, optional (default = ``None``)
        If specified, will truncate tokens to specified maximum length.
    skip_label_indexing: ``bool``, optional (default = ``False``)
        Whether or not to skip label indexing. You might want to skip label indexing if your
        labels are numbers, so the dataset reader doesn't re-number them starting from 0.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    """
    def __init__(
            self,
            token_cols: List[str],
            label_col: str = 'label',
            token_indexers: Dict[str, TokenIndexer] = None,
            tokenizer: Tokenizer = None,
            max_sequence_length: int = None,
            skip_label_indexing: bool = False,
            lazy: bool = False,
    ) -> None:
        super().__init__(lazy=lazy)
        self._token_cols = token_cols
        self._label_col = str
        self._tokenizer = tokenizer or WordTokenizer()
        self._max_sequence_length = max_sequence_length
        self._skip_label_indexing = skip_label_indexing
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            # without the quoting arg, errors will occur with line having quoting characters "/'
            df = pandas.read_csv(data_file, sep='\t', quoting=csv.QUOTE_NONE)
            has_label = self._label_col in df.columns
            for rid in range(1, df.shape[0]):
                tokens = {}
                for token_col in self._token_cols:
                    tokens[token_col] = df.iloc[rid][token_col]
                if has_label:
                    label = df.iloc[rid][self._label_col]
                else:
                    label = None
                instance = self.text_to_instance(tokens=tokens, label=label)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self,
                         tokens: Dict[str, str],
                         label: Optional[str] = None) -> Instance:  # type: ignore
        fields: Dict[str, Field] = {}

        for token_col in self._token_cols:
            fields[token_col] = TextField(self._tokenizer.tokenize(tokens[token_col]),
                                          self._token_indexers)

        if label is not None:
            fields[self._label_col] = LabelField(label, skip_indexing=False)
        return Instance(fields)
