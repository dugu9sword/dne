from typing import Dict
import logging

import math
import random
import numpy as np
import csv

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer

logger = logging.getLogger(__name__)

@DatasetReader.register('agnews')
class AGNewsDataReader(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 lazy: bool = False,
                 crop: bool = False,
                 crop_batch_size: int = 10,
                 crop_window_size_rate: float = 0.5,
                 crop_min_window_size: int = 3
                 ) -> None:
        """
        crop: whether crop the sentence
        crop_batch_size: only used when crop is True, crop a sentence into $crop_batch_size$ sub-sentences
        crop_windows_size_rate: only used when crop is True, the length of each sub-sentence is defined by rate * sentence_length
        crop_min_windows_size: only used when crop is True, the min length of the sub-sentences
        """
        super().__init__(lazy=lazy)

        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        
        self.crop = crop
        self.crop_batch_size = crop_batch_size
        self.crop_windows_size_rate = crop_window_size_rate
        self.crop_min_windows_size = crop_min_window_size

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            csv_file = csv.reader(data_file)
            for row in csv_file:
                if self.crop:
                    tokens = self._tokenizer.tokenize("{} {}".format(row[1], row[2])) 
                    sentence_length = len(tokens)
                    window_size = int(math.ceil(sentence_length * self.crop_windows_size_rate))
                    if window_size < self.crop_min_windows_size:
                        window_size = self.crop_min_windows_size
                    window_number = sentence_length - window_size + 1
                    if window_number > self.crop_batch_size:
                        crop_start_index = random.sample(range(window_number), self.crop_batch_size)
                    else:
                        crop_start_index = list(range(window_number))
                        crop_start_index.extend(np.random.randint(0,window_number,self.crop_batch_size-window_number))
                        
                    subsentence = [" ".join(str(token)for token in tokens[index:index+window_size]) for index in crop_start_index]
                    for sentence in subsentence:
                        yield self.text_to_instance(sentence, row[0])
                else:
                    yield self.text_to_instance("{} {}".format(row[1], row[2]), row[0])
    @overrides
    def text_to_instance(self, string: str, label: str = None) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(string)
        fields['tokens'] = TextField(tokens, self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label, skip_indexing=False)
        return Instance(fields)