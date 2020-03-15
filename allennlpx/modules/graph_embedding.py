import io
import itertools
import logging
import re
import tarfile
import warnings
import zipfile
from typing import Any, cast, IO, Iterator, NamedTuple, Optional, Sequence, Tuple

import numpy
import torch
from overrides import overrides
from torch.nn.functional import embedding

from allennlp.common import Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path, get_file_extension, is_url_or_existing_file
from allennlp.data import Vocabulary
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.nn import util

class GraphEmbedding(Embedding):
    def __init__(
        self,
        neighbours: torch.LongTensor = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)        
        assert neighbours is not None
        self.neighbours = neighbours


    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    @overrides
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        neighbour_tokens = self.neighbours[tokens]
            
        # tokens may have extra dimensions (batch_size, d1, ..., dn, sequence_length),
        # but embedding expects (batch_size, sequence_length), so pass tokens to
        # util.combine_initial_dims (which is a no-op if there are no extra dimensions).
        # Remember the original size.
        original_size = neighbour_tokens.size()
        neighbour_tokens = util.combine_initial_dims(neighbour_tokens)
        

        embedded = embedding(
            neighbour_tokens,
            self.weight,
            padding_idx=self.padding_index,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        # Now (if necessary) add back in the extra dimensions.
        embedded = util.uncombine_initial_dims(embedded, original_size)
        
        embedded = embedded[..., 0, :] * 1.0 + embedded[..., 1:4, :].mean(-2) * 0.0

        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)
        return embedded
