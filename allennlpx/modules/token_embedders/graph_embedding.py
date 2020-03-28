
import torch
from overrides import overrides
from torch.nn.functional import embedding

from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.nn import util


class GraphEmbedding(Embedding):
    def __init__(
        self,
        gnn,
        edges,
        hop,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)        
        self.edges = edges
        self.gnn = gnn
        self.hop = hop

    @overrides
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens may have extra dimensions (batch_size, d1, ..., dn, sequence_length),
        # but embedding expects (batch_size, sequence_length), so pass tokens to
        # util.combine_initial_dims (which is a no-op if there are no extra dimensions).
        # Remember the original size.
        original_size = tokens.size()
        tokens = util.combine_initial_dims(tokens)
        
        weight = self.weight
        weight_device = util.get_device_of(weight)
        if util.get_device_of(self.edges) != weight_device:
            self.edges = util.move_to_device(self.edges, weight_device)

        for _ in range(self.hop):
            weight = self.gnn(weight, self.edges.t())

        embedded = embedding(
            tokens,
            weight,
            padding_idx=self.padding_index,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        # Now (if necessary) add back in the extra dimensions.
        embedded = util.uncombine_initial_dims(embedded, original_size)

        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)
        return embedded
