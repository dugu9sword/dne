import numpy as np
import torch
from overrides import overrides
from torch.nn.functional import embedding

from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.nn import util


class DirichletEmbedding(Embedding):
    def __init__(
        self,
        temperature,
        neighbours,
        nbr_mask,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)        
        self.neighbours = neighbours
        self.temperature = temperature
        self.current_temperature = temperature
        self.nbr_mask = nbr_mask

    @overrides
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        neighbour_num = self.neighbours.size(1)
        neighbour_tokens = self.neighbours[tokens]
        neighbour_mask = self.nbr_mask[tokens]
        
#         if tokens.size(1) > 1:
#             import pdb; pdb.set_trace()
            
        # tokens may have extra dimensions (batch_size, d1, ..., dn, sequence_length),
        # but embedding expects (batch_size, sequence_length), so pass tokens to
        # util.combine_initial_dims (which is a no-op if there are no extra dimensions).
        # Remember the original size.
        tmp_tokens = neighbour_tokens.view(-1, neighbour_num)

        # embedded = embedding(tmp_tokens, self.weight)
        embedded = embedding(
            tmp_tokens,
            self.weight,
            padding_idx=self.padding_index,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )
        
        # Now (if necessary) add back in the extra dimensions.
        alphas = np.ones(neighbour_num) / neighbour_num
        if self.current_temperature != 0.0:
            alphas = alphas / self.current_temperature
            coeff = np.random.dirichlet(alphas, embedded.size(0)).astype(np.float32)
            coeff = torch.from_numpy(coeff).to(self.weight.device)
        else:
            # sanity checks, this degrades to "mean"
            coeff = torch.from_numpy(alphas.astype(np.float32)).to(self.weight.device)
        embedded = (embedded * coeff.unsqueeze(-1)).sum(-2, keepdim=True)
        # embedded = embedded[:, 0, :] + embedded[:, 1:, :].sum(-2).detach()

        embedded = embedded.view(*tokens.size(), embedded.size(-1))

        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)
        return embedded

    

