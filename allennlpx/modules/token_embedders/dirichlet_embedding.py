import numpy as np
import torch
from overrides import overrides
from torch.nn.functional import embedding

from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.embedding import Embedding
from luna import ram_write


class DirichletEmbedding(Embedding):
    def __init__(
        self,
        temperature,
        neighbours,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)        
        self.neighbours = neighbours
        self.temperature = temperature
        self.current_temperature = temperature

    @overrides
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        neighbour_num = self.neighbours.size(1)
        neighbour_tokens = self.neighbours[tokens]
        
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
        
        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)
        
        alphas = np.ones(neighbour_num) / neighbour_num
        if self.current_temperature != 0.0:
            alphas = alphas / self.current_temperature
            coeff = np.random.dirichlet(alphas, embedded.size(0)).astype(np.float32)
            coeff = torch.from_numpy(coeff).to(self.weight.device)
        else:
            # sanity checks, this degrades to "mean"
            coeff = torch.from_numpy(alphas.astype(np.float32)).expand_as(tmp_tokens).to(self.weight.device)
        
        zero_mask = tmp_tokens == 0

        coeff.masked_fill_(zero_mask, 0.)
        coeff[:, 0] += 1e-6
        coeff = coeff / coeff.sum(dim=1, keepdims=True)

        dist = (embedded[:, 0:1, :] - embedded).norm(dim=-1)
        regularization = dist.masked_fill(zero_mask, 0.).sum() / torch.sum(~zero_mask)
        ram_write("dist_reg", regularization)
        
#         embedded = embedded[:, 0, :]
        embedded = (embedded * coeff.unsqueeze(-1)).sum(-2)
        # embedded = embedded[:, 0, :] + embedded[:, 1:, :].sum(-2).detach()

        embedded = embedded.view(*tokens.size(), embedded.size(-1))
        
        return embedded

    

# zero_mask = tmp_tokens == 0
# nonzero_num = (~zero_mask).sum(dim=1).tolist()
# cnter = Counter(nonzero_num)
# coeffs_dct = {}
# for k in cnter:
#     alphas = np.ones(k) / k
#     alphas = alphas / current_temperature
#     coeff = np.random.dirichlet(alphas, cnter[k]).astype(np.float32).tolist()
#     coeffs_dct[k] = coeff
# coeffs = []
# for n in nonzero_num:
#     if n == 0:
#         coeffs.append([1])
#     else:
#         coeffs.append(coeffs_dct[n].pop())
# coeffs = batch_pad(coeffs, 0, pad_len=neighbour_num)
# coeffs = torch.tensor(coeffs)