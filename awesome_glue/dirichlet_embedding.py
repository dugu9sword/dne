import numpy as np
import torch
from overrides import overrides
from torch.nn.functional import embedding

from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.embedding import Embedding
from luna import ram_write
from allennlpx.training import adv_utils
import torch.nn.functional as F
from awesome_glue.utils import dirichlet_sampling

class DirichletEmbedding(Embedding):
    def __init__(
        self,
        alpha,
        neighbours,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)        
        self.neighbours = neighbours
        self.alphas = alpha
        self._current_alpha = alpha

    @overrides
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        max_neighbour_num = self.neighbours.size(1)
        neighbour_tokens = self.neighbours[tokens]
        
        # n_words x n_samples
        tmp_tokens = neighbour_tokens.view(-1, max_neighbour_num)
        neighbour_num_lst = (tmp_tokens != 0).sum(dim=1).tolist()

        # n_words x n_nbrs x dim
        embedded = embedding(
            tmp_tokens,
            self.weight,
            padding_idx=self.padding_index,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        if not adv_utils.is_adv_mode():
            n_samples = 1
        else:
            n_samples = 100

        coeff = dirichlet_sampling(neighbour_num_lst * n_samples, self._current_alpha, max_neighbour_num)
        coeff = torch.from_numpy(coeff).to(self.weight.device)
        # n_words x n_samples x n_nbrs
        coeff = coeff.view(embedded.size(0), n_samples, -1)
        
        # n_words x n_samples x 1
        zero_mask = (tmp_tokens == 0).unsqueeze(1)
        coeff.masked_fill_(zero_mask, 0.)

        coeff[:, :, 0] += 1e-6
        coeff = coeff / coeff.sum(dim=2, keepdims=True)

        # n_words x n_samples x dim
        cand_embedded = (embedded.unsqueeze(1) * coeff.unsqueeze(-1)).sum(-2)
            
        if not adv_utils.is_adv_mode():
            embedded = cand_embedded[:, 0, :]
            embedded = embedded.view(*tokens.size(), self.weight.size(1))
        else:
            last_fw, last_bw = adv_utils.read_var_hook("embedding")
            grad_norm = torch.norm(last_bw, dim=-1, keepdim=True) + 1e-6
            # n_words x dim
            last_embedded = last_fw.view(-1, self.weight.size(1))
            new_embedded = last_fw + adv_utils.recieve("step") * last_bw / grad_norm
            # n_words x dim
            new_embedded = new_embedded.view(-1, self.weight.size(1))
            
            # avg_dist = (cand_embedded - last_embedded.unsqueeze(1)).norm(dim=2)
            # print("|cand-last|", avg_dist.mean().item(), avg_dist.min().item())
            # avg_dist = (cand_embedded[:, 1:, :] - cand_embedded[:, 0:1 ,:]).norm(dim=2)
            # print("|inner_cand|", avg_dist.mean().item(), avg_dist.min().item())
            
            # n_words x n_samples
            distance = (cand_embedded - new_embedded.unsqueeze(1)).norm(dim=2)
            # n_words x 1 x 1 -> n_words x 1 x dim
            dummy = distance.min(1)[1].unsqueeze(-1).unsqueeze(-1)
            dummy = dummy.expand(dummy.shape[0], dummy.shape[1], self.weight.size(1))
            embedded = cand_embedded.gather(1, dummy).squeeze(1)
            
            # rdm_delta = cand_embedded[:, 0, :] - new_embedded
            # rdm_delta = rdm_delta.norm(dim=1).mean()
            # delta = embedded - new_embedded
            # delta = delta.norm(dim=1).mean()
            # print("rdm_δ ", rdm_delta.item(), "min_δ ",  delta.item())
            embedded = embedded.view(*tokens.size(), embedded.size(-1))
        
        adv_utils.register_var_hook("embedding", embedded)
        return embedded

