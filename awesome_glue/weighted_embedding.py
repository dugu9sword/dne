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

class WeightedEmbedding(Embedding):
    def __init__(
        self,
        alpha,
        neighbours,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)        
        self.neighbours = neighbours
        self.alpha = alpha

    @overrides
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        max_nbr_num = self.neighbours.size(1)        
        nbr_tokens = self.neighbours[tokens].view(-1, max_nbr_num)
        nbr_num_lst = (nbr_tokens != 0).sum(dim=1).tolist()

        # n_words x n_nbrs x dim
        embedded = embedding(
            nbr_tokens,
            self.weight,
            padding_idx=self.padding_index,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )
        
        if not adv_utils.is_adv_mode():
            _coeff = dirichlet_sampling(nbr_num_lst, self.alpha, max_nbr_num)
            _coeff = torch.from_numpy(_coeff).to(self.weight.device)        
            coeff_logit = (_coeff + 1e-6).log()
        else:
            last_fw, last_bw = adv_utils.read_var_hook("coeff_logit")
            grad_norm = torch.norm(last_bw, dim=-1, keepdim=True) + 1e-6
            coeff_logit = last_fw + adv_utils.recieve("step") * last_bw / grad_norm
        
        coeff_logit.requires_grad_()
        adv_utils.register_var_hook("coeff_logit", coeff_logit)
        coeff = F.softmax(coeff_logit, dim=1)
        embedded = (embedded * coeff.unsqueeze(-1)).sum(-2)
        embedded = embedded.view(*tokens.size(), self.weight.size(1))
        return embedded
