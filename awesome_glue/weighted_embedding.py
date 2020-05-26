import numpy as np
import torch
from overrides import overrides
from torch.nn.functional import embedding

from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.embedding import Embedding
from luna import ram_write, ram_read, ram_has_flag, ram_set_flag
from allennlpx.training import adv_utils
import torch.nn.functional as F
from .weighted_util import WeightedHull, SameAlphaHull, DecayAlphaHull
from allennlp.nn import util


class WeightedEmbedding(Embedding):
    def __init__(
        self,
        hull,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)        
        self.hull: WeightedHull = hull

    @overrides
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if not ram_has_flag("EXE_ONCE.weighted_embedding"):
            print("The weighted embedding is working")
            import sys
            sys.stdout.flush()
            ram_set_flag("EXE_ONCE.weighted_embedding")
            
        if ram_has_flag("warm_mode", True) or ram_has_flag("weighted_off", True):
            embedded = embedding(
                util.combine_initial_dims(tokens),
                self.weight,
                padding_idx=self.padding_index,
                max_norm=self.max_norm,
                norm_type=self.norm_type,
                scale_grad_by_freq=self.scale_grad_by_freq,
                sparse=self.sparse,
            )
            embedded = util.uncombine_initial_dims(embedded, tokens.size())
            return embedded
        nbr_tokens, _coeff = self.hull.get_nbr_and_coeff(tokens.view(-1))

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
            coeff_logit = (_coeff + 1e-6).log()
        else:
            last_fw, last_bw = adv_utils.read_var_hook("coeff_logit")
            # coeff_logit = last_fw + adv_utils.recieve("step") * last_bw
            norm_last_bw = last_bw / (torch.norm(last_bw, dim=-1, keepdim=True) + 1e-6)
            coeff_logit = last_fw + adv_utils.recieve("step") * norm_last_bw
        
        coeff_logit = coeff_logit - coeff_logit.max(1, keepdim=True)[0]

        coeff_logit.requires_grad_()
        adv_utils.register_var_hook("coeff_logit", coeff_logit)
        coeff = F.softmax(coeff_logit, dim=1)

        # if adv_utils.is_adv_mode():
        #     last_coeff = F.softmax(last_fw, dim=1)
        #     new_points = (embedded[:20] * coeff[:20].unsqueeze(-1)).sum(-2)
        #     old_points = (embedded[:20] * last_coeff[:20].unsqueeze(-1)).sum(-2)
        #     step_size = (new_points - old_points).norm(dim=-1).mean()
        #     inner_size = (embedded[:20, 1:] - embedded[:20, :1]).norm(dim=-1).mean()
        #     print(round(inner_size.item(), 3), round(step_size.item(), 3))
        embedded = (embedded * coeff.unsqueeze(-1)).sum(-2)
        embedded = embedded.view(*tokens.size(), self.weight.size(1))
        if adv_utils.is_adv_mode():
            if ram_read("adjust_point"):
                raw_embedded = embedding(
                    tokens,
                    self.weight,
                    padding_idx=self.padding_index,
                    max_norm=self.max_norm,
                    norm_type=self.norm_type,
                    scale_grad_by_freq=self.scale_grad_by_freq,
                    sparse=self.sparse,
                )
                delta = embedded.detach() - raw_embedded.detach()
                embedded = raw_embedded + delta
        return embedded
