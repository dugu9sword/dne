from dataclasses import dataclass
from allennlpx.interpret.attackers.cached_searcher import CachedIndexSearcher
from allennlpx.interpret.attackers.policies import EmbeddingPolicy
import torch
from luna import batch_pad


@dataclass
class AdvTrainingPolicy:
    normal_iteration: int = 1
    adv_iteration: int = 1


@dataclass
class HotFlipPolicy(AdvTrainingPolicy):
    searcher: CachedIndexSearcher = None
    replace_num: int = None


@dataclass
class NeariestNeighbourPolicy(AdvTrainingPolicy):
    searcher: CachedIndexSearcher = None
    replace_num: int = None


def apply_constraint(searcher, src_tokens, scores):
    mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    src_tokens_lst = src_tokens.tolist()
    idxes_to_mask = []
    for bid in range(src_tokens.size(0)):
        for sid in range(src_tokens.size(1)):
            if src_tokens_lst[bid][sid] == 0:
                idxes_to_mask.append([])
                continue
            _, idxs = searcher.search(src_tokens_lst[bid][sid])
            idxes_to_mask.append(idxs)
    idxes_to_mask = src_tokens.new_tensor(batch_pad(idxes_to_mask, 0))
    idxes_to_mask = idxes_to_mask.view(*src_tokens.size(), -1)
    mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    mask.scatter_(dim=2,
                  index=idxes_to_mask,
                  src=idxes_to_mask.new_ones(idxes_to_mask.size(), dtype=torch.bool))
    mask[:, :, 0] = False
    mask = ~mask
    scores.masked_fill_(mask, -19260817.)


def hotflip(*, src_tokens, embeds, grads, embedding_matrix, replace_num=3, constrain_fn_=None):
    replace_num = min(replace_num, src_tokens.size(1))

    # compute the direction vector dot the gradient
    prev_embed_dot_grad = torch.einsum("bij,bij->bi", grads, embeds)
    new_embed_dot_grad = torch.einsum("bij,kj->bik", grads, embedding_matrix)
    dir_dot_grad = new_embed_dot_grad - prev_embed_dot_grad.unsqueeze(-1)
    dir_dot_grad[src_tokens == 0] = -19260817.

    # maybe some constraints
    if constrain_fn_ is not None:
        constrain_fn_(src_tokens, dir_dot_grad)

    # select the top-k step
    score_at_each_step, best_at_each_step = dir_dot_grad.max(2)
    _, best_positions = score_at_each_step.topk(replace_num)

    # use the selected token index to replace the original one
    adv_tokens = src_tokens.clone()
    src = best_at_each_step.gather(dim=1, index=best_positions)
    adv_tokens.scatter_(dim=1, index=best_positions, src=src)
    adv_tokens[src_tokens == 0] = 0
    return adv_tokens
