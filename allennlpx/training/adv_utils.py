from dataclasses import dataclass
from allennlpx.interpret.attackers.searchers import EmbeddingSearcher
import torch
from luna import batch_pad
import random
from typing import Union


@dataclass
class AdvTrainingPolicy:
    adv_iteration: int = 1
    adv_field: str = 'sent'


@dataclass
class NoPolicy(AdvTrainingPolicy):
    adv_iteration: int = 0


@dataclass
class HotFlipPolicy(AdvTrainingPolicy):
    """
        Use hotflip to change some words to maximize the loss.
        Since there may exist two sentences (sent_a, sent_b) forwarding through the
        embedding one-by-one, the forward hooks will catch (output_a, output_b),
        while the backward hooks will catch (grad_b, grad_a). We only support
        change one sentence during adversarial training, so you should specify the
        index of the sentence you want to change, if sent_b, set `forward_order` to
        1; if sent_a, set it to 0. 
    """
    forward_order: int = 0
    # searcher: CachedIndexSearcher = None
    searcher: EmbeddingSearcher = None
    replace_num: Union[int, float] = None


@dataclass
class RandomNeighbourPolicy(AdvTrainingPolicy):
    """
        Randomly change some words during training.
    """
    searcher: EmbeddingSearcher = None
    replace_num: Union[int, float] = None


def apply_constraint_(searcher, src_tokens, scores):
    mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    src_tokens_lst = src_tokens.tolist()

    # idxes_to_mask stores all allowed word indicies, padded with 0s.
    idxes_to_mask = []
    for bid in range(src_tokens.size(0)):
        for sid in range(src_tokens.size(1)):
            if src_tokens_lst[bid][sid] == 0:
                idxes_to_mask.append([])
                continue
            _, idxs = searcher.find_neighbours(src_tokens_lst[bid][sid], 'euc', 10, None)
            if idxs is None:
                idxes_to_mask.append([src_tokens_lst[bid][sid]])
            else:
                idxes_to_mask.append(idxs.cpu().numpy().tolist())
    idxes_to_mask = src_tokens.new_tensor(batch_pad(idxes_to_mask, 0))
    idxes_to_mask = idxes_to_mask.view(*src_tokens.size(), -1)

    # mask is a bool tensor that stores all *allowed* word indicies
    # but 0th word(<pad>) is also True, so we set 0th value to False
    mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    mask.scatter_(dim=2,
                  index=idxes_to_mask,
                  src=idxes_to_mask.new_ones(idxes_to_mask.size(),
                                             dtype=torch.bool))
    mask[:, :, 0] = False

    # fill all the unallowed values to -inf
    mask = ~mask
    scores.masked_fill_(mask, -19260817.)


def get_replace_num(replace_num, length):
    if replace_num > 1:
        return int(min(replace_num, length))
    else:
        return int(replace_num * length)


def hotflip(*,
            src_tokens,
            embeds,
            grads,
            embedding_matrix,
            replace_num,
            searcher=None):
    replace_num = get_replace_num(replace_num, src_tokens.size(1))

    # compute the direction vector dot the gradient
    prev_embed_dot_grad = torch.einsum("bij,bij->bi", grads, embeds)
    new_embed_dot_grad = torch.einsum("bij,kj->bik", grads, embedding_matrix)
    dir_dot_grad = new_embed_dot_grad - prev_embed_dot_grad.unsqueeze(-1)

    # maybe some constraints
    if searcher is not None:
        apply_constraint_(searcher, src_tokens, dir_dot_grad)

    # supposing that vocab[0]=<pad>, vocab[1]=<unk>.
    # we set value of <pad> to be smaller than the <unk>.
    # if none of words in the vocab are selected, (all their values are -19260817)
    # the "argmax" will select <unk> instead of other words.
    dir_dot_grad[:, :, 0] = -19260818
    dir_dot_grad[:, :, 1] = -19260816

    # at each step, we select the best substitute(best_at_each_step)
    # and get the score(score_at_each_step), then select the best positions
    # to replace.
    score_at_each_step, best_at_each_step = dir_dot_grad.max(2)
    _, best_positions = score_at_each_step.topk(replace_num)

    # use the selected token index to replace the original one
    adv_tokens = src_tokens.clone()
    src = best_at_each_step.gather(dim=1, index=best_positions)
    adv_tokens.scatter_(dim=1, index=best_positions, src=src)
    adv_tokens[src_tokens == 0] = 0
    return adv_tokens


def random_swap(*, src_tokens, replace_num, searcher):
    adv_tokens_lst = src_tokens.tolist()
    for bid in range(src_tokens.size(0)):
        sids = [
            sid for sid in range(src_tokens.size(1))
            if adv_tokens_lst[bid][sid] != 0
        ]
        sids = random.sample(sids, k=get_replace_num(replace_num, len(sids)))
        for sid in sids:
            _, idxs = searcher.search(adv_tokens_lst[bid][sid], 'euc', 10,
                                      None)
            if idxs is not None:
                adv_tokens_lst[bid][sid] = random.choice(
                    idxs.cpu().numpy().tolist())
    return torch.tensor(adv_tokens_lst, device=src_tokens.device)


def guess_token_key_from_field(batch_fields):
    # Given an indexed field, we suppose that the key of the token indexer
    # is "tokens", but we still need to guess the key of the indexed tokens.
    # The structure of an indexed field is like:
    #    { indexer_name_1: {key_1: tensor, key_2: tensor},
    #      indexer_name_2: {key: tensor}, ... }
    # For a vanilla word-based model, the field is like:
    #    { "tokens": {"tokens": tensor} }
    # For a bert-based model, the field is like:
    #    { "tokens": {"token_ids": tensor, "offsets": tensor, "mask": tensor}}
    if "tokens" not in batch_fields:
        raise Exception("The name of the token indexer is not `tokens`")
    if 'tokens' in batch_fields['tokens']: 
        return 'tokens'
    elif 'token_ids' in batch_fields['tokens']:
        return 'token_ids'
    else:
        raise Exception('Something wrong, boy.')
