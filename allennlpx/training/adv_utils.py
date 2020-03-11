from dataclasses import dataclass
from allennlpx.interpret.attackers.cached_searcher import CachedIndexSearcher


@dataclass
class AdvTrainingPolicy:
    pass


@dataclass
class HotFlipPolicy:
    constraint: str = None
    k: int = None


@dataclass
class NeariestNeighbour:
    constraint: str = None
    k: int


class Constraint:
    def __init__(self, task_id, measure, topk, rho):
        super().__init__()
        print('Building cache of constraints for adversarial training...')

    def apply(self, src_tokens, scores):
        pass

# def apply_constraint(src_tokens, scores, ):
#     searcher = CachedIndexSearcher("nbrs.euc.top10.txt",
#                                     word2idx=vocab.get_token_index,
#                                     idx2word=vocab.get_token_from_index)
#     mask = scores.new_zeros(scores.size(), dtype=torch.bool)
#     for bid in range(src_tokens.size(0)):
#         for sid in range(src_tokens.size(1)):
#             if src_tokens[bid][sid] == 0:
#                 break
#             _, idxs = searcher.search(src_tokens[bid][sid].item())
#             for idx in idxs:
#                 mask[bid][sid] = True
#     mask = ~mask
#     scores.masked_fill(mask, -19260817)

