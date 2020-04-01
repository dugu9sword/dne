from functools import lru_cache
from typing import Union, Callable, Dict

import numpy as np
import torch
from tabulate import tabulate

from luna import cast_list, show_mean_std
from .searcher import Searcher


class EmbeddingSearcher(Searcher):
    """
        When loading a pretrained embedding matrix, some words may not
        occur in the pretrained vectors, thus they may be filled with random
        numbers. This may cause strange results when querying neariest 
        neighbours. We assume that all words that are not found in the 
        pretrained vectors will be filled with 0. Querying these words will
        return an empty list.
    """
    def __init__(
        self,
        embed: torch.Tensor,
        word2idx: Union[Callable, Dict],
        idx2word: Union[Callable, Dict],
    ):
        self.embed = embed
        if isinstance(word2idx, dict):
            self.word2idx = word2idx.__getitem__
        else:
            self.word2idx = word2idx
        if isinstance(idx2word, dict):
            self.idx2word = idx2word.__getitem__
        else:
            self.idx2word = idx2word
        self._cache = {}

    def search(self, *args, **kwargs):
        return self.find_neighbours(*args, **kwargs)
    
    def is_pretrained(self, element: Union[int, str]):
        return not all(self.as_vector(element) == 0.0)

    def as_vector(self, element: Union[int, str, torch.Tensor]):
        if isinstance(element, int):
            idx = element
            query_vector = self.embed[idx]
        elif isinstance(element, str):
            idx = self.word2idx(element)
            query_vector = self.embed[idx]
        elif isinstance(element, torch.Tensor):
            query_vector = element
        else:
            raise TypeError('You passed a {}, int/str/torch.Tensor required'.format(type(element)))
        return query_vector

    def as_index(self, element: Union[int, str]):
        if isinstance(element, int):
            idx = element
        elif isinstance(element, str):
            idx = self.word2idx(element)
        else:
            raise TypeError('You passed a {}, int/str required'.format(type(element)))
        return idx

    # search neighbours of all words and save them into a cache,
    # this will speed up the query process. 
    # The pre_search is rather fast, feel free to use it.
    def pre_search(self, measure='euc', topk=None, rho=None, gpu=True):
        import faiss
        data = self.embed.cpu().numpy()
        dim = self.embed.size(1)
        index = faiss.IndexFlatL2(dim)
        index.add(data)
        if gpu:
            res = faiss.StandardGpuResources()  # use a single GPU
            index = faiss.index_cpu_to_gpu(res, 0, index)
        D, I = index.search(data, topk)
        self._cache[f'D-{measure}-{topk}-{rho}'] = D
        self._cache[f'I-{measure}-{topk}-{rho}'] = I

    @lru_cache(maxsize=None)
    @torch.no_grad()
    def find_neighbours(self,
                        element: Union[int, str, torch.Tensor],
                        measure='euc',
                        topk=None,
                        rho=None,
                        return_words=False, # by default, return (D, I)
                        verbose=False):
        # checking args
        assert measure in ['euc', 'cos']
        if rho is not None:
            assert (measure == 'euc' and rho > 0) or (
                measure == 'cos' and 0 < rho < 1
            ), "threshold for euc distance must be larger than 0, for cos distance must be between 0 and 1"

        measure_fn = cos_dist if measure == 'cos' else euc_dist
        query_vector = self.as_vector(element)

        # Assume that a vector equals to 0 has no neighbours
        if not self.is_pretrained(query_vector):
            if return_words:
                return []
            else:
                return None, None

        if f'D-{measure}-{topk}-{rho}' in self._cache:
            _idx = self.as_index(element)
            D = self._cache[f'D-{measure}-{topk}-{rho}'][_idx]
            I = self._cache[f'I-{measure}-{topk}-{rho}'][_idx]
            tk_vals = torch.tensor(D, device=self.embed.device)
            tk_idxs = torch.tensor(I, device=self.embed.device)
            if return_words:
                return [self.idx2word(ele) for ele in cast_list(tk_idxs)]
            else:
                return tk_vals, tk_idxs

        if topk is None:
            _topk = self.embed.size(0)
        else:
            _topk = topk
            
        dists = measure_fn(query_vector, self.embed)
        tk_vals, tk_idxs = torch.topk(dists, _topk, largest=False)
        
        if rho is not None:
            mask_idx = tk_vals < rho
            tk_vals = tk_vals[mask_idx]
            tk_idxs = tk_idxs[mask_idx]

        if verbose:
            table = []
            print('Neariest neighbours measured by {}, topk={}, rho={}'.format(measure, topk, rho))
            for i in tk_idxs:
                i = i.item()
                table.append([i, self.idx2word(i), dists[i].item()])
            print(tabulate(table))
        if return_words:
            return [self.idx2word(ele) for ele in cast_list(tk_idxs)]
        else:
            return tk_vals, tk_idxs
        
    def show_embedding_info(self, measure='euc'):
        print('*** Statistics of parameters and 2-norm ***')
        show_mean_std(self.embed, 'Param')
        show_mean_std(torch.norm(self.embed, p=2, dim=1), 'Norm')

        print('*** Statistics of distances in a N-nearest neighbourhood ***')
        nbr_num = [5, 10, 20, 50, 100, 200, 500, 10000, 20000]
        dists = {nbr: [] for nbr in nbr_num}
        for ele in cast_list(torch.randint(self.embed.size(0), (50, ))):
            if self.embed[ele].sum() == 0.:
                continue
            vals, idxs = self.find_neighbours(ele, measure, None)
            for nbr in nbr_num:
                dists[nbr].append(vals[1:nbr + 1])
        table = []
        for nbr in nbr_num:
            dists[nbr] = torch.cat(dists[nbr])
            table.append([nbr, dists[nbr].mean().item(), dists[nbr].std().item()])
        print(tabulate(table, headers=['N', 'mean', 'std'], floatfmt='.2f'))
        # exit()



# torch.tensor([100]).expand(2000, 100):
#   return a view, in memory it's still [100]
# torch.tensor([100]).expand(2000, 1):
#   return a copied tensor
def cos_dist(qry, mem):
    return 1 - cos_sim(qry, mem)


def cos_sim(qry, mem):
    return torch.nn.functional.cosine_similarity(mem, qry.expand(mem.size(0), mem.size(1)), dim=1)


def euc_dist(qry, mem):
    return torch.sqrt((qry - mem).pow(2).sum(dim=1))


def compare_idxes(nbr1, nbr2):
    nbr1 = set(cast_list(nbr1))
    nbr2 = set(cast_list(nbr2))
    inter = nbr1.intersection(nbr2)
    return len(inter)


"""
    DEPRECATED CODE.
    
    def use_faiss_backend(
        self,
        gpu=False,
        ann=False,
        ann_center=10,
        ann_nprob=1,
    ):
        # The method is just a running example of the faiss library which supports
        #   - batch queries
        #   - approximate nearest neighbours
        #   - several distances
        # However, for most of the time, the vanilla pytorch version is enough.
        #
        # Some details:
        # 1. GPU is not always better than CPU, see:
        #       https://github.com/facebookresearch/faiss/wiki/Comparing-GPU-vs-CPU
        #    From my preliminary experiment (vocab size 20000+, query size 1):
        #       when ann=True, you may set gpu=False
        #       when ann=False, you may set gpu=True
        # 2. There are some overheads in the code. The faiss lib is based on numpy,
        #    while in this code there are some conversions between:
        #       [tensor(GPU) <->] tensor <-> numpy [<-> GPU]
        #
        import faiss
        data = self.embed.cpu().numpy()
        dim = self.embed.size(1)
        index = faiss.IndexFlatL2(dim)
        if ann:
            fast_index = faiss.IndexIVFFlat(index, dim, ann_center)
            fast_index.train(data)
            fast_index.nprobe = ann_nprob
            index = fast_index
        index.add(data)
        if gpu:
            res = faiss.StandardGpuResources()  # use a single GPU
            index = faiss.index_cpu_to_gpu(res, 0, index)
        self.faiss_index = index
        
    def search():
        if self.faiss_index is None:
            dists = measure_fn(query_vector, self.embed)
            tk_vals, tk_idxs = torch.topk(dists, _topk, largest=False)
        else:
            if measure == 'cos':
                raise Exception("cos still not compatible with faiss (since I am lazy)")
            D, I = self.faiss_index.search(query_vector.unsqueeze(0).cpu().numpy(), _topk)
            tk_vals = torch.tensor(D[0], device=self.embed.device)
            tk_idxs = torch.tensor(I[0], device=self.embed.device)
"""