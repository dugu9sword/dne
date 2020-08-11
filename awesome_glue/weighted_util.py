import os
from overrides import overrides
from allennlp.data import Vocabulary
import random
import numpy as np
from typing import Dict, Any
from luna import batch_pad
from collections import Counter
from functools import lru_cache
from allennlpx.interpret.attackers.searchers import CachedWordSearcher
import torch


class WeightedHull:
    """
        Given tokens (token_num, ), return:
            nbr_tokens (token_num, max_nbr_num)
            coeff (token_num, max_nbr_num) --> Sometimes not required
    """
    def get_nbr_and_coeff(self, tokens, require_coeff=True):
        raise NotImplementedError


def build_neighbour_matrix(searcher, vocab):
    t2i = vocab.get_token_to_index_vocabulary("tokens")
    vocab_size = vocab.get_vocab_size("tokens")
    nbr_matrix = []
    for idx in range(vocab_size):
        token = vocab.get_token_from_index(idx)
        nbrs = [idx]
        for nbr in searcher.search(token):
            assert nbr in t2i
            nbrs.append(t2i[nbr])
        nbr_matrix.append(nbrs)
    nbr_lens = list(map(len, nbr_matrix))
    nbr_matrix = batch_pad(nbr_matrix)
    return nbr_matrix, nbr_lens


class SameAlphaHull(WeightedHull):
    def __init__(self, alpha, nbrs):
        self.alpha = alpha
        self.nbrs = nbrs
    
    @classmethod
    def build(cls, alpha, nbr_file, vocab, nbr_num, second_order):
        t2i = vocab.get_token_to_index_vocabulary("tokens")
        searcher = CachedWordSearcher(
            nbr_file,
            vocab_list=t2i,
            second_order=second_order
        )
        nbrs, _ = build_neighbour_matrix(searcher, vocab)
        nbrs = torch.tensor(nbrs)[:, :nbr_num].cuda()
        return cls(alpha, nbrs)

    @overrides
    def get_nbr_and_coeff(self, tokens, require_coeff=True):
        nbr_tokens = self.nbrs[tokens]
        nbr_num_lst = (nbr_tokens != 0).sum(dim=1).tolist()
        max_nbr_num = max(nbr_num_lst)
        nbr_tokens = nbr_tokens[:, :max_nbr_num]
        if require_coeff:
            coeffs = dirichlet_sampling_fast(
                nbr_num_lst, self.alpha, self.nbrs.shape[1]
            )
            torch.cuda.empty_cache()
            coeffs = torch.Tensor(coeffs)[:, :max_nbr_num].to(tokens.device)
        else:
            coeffs = None
        return nbr_tokens, coeffs


class DecayAlphaHull(WeightedHull):
    def __init__(self, alpha, decay, nbrs, first_order_lens):
        self.alpha = alpha
        self.decay = decay
        self.nbrs = nbrs
        self.first_order_lens = first_order_lens

    @overrides
    def get_nbr_and_coeff(self, tokens, require_coeff=True):
        nbr_tokens = self.nbrs[tokens]
        first_order_lens_lst = self.first_order_lens[tokens].tolist()
        nbr_num_lst = (nbr_tokens != 0).sum(dim=1).tolist()
        max_nbr_num = max(nbr_num_lst)
        nbr_tokens = nbr_tokens[:, :max_nbr_num]
        if require_coeff:
            coeffs = dirichlet_sampling_fast_2nd(
                nbr_num_lst, first_order_lens_lst, self.alpha, self.decay, self.nbrs.shape[1]
            )
            coeffs = torch.Tensor(coeffs)[:, :max_nbr_num].to(tokens.device)
        else:
            coeffs = None
        return nbr_tokens, coeffs

    @classmethod
    def build(cls, alpha, decay, nbr_file, vocab, nbr_num, second_order):
        assert second_order
        t2i = vocab.get_token_to_index_vocabulary("tokens")
        first_order_searcher = CachedWordSearcher(
            nbr_file, vocab_list=t2i, second_order=False
        )
        _, first_order_lens = build_neighbour_matrix(
            first_order_searcher, vocab
        )
        second_order_searcher = CachedWordSearcher(
            nbr_file, vocab_list=t2i, second_order=True
        )
        nbrs, second_order_lens = build_neighbour_matrix(
            second_order_searcher, vocab
        )
        nbrs = torch.tensor(nbrs)[:, :nbr_num].cuda()
        first_order_lens = torch.tensor(first_order_lens).cuda()
        return cls(alpha, decay, nbrs, first_order_lens)


_cache_dirichlet_size = 10000


@lru_cache(maxsize=None)
def _cache_dirichlet(alpha, vertex_num, max_vertex_num, v0_num=None, decay=None):
    if vertex_num == 0:
        return None
    # import pdb; pdb.set_trace()
    if alpha > 0.0:
        if v0_num is None:
            alphas = [alpha] * vertex_num
        else:
            alphas = [alpha] * v0_num + [alpha * decay] * (vertex_num - v0_num)
        diri = np.random.dirichlet(alphas, _cache_dirichlet_size).astype(np.float32)
    else:
        # if alpha == 0, generate a random one-hot matrix
        diri = np.eye(vertex_num)[np.random.choice(vertex_num, _cache_dirichlet_size)].astype(np.float32)
    zero = np.zeros((_cache_dirichlet_size, max_vertex_num - vertex_num),
                    dtype=np.float32)
    ret = np.concatenate((diri, zero), axis=1).tolist()
    return ret


_cache_probs_2nd = {}
_cache_offsets_2nd = {}


def dirichlet_sampling_fast_2nd(vertex_nums, v0_nums, alpha, decay, max_vertex_num):
    ret = []
    default_prob = [1.0] + [0.0] * (max_vertex_num - 1)
    for i, (s, s1) in enumerate(zip(vertex_nums, v0_nums)):
        if s == 0:
            ret.append(default_prob)
        else:
            hash_idx = s << 5 + s1
            if hash_idx not in _cache_probs_2nd:
                _cache_probs_2nd[hash_idx] = _cache_dirichlet(
                    alpha, s, max_vertex_num, s1, decay
                )
                _cache_offsets_2nd[hash_idx] = random.randint(0, _cache_dirichlet_size)  
            _cache_offsets_2nd[hash_idx] = (_cache_offsets_2nd[hash_idx] + i) % _cache_dirichlet_size
            ret.append(_cache_probs_2nd[hash_idx][_cache_offsets_2nd[hash_idx]])
    return ret


def dirichlet_sampling_fast(vertex_nums, alpha, max_vertex_num):
    ret = []
    default_prob = [1.0] + [0.0] * (max_vertex_num - 1)
    cache_probs = []
    cache_offsets = []
    for v_num in range(max_vertex_num + 1):
        cache_probs.append(_cache_dirichlet(alpha, v_num, max_vertex_num))
        cache_offsets.append(random.randint(0, _cache_dirichlet_size))
    for i, n in enumerate(vertex_nums):
        if n == 0:
            ret.append(default_prob)
        else:
            cache_offsets[n] = (cache_offsets[n] + i) % _cache_dirichlet_size
            ret.append(cache_probs[n][cache_offsets[n]])
    return ret
