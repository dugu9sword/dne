import os

from allennlp.data import Vocabulary
from tabulate import tabulate
import torch
import numpy as np
from allennlpx.training.adv_trainer import EpochCallback, BatchCallback
from typing import Dict, Any
import logging
import pandas
from luna import batch_pad
from collections import Counter
from functools import lru_cache

logger = logging.getLogger(__name__)


def set_environments():
    os.environ["TORCH_HOME"] = '/disks/sdb/torch_home'


def read_hyper(task_id, arch, key):
    df = pandas.read_csv("hyper_params.csv", delimiter=',\s+', engine='python')
    df.columns = df.columns.str.strip()
    for t in [task_id, '*']:
        for a in [arch, '*']:
            x = df.query(f"task=='{t}' and arch=='{a}'")
            if x.shape[0] == 1:
                return x[key].values[0]
    raise Exception()


def get_neighbour_matrix(vocab, searcher):
    vocab_size = vocab.get_vocab_size("tokens")
    nbr_matrix = []
    t2i = vocab.get_token_to_index_vocabulary("tokens")
    for idx in range(vocab_size):
        token = vocab.get_token_from_index(idx)
        nbrs = [idx]
        for nbr in searcher.search(token):
            if nbr in t2i:
                nbrs.append(t2i[nbr])
        nbr_matrix.append(nbrs)
    nbr_matrix = batch_pad(nbr_matrix)
    return torch.tensor(nbr_matrix)


class FreqUtil:
    @staticmethod
    def print_statistics(vocab: Vocabulary):
        tokens_with_counts = list(vocab._retained_counter['tokens'].items())
        tokens_with_counts.sort(key=lambda x: x[1], reverse=True)
        num = len(tokens_with_counts)

        tables = []
        for i in range(1, 11):
            before = min(num // 10 * i, num - 1)
            tables.append([before, tokens_with_counts[before][1]])
        print(tabulate(tables, headers=['top-k', 'min frequency']))

    @staticmethod
    def get_group_by_frequency(vocab: Vocabulary, group_id, num_groups):
        tokens_with_counts = list(vocab._retained_counter['tokens'].items())
        tokens_with_counts.sort(key=lambda x: x[1], reverse=True)
        num = len(tokens_with_counts)

        start_idx = int(num / num_groups * group_id)
        end_idx = min(int(num / num_groups * (group_id + 1)), num)
        selected = tokens_with_counts[start_idx:end_idx]
        return list(map(lambda x: x[0], selected))

    @staticmethod
    def topk_frequency(vocab: Vocabulary,
                       number,
                       order='most',
                       exclude_words=[]):
        assert order in ['most', 'least']
        tokens_with_counts = list(vocab._retained_counter['tokens'].items())
        tokens_with_counts.sort(key=lambda x: x[1], reverse=order == 'most')
        ret = []
        for tok, cou in tokens_with_counts:
            if tok not in exclude_words:
                ret.append(tok)
            if len(ret) == number:
                break
        return ret


class AttackMetric:
    def __init__(self):
        super().__init__()
        self._succ_num = 0
        self._fail_num = 0
        self._escape_num = 0

    def succeed(self):
        self._succ_num += 1

    def fail(self):
        self._fail_num += 1

    def escape(self):
        self._escape_num += 1

    def count_label(self, gold_label, raw_label, att_label):
        if raw_label == gold_label:
            if att_label == raw_label:
                self.fail()
            else:
                self.succeed()
        else:
            self.escape()

    @property
    def accuracy_before_attack(self):
        return (self._succ_num + self._fail_num) / (
            self._succ_num + self._fail_num + self._escape_num) * 100

    @property
    def accuracy_after_attack(self):
        return self._fail_num / (self._succ_num + self._fail_num +
                                 self._escape_num) * 100

    @property
    def flip_ratio(self):
        return self._succ_num / (self._succ_num + self._fail_num + 1e-40) * 100

    def __repr__(self):
        return "Accu before: {:.2f}%, after: {:.2f}%, Flip ratio {:.2f}%".format(
            self.accuracy_before_attack, self.accuracy_after_attack,
            self.flip_ratio)


def text_diff(a_text, b_text):
    if isinstance(a_text, list):
        a_lst, b_lst = a_text, b_text
    else:
        a_lst = a_text.split(" ")
        b_lst = b_text.split(" ")
    assert len(a_lst) == len(b_lst), (a_lst, b_lst)
    a_changes = []
    b_changes = []
    for a_word, b_word in zip(a_lst, b_lst):
        if a_word != b_word:
            a_changes.append(a_word)
            b_changes.append(b_word)
    return {
        "a_changes": a_changes,
        "b_changes": b_changes,
        "change_num": len(a_changes),
        "change_ratio": len(a_changes) / len(a_lst)
    }


# class DirichletAnnealing(BatchCallback):
#     def __init__(self, anneal_epoch_num=3, batch_per_epoch=None):
#         self.anneal_epoch_num = anneal_epoch_num
#         self.batch_per_epoch = batch_per_epoch
#         self.total_steps = anneal_epoch_num * batch_per_epoch
#         self.start_alpha = 10.0

#     def __call__(self, trainer, epoch: int, batch_number: int,
#                  is_training: bool):
#         if hasattr(trainer.model, "word_embedders"):
#             dir_embed = trainer.model.word_embedders.token_embedder_tokens
#         elif hasattr(trainer.model, "bert_embedder"):
#             dir_embed = trainer.model.bert_embedder.transformer_model.embeddings.word_embeddings
#         if dir_embed.alpha > self.start_alpha:
#             return
#         cur_step = epoch * self.batch_per_epoch + batch_number
#         if epoch >= self.anneal_epoch_num:
#             cur_alpha = dir_embed.alpha
#         else:
#             cur_ratio = cur_step / self.total_steps
#             cur_alpha = cur_ratio * dir_embed.alpha + (
#                 1 - cur_ratio) * self.start_alpha
#         dir_embed.current_alpha = cur_alpha
#         if batch_number % (self.batch_per_epoch // 4) == 0:
#             logger.info(
#                 f'At epoch-{epoch} batch-{batch_number},' +
#                 f'alpha is set to {cur_alpha}({dir_embed.alpha})'
#             )

# def get_neighbours(vec, return_edges=False):
#     """
#     Given an embedding matrix, find the 10 closest words in the space.
#     Normally, the first word is itself, but since some words may not be
#     pretrained, thus the first found maybe zero.
#     """
#     index = faiss.IndexFlatL2(vec.shape[1])
#     res = faiss.StandardGpuResources()  # use a single GPU
#     index = faiss.index_cpu_to_gpu(res, 0, index)
#     embed = vec.cpu().numpy()
#     index.add(embed)
#     _, I = index.search(embed, k=10)

#     if return_edges:
#         edges = []
#         for idx in range(len(I)):
#             if I[idx][0] == 0:
#                 edges.append([idx, idx])
#                 continue
#             else:
#                 edges.extend([[idx, ele] for ele in I[idx]])
#         edges = torch.tensor(edges)
#         return edges
#     else:
#         I = torch.tensor(I)
#         mask = vec.sum(1) != 0.0
#         I = torch.arange(I.shape[0]).masked_fill(
#             mask, 0).unsqueeze(1) + I.masked_fill(~mask.unsqueeze(1), 0)
#         # test code
#         for _ in range(10):
#             some_idx = random.choice(range(I.shape[0]))
#             assert I[some_idx][0] == some_idx
#         return I, mask
