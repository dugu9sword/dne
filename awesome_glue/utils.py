import os
import pathlib
from collections import defaultdict

from allennlp.data import Vocabulary
from tabulate import tabulate
import faiss
import torch
import random
import numpy as np
from allennlpx.training.adv_trainer import EpochCallback
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class set_environments():
    os.environ["TORCH_HOME"] == '/disks/sdb/torch_home'


class AnnealingTemperature(EpochCallback):
    def __init__(self, anneal_num=5):
        self.anneal_num = anneal_num

    def __call__(self, trainer, metrics: Dict[str, Any], epoch: int) -> None:
        next_epoch = epoch + 1
        if hasattr(trainer.model, "word_embedders"):
            dir_embed = trainer.model.word_embedders.token_embedder_tokens
        elif hasattr(trainer.model, "bert_embedder"):
            dir_embed = trainer.model.bert_embedder.transformer_model.embeddings.word_embeddings
        if dir_embed.temperature < 0.01:
            return
        if next_epoch < self.anneal_num:
            cur_temp = np.linspace(0.01,
                                   dir_embed.temperature,
                                   num=self.anneal_num)[next_epoch]
        else:
            cur_temp = dir_embed.temperature
        dir_embed.current_temperature = cur_temp
        logger.info(
            f'Before epoch {next_epoch}, temperature is set to {cur_temp}/{dir_embed.temperature}'
        )

def get_neighbours(vec, return_edges=False):
    """
    Given an embedding matrix, find the closest 10 words in the space.
    Normally, the first word is itself, but since some words may not be
    pretrained, thus the first found maybe zero. 
    """
    index = faiss.IndexFlatL2(vec.shape[1])
    res = faiss.StandardGpuResources()  # use a single GPU
    index = faiss.index_cpu_to_gpu(res, 0, index)
    embed = vec.cpu().numpy()
    index.add(embed)
    _, I = index.search(embed, k=10)

    if return_edges:
        edges = []
        for idx in range(len(I)):
            if I[idx][0] == 0:
                edges.append([idx, idx])
                continue
            else:
                edges.extend([[idx, ele] for ele in I[idx]])
        edges = torch.tensor(edges)
        return edges
    else:
        I = torch.tensor(I)
        mask = vec.sum(1) != 0.0
        I = torch.arange(I.shape[0]).masked_fill(
            mask, 0).unsqueeze(1) + I.masked_fill(~mask.unsqueeze(1), 0)
        # test code
        for _ in range(10):
            some_idx = random.choice(range(I.shape[0]))
            assert I[some_idx][0] == some_idx
        return I, mask


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


