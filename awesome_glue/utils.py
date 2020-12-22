from allennlpx import allenutil
import os

import torch
from allennlpx.training.adv_trainer import EpochCallback
from typing import Dict, Any
import logging
import pandas
from luna import batch_pad, ram_set_flag, ram_reset_flag


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
        return "Accu before: {:5.2f}%, after: {:5.2f}%, Flip ratio {:5.2f}%".format(
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

class WarmupCallback(EpochCallback):
    def __init__(self, warm_up_epochs):
        self.warm_up_epochs = warm_up_epochs

    def __call__(
        self, trainer, metrics: Dict[str, Any], epoch: int
    ) -> None:
        if epoch < self.warm_up_epochs:
            logger.warning(f'At epoch {epoch}, set warm_mode to True')
            ram_set_flag("warm_mode")
        else:
            logger.warning(f'At epoch {epoch}, set warm_mode to False')
            ram_reset_flag("warm_mode")


def allen_instances_for_attack(instances):
    ret = []
    for inst in instances:
        ret.append((allenutil.as_sentence(inst), inst['label'].label))
    return ret