from awesome_glue.task_specs import TASK_SPECS
from allennlpx.data.dataset_readers.spacy_tsv import SpacyTSVReader
from allennlpx.data.dataset_readers.berty_tsv import BertyTSVReader
from allennlp.data.vocabulary import Vocabulary
from allennlpx.interpret.attackers.attacker import DEFAULT_IGNORE_TOKENS
from copy import copy
from luna import auto_create
import json
from collections import defaultdict
from functools import lru_cache


@lru_cache(maxsize=None)
def load_data(task_id: str, tokenizer: str):
    """
        Load data by the task_id and the tokenizer. 
        (BERT will not use the spacy tokenizer.)
    """
    spec = TASK_SPECS[task_id]

    reader = {
        'spacy': SpacyTSVReader,
        'bert': BertyTSVReader
    }[tokenizer](sent1_col=spec['sent1_col'],
                 sent2_col=spec['sent2_col'],
                 label_col=spec['label_col'],
                 skip_label_indexing=spec['skip_label_indexing'],
                 lower=True
                )

    def __load_data():
        train_data = reader.read(f'{spec["path"]}/train.tsv')
        dev_data = reader.read(f'{spec["path"]}/dev.tsv')
        test_data = reader.read(f'{spec["path"]}/test.tsv')
        _MIN_COUNT = 3
        vocab = Vocabulary.from_instances(
            train_data,
            min_count={
                "tokens": _MIN_COUNT,
                "sent": _MIN_COUNT,
                "sent1": _MIN_COUNT,
                "sent2": _MIN_COUNT
            },
        )
        train_data.index_with(vocab)
        dev_data.index_with(vocab)
        test_data.index_with(vocab)
        return train_data, dev_data, test_data, vocab

    # The cache name is {task}-{tokenizer}
    train_data, dev_data, test_data, vocab = auto_create(
        f"{task_id}-{tokenizer}.data", __load_data, True)

    return {
        "reader": reader,
        "data": (train_data, dev_data, test_data),
        "vocab": vocab
    }


def load_banned_words(task_id: str):
    banned_words = copy(DEFAULT_IGNORE_TOKENS)
    if 'banned_words' in TASK_SPECS[task_id]:
        banned_words.extend([
            line.rstrip('\n')
            for line in open(TASK_SPECS[task_id]['banned_words'])
        ])
    return banned_words


def load_neighbour_words(vocab: Vocabulary,
                         file_name='external_data/counter_fitted_neighbors.json'):
    nbr_dct = json.load(open(file_name))
    tokens = vocab.get_token_to_index_vocabulary()
    ret = {}
    for k in nbr_dct:
        if k in tokens:
            ret[k] = []
            for v in nbr_dct[k]:
                if v in tokens:
                    ret[k].append(v)
    ret = defaultdict(lambda: [], ret)
    return ret
