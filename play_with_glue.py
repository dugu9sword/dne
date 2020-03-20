import os
os.environ["TORCH_HOME"] = "/disks/sdb/torch_home"

import logging
logging.getLogger().setLevel(logging.WARNING)

from awesome_glue.config import Config
from awesome_glue.task import Task
from allennlp.common.tqdm import Tqdm
from luna.ram import ram_write
from luna.logging import log, log_config
import numpy as np

import torch

config = Config()._parse_args()

torch.manual_seed(config.seed)
np.random.seed(config.seed)

if config.alchemist:
    Tqdm.set_slower_interval(True)

ram_write("config", config)

log_config("log", "c")
log(config)

task = Task(config)

{
    'train': task.train,
    'meval': task.evaluate_model,
    'peval': task.evaluate_predictor,
    'attack': task.attack,
    'transfer': task.transfer_attack,

    # BELOW COMMANDS ARE NOT STABLE
#     'knn_build': task.knn_build_index,
#     'knn_eval': task.knn_evaluate,
#     'knn_attack': task.knn_attack,
#     'build_manifold': task.build_manifold,
#     'test_distance': task.test_distance,
#     'test_ppl': task.test_ppl
}[config.mode]()
