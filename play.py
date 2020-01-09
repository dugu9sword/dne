import numpy as np
import torch
from luna import (auto_create, flt2str, log, log_config, ram_read, ram_reset, ram_write)
from awesome_sst.sst_model import LstmClassifier, WORD2VECS
from awesome_sst.config import Config
from awesome_sst.task import Task

from allennlp.common.tqdm import Tqdm

config = Config()._parse_args()

if config.alchemist:
    Tqdm.set_slower_interval(True)

ram_write("config", config)

log_config("log", "cf")
log(config)

task = Task(config)

 
if config.mode == 'train':
    task.train()
elif config.mode == 'eval':
    task.from_trained()
    task.evaluate()
elif config.mode == 'attack':
    task.from_trained()
    task.attack()
