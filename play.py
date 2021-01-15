import os
os.environ["TORCH_HOME"] = "/disks/sdb/torch_home"

from awesome_glue.config import Config
config = Config()._parse_args()
if not config.alchemist:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda)

import logging
from awesome_glue.task import Task
from allennlp.common import logging as common_logging
from luna.ram import ram_write
from luna.logging import log, log_config
from luna import set_seed
import sys

logging.getLogger().setLevel(logging.WARNING)
log_config("log", "c")
log(config)
sys.stdout.flush()
set_seed(config.seed)
if config.alchemist:
    common_logging.FILE_FRIENDLY_LOGGING = True

ram_write("config", config)
task = Task(config)

{
    'train': task.train,
    'meval': task.evaluate_model,
    'peval': task.evaluate_predictor,
    'attack': task.attack,
    # 'attack_pro': task.attack_pro
}[config.mode]()
