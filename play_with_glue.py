from awesome_glue.config import Config
from awesome_glue.task import Task
from allennlp.common.tqdm import Tqdm
from luna.ram import ram_write
from luna.logging import log, log_config

config = Config()._parse_args()

if config.alchemist:
    Tqdm.set_slower_interval(True)

ram_write("config", config)

log_config("log", "c")
log(config)

task = Task(config)

#
if config.mode == 'train':
    task.train()
elif config.mode == 'eval':
    task.from_trained()
    task.evaluate()
elif config.mode == 'attack':
    task.from_trained()
    task.attack()