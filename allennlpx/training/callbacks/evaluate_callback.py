from typing import Iterable

from allennlp.data import Instance
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events

from allennlpx.training.util import evaluate


class EvaluateCallback(Callback):
    def __init__(self, eval_data: Iterable[Instance]):
        self.eval_data = eval_data

    @handle_event(Events.EPOCH_END)
    def epoch_end_stuff(self, trainer) -> None:
        evaluate(trainer.model,
                 self.eval_data,
                 trainer.iterator,
                 cuda_device=next(trainer.model.parameters()).device.index,
                 batch_weight_key=None)
