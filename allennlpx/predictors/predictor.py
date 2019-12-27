from typing import List, Iterator, Dict, Tuple, Any
import json
from contextlib import contextmanager
import numpy
from torch.utils.hooks import RemovableHandle
from torch import Tensor

from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.models.archival import Archive, load_archive
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.dataset import Batch

from allennlp.predictors.predictor import Predictor as Predictor_

class Predictor(Predictor_):
    # """
    # a ``Predictor`` is a thin wrapper around an AllenNLP model that handles JSON -> JSON predictions
    # that can be used for serving models through the web API or making predictions in bulk.
    # """
    # def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
    #     self._model = model
    #     self._dataset_reader = dataset_reader

    # def load_line(self, line: str) -> JsonDict:  # pylint: disable=no-self-use
    #     """
    #     If your inputs are not in JSON-lines format (e.g. you have a CSV)
    #     you can override this function to parse them correctly.
    #     """
    #     return json.loads(line)

    # def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
    #     """
    #     If you don't want your outputs in JSON-lines format
    #     you can override this function to output them differently.
    #     """
    #     return json.dumps(outputs) + "\n"

    # def predict_json(self, inputs: JsonDict) -> JsonDict:
    #     instance = self._json_to_instance(inputs)
    #     return self.predict_instance(instance)

    # def json_to_labeled_instances(self, inputs: JsonDict) -> List[Instance]:
    #     """
    #     Converts incoming json to a :class:`~allennlp.data.instance.Instance`,
    #     runs the model on the newly created instance, and adds labels to the
    #     :class:`~allennlp.data.instance.Instance`s given by the model's output.
    #     Returns
    #     -------
    #     List[instance]
    #     A list of :class:`~allennlp.data.instance.Instance`
    #     """
    #     instance = self._json_to_instance(inputs)
    #     outputs = self._model.forward_on_instance(instance)
    #     new_instances = self.predictions_to_labeled_instances(instance, outputs)
    #     return new_instances

    def get_gradients(self,
                      instances: List[Instance]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Gets the gradients of the loss with respect to the model inputs.

        Parameters
        ----------
        instances: List[Instance]

        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
        The first item is a Dict of gradient entries for each input.
        The keys have the form  ``{grad_input_1: ..., grad_input_2: ... }``
        up to the number of inputs given. The second item is the model's output.

        Notes
        -----
        Takes a ``JsonDict`` representing the inputs of the model and converts
        them to :class:`~allennlp.data.instance.Instance`s, sends these through
        the model :func:`forward` function after registering hooks on the embedding
        layer of the model. Calls :func:`backward` on the loss and then removes the
        hooks.
        """
        embedding_gradients: List[Tensor] = []
        hooks: List[RemovableHandle] = self._register_embedding_gradient_hooks(embedding_gradients)

        dataset = Batch(instances)
        dataset.index_instances(self._model.vocab)
        outputs = self._model.decode(self._model.forward(**dataset.as_tensor_dict()))

        loss = outputs['loss']
        self._model.zero_grad()
        loss.backward()

        for hook in hooks:
            hook.remove()

        grad_dict = dict()
        for idx, grad in enumerate(embedding_gradients):
            key = 'grad_input_' + str(idx + 1)
            grad_dict[key] = grad.squeeze_(0)

        return grad_dict, outputs

    def _register_embedding_gradient_hooks(self, embedding_gradients):
        """
        Registers a backward hook on the
        :class:`~allennlp.modules.text_field_embedder.basic_text_field_embbedder.BasicTextFieldEmbedder`
        class. Used to save the gradients of the embeddings for use in get_gradients()

        When there are multiple inputs (e.g., a passage and question), the hook
        will be called multiple times. We append all the embeddings gradients
        to a list.
        """
        def hook_layers(module, grad_in, grad_out): # pylint: disable=unused-argument
            embedding_gradients.append(grad_out[0])

        backward_hooks = []
        for module in self._model.modules():
            if isinstance(module, TextFieldEmbedder):
                backward_hooks.append(module.register_backward_hook(hook_layers))

        return backward_hooks

    # @contextmanager
    # def capture_model_internals(self) -> Iterator[dict]:
    #     """
    #     Context manager that captures the internal-module outputs of
    #     this predictor's model. The idea is that you could use it as follows:

    #     .. code-block:: python

    #         with predictor.capture_model_internals() as internals:
    #             outputs = predictor.predict_json(inputs)

    #         return {**outputs, "model_internals": internals}
    #     """
    #     results = {}
    #     hooks = []

    #     # First we'll register hooks to add the outputs of each module to the results dict.
    #     def add_output(idx: int):
    #         def _add_output(mod, _, outputs):
    #             results[idx] = {"name": str(mod), "output": sanitize(outputs)}
    #         return _add_output

    #     for idx, module in enumerate(self._model.modules()):
    #         if module != self._model:
    #             hook = module.register_forward_hook(add_output(idx))
    #             hooks.append(hook)

    #     # If you capture the return value of the context manager, you get the results dict.
    #     yield results

    #     # And then when you exit the context we remove all the hooks.
    #     for hook in hooks:
    #         hook.remove()

    # def predict_instance(self, instance: Instance) -> JsonDict:
    #     outputs = self._model.forward_on_instance(instance)
    #     return sanitize(outputs)
