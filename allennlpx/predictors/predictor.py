import re
from contextlib import contextmanager
from typing import Callable, Iterator, List, Union

from allennlp.data import Instance
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.predictors.predictor import Predictor as Predictor_
from allennlp.common.util import JsonDict, sanitize


class Predictor(Predictor_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transform_fn: Callable[[List[Instance]], List[Instance]] = lambda x: x
        self._ensemble_num: int = 1

    def _register_embedding_gradient_hooks(self, embedding_gradients):
        """
        Registers a backward hook on the
        :class:`~allennlp.modules.text_field_embedder.basic_text_field_embbedder.BasicTextFieldEmbedder`
        class. Used to save the gradients of the embeddings for use in get_gradients()

        When there are multiple inputs (e.g., a passage and question), the hook
        will be called multiple times. We append all the embeddings gradients
        to a list.
        """
        def hook_layers(module, grad_in, grad_out):  # pylint: disable=unused-argument
            embedding_gradients.append(grad_out[0])

        backward_hooks = []
        for module in self._model.modules():
            if isinstance(module, TextFieldEmbedder):
                backward_hooks.append(module.register_backward_hook(hook_layers))

        return backward_hooks

    @contextmanager
    def capture_named_internals(self,
                                names: Union[str, List[str]],
                                match_type='full') -> Iterator[dict]:
        """
            with predictor.capture_named_internals([""]) as internals:
                outputs = predictor.predict_json(inputs)
        """
        assert match_type in ['full', 'partial', 're']
        if isinstance(names, str):
            names = [names]

        results = {}
        hooks = []

        # First we'll register hooks to add the outputs of each module to the results dict.
        def add_output(name):
            def _add_output(mod, _, outputs):
                results[name] = outputs

            return _add_output

        for idx, (mod_name, module) in enumerate(self._model.named_modules()):
            if match_type == 'full':
                if mod_name in names:
                    hook = module.register_forward_hook(add_output(mod_name))
                    hooks.append(hook)
            elif match_type == 'partial':
                for name in names:
                    if name in mod_name:
                        hook = module.register_forward_hook(add_output(mod_name))
                        hooks.append(hook)
            elif match_type == 're':
                for name in names:
                    if re.match(name, mod_name):
                        hook = module.register_forward_hook(add_output(mod_name))
                        hooks.append(hook)

        # If you capture the return value of the context manager, you get the results dict.
        yield results

        # And then when you exit the context we remove all the hooks.
        for hook in hooks:
            hook.remove()

    def predict_instance(self, instance: Instance) -> JsonDict:
        instances = [instance for i in range(self._ensemble_num)]
        instances = self._transform_fn(instances)
        outputs = self._model.forward_on_instances(instances)
        ret = {'probs': outputs[0]['probs']}
        for output in outputs[1:]:
            ret['probs'] += output['probs']
        ret['probs'] /= self._ensemble_num
        return sanitize(ret)

    def predict_batch_instance(self, instances):
        bsz = len(instances)
        b_en_instances = []  # batch x ensemble
        for instance in instances:
            b_en_instances.extend([instance for _ in range(self._ensemble_num)])
        b_en_instances = self._transform_fn(b_en_instances)
        outputs = self._model.forward_on_instances(b_en_instances)
        ret = []
        for bid in range(bsz):
            offset = bid * self._ensemble_num
            en_output = {'probs': outputs[offset]['probs']}
            for eid in range(1, self._ensemble_num):
                en_output['probs'] += outputs[offset + eid]['probs']
            en_output['probs'] /= self._ensemble_num
            ret.append(en_output)
        return sanitize(ret)

    def set_ensemble_num(self, ensemble_num):
        self._ensemble_num = ensemble_num

    def set_transform_fn(self, transform_fn):
        self._transform_fn = transform_fn
