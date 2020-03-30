import re
import torch
from contextlib import contextmanager
from copy import deepcopy
from typing import Callable, Iterator, List, Union

from allennlp.data import Instance
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.predictors.predictor import Predictor as Predictor_
from allennlp.common.util import JsonDict, sanitize, lazy_groups_of


class Predictor(Predictor_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transform_fn: Callable[[List[Instance]], List[Instance]] = lambda x: x
        self._ensemble_num: int = 1
        # Following CVPR 2019 - Defense Against Adversarial Images using Web-Scale
        # Nearest-Neighbor Search, we prefer probabilities that are more "confident",
        # thus [0.9, 0.1] has larger weight than [0.6, 0.4].
        # if p=0, it is equivilant to simply averaging all probabilities.
        self._ensemble_p = 3

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
        # Here we do not apply deepcopy for each element,
        # but an element wise deepcopy in _transform_fn is needed!
        # if id(instances[0]) == id(instances[1])
        #   d1 = deepcopy(instances), id(d1[0]) == id(d1[1])
        #   d2 = [deepcopy(e) for e in instances], id(d1[0]) != id(d1[1])
        instances = [instance for i in range(self._ensemble_num)]
        instances = self._transform_fn(instances)
        outputs = self._model.forward_on_instances(instances)
        probs_to_ensemble = [ele['probs'] for ele in outputs]
        ret = {'probs': self._weighted_average(probs_to_ensemble)}
        return sanitize(ret)

    def predict_batch_instance(self, instances):
        ret = []
        for group in lazy_groups_of(instances, int(1024 / self._ensemble_num)):
            bsz = len(group)
            b_en_instances = []  # batch x ensemble
            for instance in group:
                b_en_instances.extend([instance for _ in range(self._ensemble_num)])
            b_en_instances = self._transform_fn(b_en_instances)
            outputs = self._model.forward_on_instances(b_en_instances)
            
            for bid in range(bsz):
                offset = bid * self._ensemble_num
                probs_to_ensemble = [ele['probs'] for ele in outputs[offset: offset + self._ensemble_num]]
                en_output = {'probs': self._weighted_average(probs_to_ensemble)}
                ret.append(en_output)
        return sanitize(ret)

    def set_ensemble_num(self, ensemble_num):
        self._ensemble_num = ensemble_num

    def set_transform_fn(self, transform_fn):
        self._transform_fn = transform_fn
        
    def _weighted_average(self, probs_to_ensemble):
        # w = \sigma (max_prob - other_prob) ^ p
        # p = 0, weighted_average = mean
        # tensor([[0.2685, 0.7315],
        #         [0.3716, 0.6284],
        #         [0.4910, 0.5090],
        #         [0.8707, 0.1293],
        #         [0.4254, 0.5746]])
        # tensor([0.4854, 0.5146])
        # tensor([0.7384, 0.2616])
        if isinstance(probs_to_ensemble, list):
            probs_to_ensemble = torch.tensor(probs_to_ensemble)
        delta = probs_to_ensemble.max(dim=1, keepdims=True)[0] - probs_to_ensemble
        delta = delta ** self._ensemble_p
        delta = delta.sum(dim=1)
        delta = delta / delta.sum()
        ret = delta @ probs_to_ensemble
        # print(probs_to_ensemble)
        # print(probs_to_ensemble.mean(dim=0))
        # print(ret)
        return ret
