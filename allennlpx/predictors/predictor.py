import re
import torch
from contextlib import contextmanager
from copy import deepcopy
from typing import Callable, Iterator, List, Union

from allennlp.data import Instance
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.predictors.predictor import Predictor as Predictor_
from allennlp.common.util import JsonDict, sanitize, lazy_groups_of
from allennlpx import allenutil
import logging
from functools import lru_cache
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


class Predictor(Predictor_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_tokens: int = None
        self._grad_enabled: bool = False
        self._ensemble_num: int = 1
        self._transform_fn: Callable[[List[str]], List[str]] = None
        self._transform_field: str = 'sent'
        
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

    def predict_batch_json(self, json_dicts: List[JsonDict], fast=False):
        sent_size = len(json_dicts[-1][self._transform_field].split(" "))
        if self._max_tokens:
            max_batch_forward = self._max_tokens // sent_size
        else:
            max_batch_forward = guess_max_batch(sent_size, guess_bert(self._model))

        ret = []
        for group in lazy_groups_of(json_dicts, int(max_batch_forward / self._ensemble_num)):
            # print('SPLIT')
            torch.cuda.empty_cache()
            bsz = len(group)
            b_en_jsons = []  # batch x ensemble
            for json_dict in group:
                # Assuming that the values of the dict are all strings, so copy = deepcopy
                b_en_jsons.extend([json_dict.copy() for _ in range(self._ensemble_num)])
            if self._transform_fn:
                tf_in = list(map(lambda x: x[self._transform_field], b_en_jsons))
                tf_out = self._transform_fn(tf_in)
                for i in range(bsz):
                    b_en_jsons[i][self._transform_field] = tf_out[i]
            
            if not fast:
                b_en_insts = self._batch_json_to_instances(b_en_jsons)
                outputs = self._model.forward_on_instances(b_en_insts)
                for bid in range(bsz):
                    offset = bid * self._ensemble_num
                    probs_to_ensemble = [ele['probs'] for ele in outputs[offset: offset + self._ensemble_num]]
                    en_output = {'probs': self._weighted_average(probs_to_ensemble)}
                    ret.append(en_output)
            else:
                t2i = self._model.vocab.get_token_to_index_vocabulary()
                t2i = defaultdict(lambda: t2i['@@UNKNOWN@@'], t2i)
                sents = []
                for json_dict in b_en_jsons:
                    sents.append([t2i[x] for x in json_dict['sent'].split(' ')])
                outputs = self._model({"tokens": {"tokens": torch.tensor(sents).cuda()}})
                for bid in range(bsz):
                    offset = bid * self._ensemble_num
                    probs_to_ensemble = outputs['probs'][offset: offset + self._ensemble_num]
                    en_output = {'probs': self._weighted_average(probs_to_ensemble)}
                    ret.append(en_output)
        return sanitize(ret)

    def predict_json(self, json_dict: JsonDict):
        return self.predict_batch_json([json_dict])[0]

    def predict_batch_instance(self, instances: List[Instance]):
        if self._ensemble_num == 1 and self._transform_fn is None:
            # In a normal mode, we split the instances and feed into the model
            sent_size = len(instances[-1][self._transform_field].tokens)
            if self._max_tokens:
                max_batch_forward = self._max_tokens // sent_size
            else:
                max_batch_forward = guess_max_batch(sent_size, guess_bert(self._model))   
            results = []
            for group in lazy_groups_of(instances, max_batch_forward):
                results.extend(self._model.forward_on_instances(group))
            results = sanitize(results)
        else:
            # otherwise, we transform them into jsons 
            json_dicts = list(map(self._dataset_reader.instance_to_text, instances))
            results = self.predict_batch_json(json_dicts)
        return results

    def predict_instance(self, instance):
        return self.predict_batch_instance([instance])[0]

    def _json_to_labeled_instance(self, json):
        instance = self._json_to_instance(json)
        result = self.predict_json(json)
        ret = self.predictions_to_labeled_instances(instance, result)[0]
        return ret

    def set_ensemble_num(self, ensemble_num):
        self._ensemble_num = ensemble_num

    def set_transform_fn(self, transform_fn):
        if not check_identity(transform_fn):
            self._transform_fn = transform_fn
        else:
            logger.info("Identity transformation detected.")

    def set_transform_field(self, transform_field):
        self._transform_field = transform_field

    def set_grad_enabled(self, grad_enabled):
        self._grad_enabled = grad_enabled
        
    def set_max_tokens(self, max_tokens):
        self._max_tokens = max_tokens
    
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


def check_identity(fn):
    if fn is None:
        return True
    x = ["this method is used to check whether a function that ",
         "maps a list of string to another is an identity function ",
         "lambda x: x. The id of input should be equal to output "]
    y = fn(x)
    return id(x) == id(y)


@lru_cache(maxsize=None)
def guess_bert(model):
    counts = 0
    for name, param in model.named_parameters():
        if 'bert' in name:
            counts += 1
    return counts > 3


@lru_cache(maxsize=None)
def guess_max_batch(sent_size, using_bert=False):
    """
        Here are some statistics of GPU usage of different models:
        (sentence length x batch size -> GPU usage)
            - BERT-BASE -> 1077
                297 * 100 -> 3115 (+2038)
                297 * 300 -> 7171 (+4056)
            For Bert, 30000 tokens -> 2000MB
            - LSTM      ->  655 (embedding: 38689x300, LSTM: 2x300)
                297 * 100 ->  823 (+ 168)
                297 * 300 -> 1131 (+ 308)
                297 * 700 -> 1741 (+ 610)
                297 * 2000-> 3765 (+2034)
            For LSTM, 30000 tokens -> 150MB
        Genrally speaking, I can deduce the maximum number of batches to forward,
        but the result is not stable, especially on BERT, it may cause OOM on extreme
        cases. The model forwarding procedure is not very time-consuming, instead the
        data processing may cost more time, so I specify a number for BERT/non-BERT.
    """
    # TOTAL = 12000
    # EPS = 2000
    # DECAY = 0.7
    # MODEL_SIZE = 2000
    # if using_bert:
    #     max_tokens = int((TOTAL - EPS - MODEL_SIZE) / (2000 / 30000) * DECAY)
    # else:
    #     max_tokens = int((TOTAL - EPS - MODEL_SIZE) / (150 / 30000) * DECAY)
    if using_bert:
        max_tokens = 60000
    else:
        max_tokens = 120000
    ret = min(max_tokens // sent_size, 2048)
    # print(f"[GUESS] sent_size is {sent_size}, the max batch is set to {ret}")
    return ret
