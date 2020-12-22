import csv
import random
from typing import Callable, Dict, List, Union

import nltk
import torch
from allennlp.data import Instance
from allennlp.data.batch import Batch
from overrides import overrides
from luna.registry import setup_registry
from fastnumbers import fast_real
from allennlpx.interpret.attackers.searchers import CachedWordSearcher

register, R = setup_registry('transforms')

# same definition as TensorDict in allennlp.data.dataloader
TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]


def transform_collate(
        vocab,  # Use vocab to index the transformed instances
        reader,  # call reader's function to transform instances
        transform: Callable,
        instances: List[Instance]) -> TensorDict:
    new_instances = reader.transform_instances(transform, instances)
    batch = Batch(new_instances)
    batch.index_instances(vocab)
    ret = batch.as_tensor_dict(batch.get_padding_lengths())
    return ret


class Transform:
    def __call__(self, xs: List[str]) -> List[str]:
        raise NotImplementedError


class WordTransform(Transform):
    def __init__(self, change_num_or_ratio=0.15):
        self.change_num_or_ratio = change_num_or_ratio

    def change_num(self, x_len):
        if self.change_num_or_ratio < 1:
            change_num = max(1, int(x_len * self.change_num_or_ratio))
        else:
            change_num = min(self.change_num_or_ratio, x_len)
        return change_num

        
@register('crop')
class Crop(Transform):
    def __init__(self, crop_ratio=0.3):
        self.crop_ratio = crop_ratio

    @overrides
    def __call__(self, xs):
        ys = []
        for x in xs:
            x_split = x.split(" ")
            x_len = len(x_split)
            # crop at least 5 words
            crop_num = max(int(self.crop_ratio * x_len), 5)
            if x_len <= crop_num:
                ys.append(x)
            else:
                start_crop_idx = random.choice(range(x_len - crop_num))
                end_crop_idx = start_crop_idx + crop_num
                ys.append(" ".join(x_split[start_crop_idx:end_crop_idx]))
        return ys


@register('identity')
class Identity(Transform):
    @overrides
    def __call__(self, xs):
        return xs


@register('rand_drop')
class RandDrop(WordTransform):
    @overrides
    def __call__(self, xs):
        ys = []
        for x in xs:
            x_split = x.split(" ")
            if len(x_split) < 5:
                ys.append(x)
            for i in range(self.change_num(len(x_split))):
                x_split.pop(random.randrange(len(x_split)))
            ys.append(" ".join(x_split))
        return ys


@register('embed_aug')
class EmbedAug(WordTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.searcher = CachedWordSearcher(
            "external_data/ibp-nbrs.json",
            None,
            second_order=False
        )

    @overrides
    def __call__(self, xs):
        ys = []
        for x in xs:
            x_split = x.split(" ")
            for cid in random.sample(range(len(x_split)), k=self.change_num(len(x_split))):
                word = x_split[cid]
                nbrs = self.searcher.search(word)
                if len(nbrs) > 1:
                    x_split[cid] = random.choice(nbrs)
            ys.append(" ".join(x_split))
        return ys


# @register('syn_aug')
# class SynAug(WordTransform):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         nltk.download('wordnet')
#         nltk.download('averaged_perceptron_tagger')
#         self.aug = naw.SynonymAug()

#     @overrides
#     def __call__(self, xs):
#         ys = []
#         for x in xs:
#             x_split = x.split(" ")
#             self.aug.aug_min = self.aug.aug_max = self.change_num(len(x_split))
#             ys.append(self.aug.substitute(x))
#         return ys


# @register('bert_aug')
# class BertAug(WordTransform):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased',
#                                              top_k=10,
#                                              action="substitute")

#     @overrides
#     def __call__(self, xs):
#         ys = []
#         for x in xs:
#             x_split = x.split(" ")
#             self.aug.aug_min = self.aug.aug_max = self.change_num(len(x_split))
#             ys.append(self.aug.substitute(x))
#         return ys


def parse_transform_fn_from_args(tf_names, tf_args):
    """
    This function is used to combine a group of transform functions.
    Case 1:
        tf_names = crop
        tf_args = 0.2
    Case 2:
        tf_names = crop|embed_aug
        tf_args = 0.4|0.2
        In this case, the data will be passed through them one-by-one.
    """
    if "|" in tf_names:
        tf_names = tf_names.split("|")
        tf_args = tf_args.split("|")
        assert len(tf_names) == len(tf_args)
    else:
        if tf_names == '':
            tf_names = 'identity'
        tf_names = [tf_names]
        tf_args = [tf_args]
    tf_args = list(map(fast_real, tf_args))
    tf_objs = []
    for tf_name, tf_arg in zip(tf_names, tf_args):
        tf_cls = R[tf_name]
        if tf_arg == '':
            tf_obj = tf_cls()
        else:
            tf_obj = tf_cls(tf_arg)
        tf_objs.append(tf_obj)
    def chained(xs):
        for obj in tf_objs:
            xs = obj(xs)
        return xs
    return chained
    