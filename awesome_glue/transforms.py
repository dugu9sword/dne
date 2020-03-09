import random
from typing import Callable, Dict, List, Union

import nlpaug.augmenter.word as naw
import nltk
import torch
from allennlp.data import Instance
from allennlp.data.batch import Batch
from overrides import overrides

# same definition as TensorDict in allennlp.data.dataloader
TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]


def transform_collate(
        reader,
        vocab,
        transform: Callable, 
        instances: List[Instance]
    ) -> TensorDict:
    new_instances = reader.transform_instances(transform, instances)

    batch = Batch(new_instances)
    batch.index_instances(vocab)
    ret = batch.as_tensor_dict(batch.get_padding_lengths())
#     print(ret)
#     exit()
    return ret


class Transform:
    def __call__(self, xs: List[str]) -> List[str]:
        raise NotImplemented

        
class WordTransform(Transform):
    def __init__(self, change_num_or_ratio = 0.15):
        self.change_num_or_ratio = change_num_or_ratio  
    
    def change_num(self, x_len):
        if self.change_num_or_ratio < 1:
            change_num = max(1, int(x_len * self.change_num_or_ratio))
        else:
            change_num = min(self.change_num_or_ratio, x_len)
        return change_num
    
    
class Crop(Transform):
    def __init__(self, crop_ratio = 0.3):
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
                ys.append(" ".join(x_split[start_crop_idx: end_crop_idx]))
        return ys
            
        
class Identity(Transform):
    @overrides
    def __call__(self, xs):
        return xs
    

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

    
class EmbedAug(WordTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug = naw.WordEmbsAug(  
            model_type = 'glove',
            top_k=10,
            model_path = '/home/zhouyi/counter-fitting/word_vectors/counter-fitted-vectors.txt',
        )

    @overrides
    def __call__(self, xs):
        ys = []
        for x in xs:
            x_split = x.split(" ")
            self.aug.aug_min = self.aug.aug_max = self.change_num(len(x_split))
            ys.append(self.aug.substitute(x))
        return ys

    
class SynAug(WordTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        self.aug = naw.SynonymAug()
        
    @overrides
    def __call__(self, xs):
        ys = []
        for x in xs:
            x_split = x.split(" ")
            self.aug.aug_min = self.aug.aug_max = self.change_num(len(x_split))
            ys.append(self.aug.substitute(x))
        return ys 


class BertAug(WordTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased', 
            top_k = 10,
            action="substitute"
        )
        
    @overrides
    def __call__(self, xs):
        ys = []
        for x in xs:
            x_split = x.split(" ")
            self.aug.aug_min = self.aug.aug_max = self.change_num(len(x_split))
            ys.append(self.aug.substitute(x))
        return ys 
