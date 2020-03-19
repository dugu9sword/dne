import csv
import random
from typing import Callable, Dict, List, Union

import nlpaug.augmenter.word as naw
import nltk
import torch
from allennlp.data import Instance
from allennlp.data.batch import Batch
from overrides import overrides
from allennlpx.interpret.attackers.policies import EmbeddingPolicy
from allennlp.data.tokenizers import SpacyTokenizer
from luna.registry import setup_registry

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
        raise NotImplemented


class WordTransform(Transform):
    def __init__(self, change_num_or_ratio=0.15):
        self.change_num_or_ratio = change_num_or_ratio

    def change_num(self, x_len):
        if self.change_num_or_ratio < 1:
            change_num = max(1, int(x_len * self.change_num_or_ratio))
        else:
            change_num = min(self.change_num_or_ratio, x_len)
        return change_num


@register('bt')
class BackTrans(Transform):
    def __init__(self):
        import hack_fairseq
        hack_fairseq.use_fairseq_9()
        self.en2z = torch.hub.load('pytorch/fairseq',
                                   'transformer.wmt19.en-de',
                                   checkpoint_file='model1.pt',
                                   tokenizer='moses',
                                   bpe='fastbpe',
                                   verbose=True).cuda()
        self.z2en = torch.hub.load('pytorch/fairseq',
                                   'transformer.wmt19.de-en',
                                   checkpoint_file='model1.pt',
                                   tokenizer='moses',
                                   bpe='fastbpe',
                                   verbose=True).cuda()
        self.en2z.eval()
        self.z2en.eval()

    def __call__(self, xs: List[str]) -> List[str]:
        with torch.no_grad():
            ys = self.en2z.translate(xs, beam=5)
            ys = self.z2en.translate(ys, beam=5)
        return ys
    

@register('dae')
class DAE(Transform):
    def __init__(self):
        import hack_fairseq
        hack_fairseq.use_fairseq_6()
        from fsgec.dae_hub import load_model
        self.translate = load_model()
#         import hack_fairseq
#         hack_fairseq.use_fairseq_9()
#         from fshub import from_pretrained, GeneratorHubInterface
#         self.tokenizer = SpacyTokenizer()
#         self.dae = GeneratorHubInterface(**from_pretrained("advdae/checkpoints/", 
#                                 checkpoint_file='checkpoint_best.pt',
#                                 data_name_or_path = 'advdae/wikitext-bin/'))
#         self.dae.cuda()
#         self.dae.eval()
        
    @overrides
    def __call__(self, xs):
        with torch.no_grad():
            return self.translate(xs)
#         with torch.no_grad():
#             xs = [" ".join([x.text for x in self.tokenizer.tokenize(x)]) for x in xs]
#             return self.dae.translate(xs, no_unk=False)

        
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
        #         self.aug = naw.WordEmbsAug(
        #             model_type = 'glove',
        #             top_k=10,
        #             model_path = '/home/zhouyi/counter-fitting/word_vectors/counter-fitted-vectors.txt',
        #         )
        f = csv.reader(open(EmbeddingPolicy('euc', 10, None).cache_name()),
                       delimiter='\t',
                       quoting=csv.QUOTE_NONE)
        self.nbrs = {}
        for row in f:
            self.nbrs[row[0]] = row[1:]

    @overrides
    def __call__(self, xs):
        ys = []
        for x in xs:
            x_split = x.split(" ")
            #             self.aug.aug_min = self.aug.aug_max = self.change_num(len(x_split))
            #             ys.append(self.aug.substitute(x))
            for cid in random.sample(range(len(x_split)), k=self.change_num(len(x_split))):
                word = x_split[cid]
                if word in self.nbrs:
                    x_split[cid] = random.choice(self.nbrs[word])
            ys.append(" ".join(x_split))
        return ys


@register('syn_aug')
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


@register('bert_aug')
class BertAug(WordTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased',
                                             top_k=10,
                                             action="substitute")

    @overrides
    def __call__(self, xs):
        ys = []
        for x in xs:
            x_split = x.split(" ")
            self.aug.aug_min = self.aug.aug_max = self.change_num(len(x_split))
            ys.append(self.aug.substitute(x))
        return ys


def chaining(objs: List[Transform]):
    def chained(xs):
        for obj in objs:
            xs = obj[xs]
        return xs
    return chained