import random
import nltk
import nlpaug.augmenter.word as naw
from luna import flt2str, ram_append, ram_has, ram_read, ram_reset, ram_write


nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def gen_aug_num(aug_num_or_ratio, x):
    x_len = len(x.split(" "))
    if aug_num_or_ratio < 1:
        aug_num = max(1, int(x_len * aug_num_or_ratio))
    else:
        aug_num = min(aug_num_or_ratio, x_len)
    return aug_num


def identity(x):
    return x

def rand_drop(aug_num_or_ratio, x):
    x_split = x.split(" ")
    for i in range(gen_aug_num(aug_num_or_ratio, x)):
        x_split.pop(random.randrange(len(x_split)))
    return " ".join(x_split)

def embed_aug(aug_num_or_ratio, x):
    if ram_has("emb_aug"):
        aug = ram_read("emb_aug")
    else:
        aug = naw.WordEmbsAug(  
            model_type = 'glove',
            top_k=10,
            model_path = '/home/zhouyi/counter-fitting/word_vectors/counter-fitted-vectors.txt',
        )
        ram_write("emb_aug", aug)
    aug.aug_min = aug.aug_max = gen_aug_num(aug_num_or_ratio, x)
    ret = aug.substitute(x)
    return ret

def syn_aug(aug_num_or_ratio, x):
    if ram_has("syn_aug"):
        aug = ram_read("syn_aug")
    else:
        aug = naw.SynonymAug()
        ram_write("syn_aug", aug)
    aug.aug_min = aug.aug_max = gen_aug_num(aug_num_or_ratio, x)
    ret = aug.substitute(x)
    return ret

def bert_aug(aug_num_or_ratio, x):
    if ram_has("bert_aug"):
        aug = ram_read("bert_aug")
    else:
        aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased', 
            top_k = 10,
            action="substitute")
        ram_write("bert_aug", aug)
    aug.aug_min = aug.aug_max = gen_aug_num(aug_num_or_ratio, x)
    ret = aug.augment(x)
    return ret    
