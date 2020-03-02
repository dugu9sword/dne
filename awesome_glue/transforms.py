import random
import nltk
import nlpaug.augmenter.word as naw
from luna import flt2str, ram_append, ram_has, ram_read, ram_reset, ram_write


nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def identity(x):
    return x

def rand_drop(x):
    x_split = x.split(" ")
    for i in range(min(3, len(x_split) - 1)):
        x_split.pop(random.randrange(len(x_split)))
    return " ".join(x_split)

def rand_stop(x):
    x_split = x.split(" ")
    idxs = random.choices(range(len(x_split)), k=5)
    for i in idxs:
        x_split[i] = 'the'
    return " ".join(x_split)

def embed_aug(x):
    aug_num = 5
    if ram_has("emb_aug"):
        aug = ram_read("emb_aug")
    else:
        aug = naw.WordEmbsAug(  
            model_type = 'glove',
            top_k=30,
            model_path = '/home/zhouyi/counter-fitting/word_vectors/counter-fitted-vectors.txt',
#                     model_path = '/home/zhouyi/counter-fitting/word_vectors/glove.txt',
            aug_min = aug_num,
            aug_max = aug_num,
            stopwords = ['a', 'the'],
            stopwords_regex = '@',
        )
        ram_write("emb_aug", aug)
    if len(x.split(' ')) < aug_num:
        aug.aug_min = 1
    ret = aug.substitute(x)
    aug.aug_min = aug_num
    return ret

def syn_aug(x):
    aug_num = 5
    if ram_has("syn_aug"):
        aug = ram_read("syn_aug")
    else:
        aug = naw.SynonymAug(
            aug_min = aug_num,
            aug_max = aug_num,
            stopwords = ['a', 'the'],
            stopwords_regex = '@',
        )
        ram_write("syn_aug", aug)
    if len(x.split(' ')) < aug_num:
        aug.aug_min = 1
    ret = aug.substitute(x)
    aug.aug_min = aug_num
    return ret

def bert_aug(x):
    aug_num = 5
    if ram_has("bert_aug"):
        aug = ram_read("bert_aug")
    else:
        aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased', 
            top_k = 30,
            aug_min = aug_num,
            aug_max = aug_num,
            stopwords = ['a', 'the'],
            stopwords_regex = '@',
            action="substitute")
        ram_write("bert_aug", aug)
    if len(x.split(' ')) < aug_num:
        aug.aug_min = 1
    ret = aug.augment(x)
    aug.aug_min = aug_num
    return ret    
