from allennlpx.modules.token_embedders.embedding import \
    _read_embeddings_from_text_file
from luna import (LabelSmoothingLoss, auto_create)
from allennlp.data.vocabulary import Vocabulary
from allennlpx.modules.token_embedders.embedding import VanillaEmbedding
# from allennlpx.modules.token_embedders.graph_embedding import GraphEmbedding
from awesome_glue.weighted_embedding import WeightedEmbedding
from collections import defaultdict
import pathlib
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.vocabulary import _read_pretrained_tokens
from allennlpx.interpret.attackers.searchers import EmbeddingNbrUtil
from tqdm import tqdm
import json
import numpy as np
import random
import os


def read_weight(vocab: Vocabulary, pretrain: str, cache_embed_path: str):
    if pretrain == 'random':
        return None
    embedding_path = WORD2VECS[pretrain]

    def __read_fn():
        return _read_embeddings_from_text_file(
            embedding_path,
            embedding_dim=EMBED_DIM[pretrain],
            vocab=vocab,
            namespace="tokens")

    if cache_embed_path:
        weight = auto_create(cache_embed_path, __read_fn, True)
    else:
        weight = __read_fn()
    return weight


def get_bert_vocab():
    vocab = Vocabulary(padding_token='[PAD]', oov_token='[UNK]')
    bert_indexer = PretrainedTransformerIndexer("bert-base-uncased", "tokens")
    bert_indexer._add_encoding_to_vocabulary_if_needed(vocab)
    assert vocab.get_vocab_size('tokens') == 30522
    return vocab


def build_embedding(vocab: Vocabulary, pretrain: str, finetune: bool, cache_embed_path: str):
    return VanillaEmbedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=EMBED_DIM[pretrain],
        weight=read_weight(vocab, pretrain, cache_embed_path),
        #  projection_dim=100,
        sparse=False,
        trainable=finetune)


# def build_graph_embedding(vocab: Vocabulary, pretrain: str,
#                           cache_embed_path: str, gnn, edges, hop):
#     return GraphEmbedding(
#         num_embeddings=vocab.get_vocab_size('tokens'),
#         embedding_dim=EMBED_DIM[pretrain],
#         weight=read_weight(vocab, pretrain, cache_embed_path),
#         #   projection_dim=100,
#         gnn=gnn,
#         edges=edges,
#         hop=hop,
#         sparse=False,
#         trainable=True)


def build_weighted_embedding(vocab: Vocabulary, pretrain: str, finetune: bool,
                             cache_embed_path: str, hull):
    return WeightedEmbedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=EMBED_DIM[pretrain],
        weight=read_weight(vocab, pretrain, cache_embed_path),
        hull=hull,
        #   projection_dim=100,
        sparse=False,
        trainable=finetune)


def generate_neighbours(vocab, file_name, measure='euc', topk=8, rho=0.6):
    if vocab is None:
        tokens = _read_pretrained_tokens(WORD2VECS['counter'])
        vocab = Vocabulary(tokens_to_add={"tokens": tokens})

    embed = read_weight(vocab, "counter", None)
    emb_util = EmbeddingNbrUtil(embed, vocab.get_token_index,
                                vocab.get_token_from_index)
    if rho is None:
        emb_util.pre_search(measure, topk + 1, None)

    nbr_num = []
    ret = {}
    tokens = list(vocab.get_token_to_index_vocabulary("tokens").keys())
    if file_name is None:
        tokens = random.choices(tokens, k=100)
    for ele in tqdm(tokens):
        nbrs = emb_util.find_neighbours(ele,
                                        measure,
                                        topk + 1,
                                        rho,
                                        return_words=True)
        if ele in nbrs:
            nbrs.remove(ele)
        ret[ele] = nbrs
        nbr_num.append(len(nbrs))
    print(nbr_num)
    print('Average neighbour num:', np.mean(nbr_num))
    if file_name is None:
        return
    json.dump(ret, open(f"external_data/{file_name}", "w"))


def maybe_path(*args):
    for arg in args:
        if os.access(arg, os.W_OK):
            if pathlib.Path(arg).exists():
                break
    return arg


WORD2VECS = {
    "fasttext":
    maybe_path(
        "/disks/sdb/zjiehang/embeddings/fasttext/crawl-300d-2M.vec",
        "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
    ),
    "glove":
    maybe_path("/disks/sdb/zjiehang/embeddings/glove/glove.42B.300d.txt",
               "/root/glove/glove.42B.300d.txt",
               "http://nlp.stanford.edu/data/glove.42B.300d.zip"),
    "counter":
    maybe_path(
        "/disks/sdb/zjiehang/embeddings/counter/counter.txt",
        "https://raw.githubusercontent.com/nmrksic/counter-fitting/master/word_vectors/counter-fitted-vectors.txt.zip"
    ),
    "random": None
}

EMBED_DIM = defaultdict(lambda: 300, {"elmo": 256, "bert": 768})
