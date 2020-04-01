from allennlpx.modules.token_embedders.embedding import \
    _read_pretrained_embeddings_file
from luna import (LabelSmoothingLoss, auto_create)
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlpx.modules.token_embedders.graph_embedding import GraphEmbedding
from allennlpx.modules.token_embedders.dirichlet_embedding import DirichletEmbedding
from collections import defaultdict
import pathlib
from allennlp.data.token_indexers import PretrainedTransformerIndexer


def read_weight(vocab: Vocabulary, pretrain: str, cache_embed_path: str):
    embedding_path = WORD2VECS[pretrain]
    weight = auto_create(
        cache_embed_path, lambda: _read_pretrained_embeddings_file(
            embedding_path,
            embedding_dim=EMBED_DIM[pretrain],
            vocab=vocab,
            namespace="tokens"), True)
    return weight


def build_bert_vocab_and_vec(pretrain):
    vocab = Vocabulary(padding_token='[PAD]', oov_token='[UNK]')
    bert_indexer = PretrainedTransformerIndexer("bert-base-uncased", "tokens")
    bert_indexer._add_encoding_to_vocabulary_if_needed(vocab)
    assert vocab.get_vocab_size('tokens') == 30522
    vec = read_weight(
        vocab, pretrain,f"{pretrain}-for-bert.vec"
    )
    return vocab, vec


def build_embedding(vocab: Vocabulary, pretrain: str, cache_embed_path: str):
    return Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                     embedding_dim=EMBED_DIM[pretrain],
                     weight=read_weight(vocab, pretrain, cache_embed_path),
                     sparse=True,
                     trainable=True)


def build_graph_embedding(vocab: Vocabulary, pretrain: str,
                          cache_embed_path: str, gnn, edges, hop):
    return GraphEmbedding(num_embeddings=vocab.get_vocab_size('tokens'),
                          embedding_dim=EMBED_DIM[pretrain],
                          weight=read_weight(vocab, pretrain,
                                             cache_embed_path),
                          gnn=gnn,
                          edges=edges,
                          hop=hop,
                          sparse=False,
                          trainable=True)


def build_dirichlet_embedding(vocab: Vocabulary, pretrain: str,
                              cache_embed_path: str, temperature, neighbours,
                              nbr_mask):
    return DirichletEmbedding(num_embeddings=vocab.get_vocab_size('tokens'),
                              embedding_dim=EMBED_DIM[pretrain],
                              weight=read_weight(vocab, pretrain,
                                                 cache_embed_path),
                              temperature=temperature,
                              neighbours=neighbours,
                              nbr_mask=nbr_mask,
                              sparse=False,
                              trainable=True)


def maybe_path(*args):
    for arg in args:
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
    )
}

EMBED_DIM = defaultdict(lambda: 300, {"elmo": 256, "bert": 768})
