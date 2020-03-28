from allennlpx.modules.token_embedders.embedding import \
    _read_pretrained_embeddings_file
from awesome_glue.utils import EMBED_DIM, WORD2VECS
from luna import (LabelSmoothingLoss, auto_create)
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlpx.modules.token_embedders.graph_embedding import GraphEmbedding
from allennlpx.modules.token_embedders.dirichlet_embedding import DirichletEmbedding


def read_weight(vocab: Vocabulary, pretrain: str, cache_embed_path: str):
    embedding_path = WORD2VECS[pretrain]
    weight = auto_create(
        cache_embed_path, lambda: _read_pretrained_embeddings_file(
            embedding_path,
            embedding_dim=EMBED_DIM[pretrain],
            vocab=vocab,
            namespace="tokens"), True)
    return weight


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
