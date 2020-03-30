import torch
import numpy as np
from typing import List
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.modules.token_embedders.embedding import Embedding

from luna.public import auto_create
cache_path = "/disks/sdb/zjiehang/frequency/cache/conll2003"

class Policy(object):
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab
        
    def generate(self, word: str, number: int = 30):
        raise NotImplementedError()

class EmbeddingPolicy(Policy):
    def __init__(self, 
                 vocab: Vocabulary,
                 embedding: torch.Tensor,
                 token_namespace: str = 'tokens'):
        super().__init__(vocab)
        self.token_namespace = token_namespace
        self.candidates_top100_dict = auto_create("top100_nearest",
                                                  lambda: self.candidate_top100(embedding, token_namespace),
                                                  True,
                                                  cache_path)
    
    def candidate_top100(self, embedding, namespace):
        import faiss
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(embedding.shape[1])
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        gpu_index_flat.add(embedding.cpu().numpy())
        distance, knn_indexes = gpu_index_flat.search(embedding.cpu().numpy(), 100)
        index_to_token_vocab = self.vocab.get_index_to_token_vocabulary(namespace)
        oov_index = self.vocab.get_token_index(DEFAULT_OOV_TOKEN,namespace)
        padding_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN,namespace)
        knn_neiborgh = {}
        # delele <UNK> and <PAD>
        for index, knn_index in enumerate(knn_indexes):
            if index != oov_index and index != padding_index:
            # delete itself
                knn_neiborgh[index_to_token_vocab[index]] = [index_to_token_vocab[knn] for knn in knn_index[1:] if knn!=padding_index and knn!=oov_index]
        return knn_neiborgh
    
    def generate(self, word: str, number: int = 30) -> List[str]:
        candidate_top100_current_word = self.candidates_top100_dict[word]
        if len(candidate_top100_current_word) < number:
            return candidate_top100_current_word
        else:
            return candidate_top100_current_word[:number]


class RandomPolicy(Policy):
    def __init__(self, 
                 vocab: Vocabulary,
                 token_namespace: str = 'tokens'):
        super().__init__(vocab)
        self.vocab_length = self.vocab.get_vocab_size(token_namespace)
        self.token_namespace = token_namespace
        
    def generate(self, word: str, number: int = 30) -> List[str]:
        random_int = np.random.randint(2, self.vocab_length, size=(number))
        index_to_token_vocab = self.vocab.get_index_to_token_vocabulary(self.token_namespace)
        return [index_to_token_vocab[rand_int] for rand_int in random_int]


def get_policy_from_str(vocab: Vocabulary, policy_type: str = 'random', embedding: Embedding = None) -> Policy:
    if policy_type == 'random':
        return RandomPolicy(vocab)
    elif policy_type == 'embedding':
        if embedding is None:
            raise ("Embedding policy needs pretrained embedding weight")
        return EmbeddingPolicy(vocab, embedding.weight.data)
    else:
        raise ("Unsuppored word substitution policy type. ")