from typing import List
from functools import lru_cache

import torch
from allennlp.common import Registrable
from allennlp.common.util import JsonDict
from allennlp.data.token_indexers.elmo_indexer import \
    ELMoTokenCharactersIndexer
from allennlp.data.token_indexers.token_characters_indexer import \
    TokenCharactersIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.modules.text_field_embedders.text_field_embedder import \
    TextFieldEmbedder
from allennlp.data.tokenizers import SpacyTokenizer, Token

from allennlpx.predictors.predictor import Predictor
from luna import cast_list, lazy_property
from allennlpx.interpret.attackers.searchers import SynonymSearcher, EmbeddingSearcher
from allennlp.nn.util import find_embedding_layer

DEFAULT_IGNORE_TOKENS = [
    "@@NULL@@", "@@PADDING@@", "@@UNKNOWN@@", ".", ",", ";", "!", "?", "[MASK]", "[SEP]", "[CLS]",
    "-LRB-", "-RRB-", "(", ")", "[", "]", "-", "$", "&", "*", "...", "'", '"'
]  # + stopwords.words("english")


class Attacker(Registrable):
    def __init__(
        self,
        predictor: Predictor,
        # change a field to attack another field
        field_to_change: str = 'sent',
        field_to_attack: str = 'label',
        use_bert: bool = False,
        *,  # only accept keyword arguments
        ignore_tokens: List[str] = DEFAULT_IGNORE_TOKENS,
        forbidden_tokens: List[str] = DEFAULT_IGNORE_TOKENS,
        max_change_num_or_ratio=None,
        vocab=None,
        token_embedding=None):
        self.predictor = predictor
        self.f2c = field_to_change
        self.f2a = field_to_attack
        self.use_bert = use_bert

        self.ignore_tokens = ignore_tokens
        self.forbidden_tokens = forbidden_tokens
        self.max_change_num_or_ratio = max_change_num_or_ratio

        self.spacy = SpacyTokenizer()
        if vocab:
            self.vocab = vocab
            self.token_embedding = token_embedding.to(self.model_device)
        else:
            self.vocab = self.predictor._model.vocab
            # self.token_embedding = self._construct_embedding_matrix().to(self.model_device)
            self.token_embedding = find_embedding_layer(self.predictor._model).weight

    def attack_from_json(self, inputs: JsonDict) -> JsonDict:
        raise NotImplementedError()

    @lru_cache(maxsize=None)
    def max_change_num(self, len_tokens):
        if self.max_change_num_or_ratio is None:
            return 100000
        else:
            if self.max_change_num_or_ratio < 1:
                max_change_num = max(1, int(len_tokens * self.max_change_num_or_ratio))
            else:
                max_change_num = self.max_change_num_or_ratio
            return max_change_num

    def _tokens_to_instance(self, tokens):
        return self.predictor._dataset_reader.text_to_instance(" ".join(tokens))

    @property
    def model_device(self):
        return next(self.predictor._model.parameters()).device

    def _construct_embedding_matrix(self):
        """
        For HotFlip, we need a word embedding matrix to search over. The below is necessary for
        models such as ELMo, character-level models, or for models that use a projection layer
        after their word embeddings.

        We run all of the tokens from the vocabulary through the TextFieldEmbedder, and save the
        final output embedding. We then group all of those output embeddings into an "embedding
        matrix".
        """
        # Gets all tokens in the vocab and their corresponding IDs
        all_tokens = self.vocab._token_to_index["tokens"]
        all_indices = list(self.vocab._index_to_token["tokens"].keys())
        all_inputs = {"tokens": torch.LongTensor(all_indices).to(self.model_device).unsqueeze(0)}
        for token_indexer in self.predictor._dataset_reader._token_indexers.values():
            # handle when a model uses character-level inputs, e.g., a CharCNN
            if isinstance(token_indexer, TokenCharactersIndexer):
                tokens = [Token(x) for x in all_tokens]
                max_token_length = max(len(x) for x in all_tokens)
                indexed_tokens = token_indexer.tokens_to_indices(tokens, self.vocab,
                                                                 "token_characters")
                padded_tokens = token_indexer.as_padded_tensor(
                    indexed_tokens, {"token_characters": len(tokens)},
                    {"num_token_characters": max_token_length})
                all_inputs['token_characters'] = torch.LongTensor(
                    padded_tokens['token_characters']).to(self.model_device).unsqueeze(0)
            # for ELMo models
            if isinstance(token_indexer, ELMoTokenCharactersIndexer):
                elmo_tokens = []
                for token in all_tokens:
                    elmo_indexed_token = token_indexer.tokens_to_indices([Token(text=token)],
                                                                         self.vocab,
                                                                         "sentence")["sentence"]
                    elmo_tokens.append(elmo_indexed_token[0])
                all_inputs["elmo"] = torch.LongTensor(elmo_tokens).to(
                    self.model_device).unsqueeze(0)

        # find the TextFieldEmbedder
        for module in self.predictor._model.modules():
            if isinstance(module, TextFieldEmbedder):
                embedder = module
        # pass all tokens through the fake matrix and create an embedding out of it.
        embedding_matrix = embedder(all_inputs).squeeze()
        return embedding_matrix

    @lazy_property
    def embed_searcher(self) -> EmbeddingSearcher:
        return EmbeddingSearcher(embed=self.token_embedding,
                                 idx2word=self.vocab.get_token_from_index,
                                 word2idx=self.vocab.get_token_index)

    @lazy_property
    def synom_searcher(self) -> SynonymSearcher:
        return SynonymSearcher(vocab_list=self.vocab.get_index_to_token_vocabulary().values())

    @lru_cache(maxsize=None)
    def neariest_neighbours(self, word, measure, topk, rho):
        # May be accelerated by caching a the distance
        vals, idxs = self.embed_searcher.find_neighbours(word, measure=measure, topk=topk, rho=rho)
        if idxs is None:
            return []
        else:
            return [self.vocab.get_token_from_index(idx) for idx in cast_list(idxs)]
