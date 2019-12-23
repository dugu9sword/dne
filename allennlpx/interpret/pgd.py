# pylint: disable=protected-access
from copy import deepcopy
from typing import List

import numpy
import torch

from allennlp.common.util import JsonDict, sanitize
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.interpret.attackers import utils
from allennlp.interpret.attackers.attacker import Attacker
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding

from allennlpx.predictors.predictor import PredictorX
from luna import cast_list
from luna import lazy_property
from .embedding_searcher import EmbeddingSearcher
from .util import select

DEFAULT_IGNORE_TOKENS = ["@@NULL@@", ".", ",", ";", "!", "?", "[MASK]", "[SEP]", "[CLS]", "-LRB-", "-RRB-"]


class PGD(Attacker):
    """
    Runs the HotFlip style attack at the word-level https://arxiv.org/abs/1712.06751.  We use the
    first-order taylor approximation described in https://arxiv.org/abs/1903.06620, in the function
    _first_order_taylor(). Constructing this object is expensive due to the construction of the
    embedding matrix.
    """
    def __init__(self, predictor: PredictorX) -> None:
        super().__init__(predictor)
        self.vocab = self.predictor._model.vocab
        self.token_embedding = None

    @lazy_property
    def embed_searcher(self) -> EmbeddingSearcher:
        return EmbeddingSearcher(embed=self.token_embedding.weight,
                                 idx2word=lambda x: self.vocab.get_token_from_index(x),
                                 word2idx=lambda x: self.vocab.get_token_index(x))

    def initialize(self):
        """
        Call this function before running attack_from_json(). We put the call to
        ``_construct_embedding_matrix()`` in this function to prevent a large amount of compute
        being done when __init__() is called.
        """
        self.token_embedding = self._construct_embedding_matrix()

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
        all_inputs = {"tokens": torch.LongTensor(all_indices).unsqueeze(0)}
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
                    padded_tokens['token_characters']).unsqueeze(0)
            # for ELMo models
            if isinstance(token_indexer, ELMoTokenCharactersIndexer):
                elmo_tokens = []
                for token in all_tokens:
                    elmo_indexed_token = token_indexer.tokens_to_indices([Token(text=token)],
                                                                         self.vocab,
                                                                         "sentence")["sentence"]
                    elmo_tokens.append(elmo_indexed_token[0])
                all_inputs["elmo"] = torch.LongTensor(elmo_tokens).unsqueeze(0)

        # find the TextFieldEmbedder
        for module in self.predictor._model.modules():
            if isinstance(module, TextFieldEmbedder):
                embedder = module
        # pass all tokens through the fake matrix and create an embedding out of it.
        embedding_matrix = embedder(all_inputs).squeeze()
        return Embedding(num_embeddings=self.vocab.get_vocab_size('tokens'),
                         embedding_dim=embedding_matrix.shape[1],
                         weight=embedding_matrix,
                         trainable=False)

    def attack_from_json(self,
                         inputs: JsonDict,
                         field_to_change: str = 'tokens',
                         field_to_attack: str = 'label',
                         grad_input_field: str = 'grad_input_1',
                         step_size: float = 100.,
                         max_step: int = 20,
                         max_change_num: int = 3,
                         iter_change_num: int = 2,
                         ignore_tokens: List[str] = DEFAULT_IGNORE_TOKENS,
                         forbidden_tokens: List[str] = DEFAULT_IGNORE_TOKENS) -> JsonDict:
        if self.token_embedding is None:
            self.initialize()

        raw_instance = self.predictor.json_to_labeled_instances(inputs)[0]
        raw_text_field: TextField = raw_instance[field_to_change]
        raw_tokens = raw_text_field.tokens

        att_instance = deepcopy(raw_instance)
        att_text_field: TextField = att_instance[field_to_change]
        att_tokens = att_text_field.tokens

        # set up some states
        change_positions__ = set()
        forbidden_idxs__ = set()
        for forbidden_token in forbidden_tokens:
            if forbidden_token in self.vocab._token_to_index['tokens']:
                forbidden_idxs__.add(self.vocab._token_to_index['tokens'][forbidden_token])

        for step in range(max_step):
            grads, _ = self.predictor.get_gradients([att_instance])
            grad = grads[grad_input_field]
            grad_norm = grad.norm(dim=-1)

            position_mask = [False for _ in range(len(att_tokens))]
            is_max_changed = len(change_positions__) > max_change_num
            for idx, token in enumerate(att_tokens):
                if token.text in ignore_tokens:
                    position_mask[idx] = True
                if is_max_changed and idx not in change_positions__:
                    position_mask[idx] = True
            if all(position_mask):
                print("All words are forbidden.")
                break
            for idx in range(len(position_mask)):
                if position_mask[idx]:
                    grad_norm[idx] = -1

            # select a word and forbid itself
            token_vids: List[int] = []
            new_token_vids: List[int] = []

            _, topk_idxs = grad_norm.sort(descending=True)
            token_sids = select(ordered_idxs=cast_list(topk_idxs),
                                num_to_select=iter_change_num,
                                selected=change_positions__,
                                max_num=max_change_num)
            token_sids = [ele for ele in token_sids if position_mask[ele] is False]

            for token_sid in token_sids:
                token_grad = grad[token_sid]

                token_vid = att_text_field._indexed_tokens["tokens"][token_sid]

                token_emb = self.token_embedding.weight[token_vid]

                change_positions__.add(token_sid)
                forbidden_idxs__.add(token_vid)

                # print(self.change_positions__)

                delta = token_grad / torch.norm(token_grad) * step_size
                new_token_emb = token_emb + delta

                tk_vals, tk_idxs = self.embed_searcher.find_neighbours(
                    new_token_emb, -1, 'euc', False)
                for tk_idx in cast_list(tk_idxs):
                    if tk_idx in forbidden_idxs__:
                        continue
                    else:
                        new_token_vid = tk_idx
                        break

                token_vids.append(token_vid)
                new_token_vids.append(new_token_vid)

                # flip token
                new_token = Token(
                    self.vocab._index_to_token["tokens"][new_token_vid])  # type: ignore
                att_text_field.tokens[token_sid] = new_token

            att_instance.indexed = False

            # Get model predictions on current_instance, and then label the instances
            grads, outputs = self.predictor.get_gradients([att_instance])  # predictions
            for key, output in outputs.items():
                outputs[key] = cast_list(outputs[key])

            # add labels to current_instances
            current_instance_labeled = self.predictor.predictions_to_labeled_instances(
                att_instance, outputs)[0]
            # if the prediction has changed, then stop
            if current_instance_labeled[field_to_attack] != raw_instance[field_to_attack]:
                break

#             raw_instance.fields['tokens'].tokens

            # print(f'Replace {token_sid} from {token_vid} to {new_token_vid}')
            # print(instance2str(att_instance))

        return sanitize({"att": att_tokens, "raw": raw_tokens, "outputs": outputs})


