# pylint: disable=protected-access
from copy import deepcopy
from typing import List

import numpy
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import (ELMoTokenCharactersIndexer,
                                          TokenCharactersIndexer)
from allennlp.data.tokenizers import Token
from allennlp.modules.text_field_embedders.text_field_embedder import \
    TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding

from allennlpx.interpret.attackers.attacker import (DEFAULT_IGNORE_TOKENS,
                                                    Attacker)
from allennlpx.interpret.attackers.embedding_searcher import EmbeddingSearcher
from allennlpx.interpret.attackers.util import select
from allennlpx.predictors.predictor import Predictor
from luna import cast_list, lazy_property


class PGD(Attacker):
    def __init__(self, 
                 predictor, 
                 *,
                 step_size: float = 100.,
                 max_step: int = 20,
                 iter_change_num: int = 2,
                 **kwargs,
                ):
        super().__init__(predictor, **kwargs)
        self.step_size = step_size
        self.max_step = max_step
        self.iter_change_num = iter_change_num
        
    def attack_from_json(self,
                         inputs: JsonDict,
                         field_to_change: str = 'tokens',
                         field_to_attack: str = 'label',
                         grad_input_field: str = 'grad_input_1',
                         ) -> JsonDict:
        raw_instance = self.predictor.json_to_labeled_instances(inputs)[0]
        raw_text_field: TextField = raw_instance[field_to_change]
        raw_tokens = raw_text_field.tokens

        adv_instance = deepcopy(raw_instance)
        adv_text_field: TextField = adv_instance[field_to_change]
        adv_tokens = adv_text_field.tokens
        
        # equal to the raw one...
        _, outputs = self.predictor.get_gradients([adv_instance])

        # set up some states
        change_positions__ = set()
        forbidden_idxs__ = set()
        for forbidden_token in self.forbidden_tokens:
            if forbidden_token in self.vocab._token_to_index['tokens']:
                forbidden_idxs__.add(self.vocab._token_to_index['tokens'][forbidden_token])

        successful = False
        for step in range(self.max_step):
            grads, _ = self.predictor.get_gradients([adv_instance])
            grad = torch.from_numpy(grads[grad_input_field][0]).to(self.model_device)
            grad_norm = grad.norm(dim=-1)

            position_mask = [False for _ in range(len(adv_tokens))]
            is_max_changed = len(change_positions__) > self.max_change_num(len(raw_tokens))

            for idx, token in enumerate(adv_tokens):
                if token.text in self.ignore_tokens:
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
                                num_to_select=self.iter_change_num,
                                selected=change_positions__,
                                max_num=self.max_change_num(len(raw_tokens)))
            token_sids = [ele for ele in token_sids if position_mask[ele] is False]

            for token_sid in token_sids:
                token_grad = grad[token_sid]

                token_vid = adv_text_field._indexed_tokens["tokens"][token_sid]

                token_emb = self.token_embedding[token_vid]

                change_positions__.add(token_sid)
                forbidden_idxs__.add(token_vid)

#                 print(change_positions__)

                delta = token_grad / torch.norm(token_grad) * self.step_size
                new_token_emb = token_emb + delta

                tk_vals, tk_idxs = self.embed_searcher.find_neighbours(
                    new_token_emb, 'cos', topk=None, rho=None)
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
                adv_text_field.tokens[token_sid] = new_token

            adv_instance.indexed = False

            # Get model predictions on current_instance, and then label the instances
            grads, outputs = self.predictor.get_gradients([adv_instance])  # predictions
            for key, output in outputs.items():
                outputs[key] = cast_list(outputs[key])

            # add labels to current_instances
            current_instance_labeled = self.predictor.predictions_to_labeled_instances(
                adv_instance, outputs)[0]
            # if the prediction has changed, then stop
            if current_instance_labeled[field_to_attack] != raw_instance[field_to_attack]:
                successful = True
                break

        return sanitize({
            "adv": adv_tokens,
            "raw": raw_tokens,
            "outputs": outputs,
            "success": 1 if successful else 0
        })
