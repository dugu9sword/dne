# pylint: disable=protected-access
from copy import deepcopy
from typing import List

import numpy
import torch

from allennlp.common.util import JsonDict, sanitize
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from luna import cast_list

from allennlpx.interpret.attackers.attacker import Attacker, DEFAULT_IGNORE_TOKENS



class HotFlip(Attacker):
    """
    Runs the HotFlip style attack at the word-level https://arxiv.org/abs/1712.06751.  We use the
    first-order taylor approximation described in https://arxiv.org/abs/1903.06620, in the function
    _first_order_taylor(). Constructing this object is expensive due to the construction of the
    embedding matrix.
    """
    def attack_from_json(self,
                         inputs: JsonDict = None,
                         field_to_change: str = 'tokens',
                         field_to_attack: str = 'label',
                         grad_input_field: str = 'grad_input_1',
                         ignore_tokens: List[str] = DEFAULT_IGNORE_TOKENS,
                         forbidden_tokens: List[str] = DEFAULT_IGNORE_TOKENS) -> JsonDict:
        if self.token_embedding is None:
            self.initialize()

        raw_instance = self.predictor.json_to_labeled_instances(inputs)[0]
        raw_text_field: TextField = raw_instance[field_to_change]  # type: ignore
        raw_tokens = deepcopy(raw_text_field.tokens)

        att_instance = deepcopy(raw_instance)

        # Gets a list of the fields that we want to check to see if they change.
        att_text_field: TextField = att_instance[field_to_change]  # type: ignore
        att_tokens = att_text_field.tokens
        grads, outputs = self.predictor.get_gradients([att_instance])

        # ignore any token that is in the ignore_tokens list by setting the token to already flipped
        ignored_positions: List[int] = []
        for index, token in enumerate(att_tokens):
            if token.text in ignore_tokens:
                ignored_positions.append(index)
                
        successful=False
        while True:
            # Compute L2 norm of all grads.
            grad = grads[grad_input_field]
            grads_magnitude = [g.dot(g) for g in grad]

            # only flip a token once
            for index in ignored_positions:
                grads_magnitude[index] = -1

            # we flip the token with highest gradient norm
            token_sid = numpy.argmax(grads_magnitude)
            # when we have already flipped all the tokens once
            if grads_magnitude[token_sid] == -1:
                break
            ignored_positions.append(token_sid)

            # Get new token using taylor approximation
            input_tokens = att_text_field._indexed_tokens["tokens"]
            token_vid = input_tokens[token_sid]
            new_token_vid = _first_order_taylor(grad[token_sid],
                                                self.token_embedding.weight,  # type: ignore
                                                token_vid)
            # flip token
            new_token = Token(self.vocab._index_to_token["tokens"][new_token_vid])  # type: ignore
            att_text_field.tokens[token_sid] = new_token
            att_instance.indexed = False

            # Get model predictions on current_instance, and then label the instances
            grads, outputs = self.predictor.get_gradients([att_instance])  # predictions
            for key, output in outputs.items():
                outputs[key] = cast_list(outputs[key])

            # add labels to current_instances
            current_instance_labeled = self.predictor.predictions_to_labeled_instances(att_instance,
                                                                                        outputs)[0]
            # if the prediction has changed, then stop
            if current_instance_labeled[field_to_attack] != raw_instance[field_to_attack]:
                successful = True
                break

        return sanitize({"att": att_tokens,
                         "raw": raw_tokens,
                         "outputs": outputs,
                         "success": 1 if successful else 0})


def _first_order_taylor(grad: numpy.ndarray,
                        embedding_matrix: torch.nn.parameter.Parameter,
                        token_idx: int) -> int:
    """
    The below code is based on
    https://github.com/pmichel31415/translate/blob/paul/pytorch_translate/
    research/adversarial/adversaries/brute_force_adversary.py

    Replaces the current token_idx with another token_idx to increase the loss. In particular, this
    function uses the grad, alongside the embedding_matrix to select the token that maximizes the
    first-order taylor approximation of the loss.
    """
    embedding_matrix = embedding_matrix.cpu()
    word_embeds = torch.nn.functional.embedding(torch.LongTensor([token_idx]),
                                                embedding_matrix)
    word_embeds = word_embeds.detach().unsqueeze(0)
    grad = grad.unsqueeze(0).unsqueeze(0)
    # solves equation (3) here https://arxiv.org/abs/1903.06620
    new_embed_dot_grad = torch.einsum("bij,kj->bik", (grad, embedding_matrix))
    prev_embed_dot_grad = torch.einsum("bij,bij->bi", (grad, word_embeds)).unsqueeze(-1)
    neg_dir_dot_grad = -1 * (prev_embed_dot_grad - new_embed_dot_grad)
    _, best_at_each_step = neg_dir_dot_grad.max(2)
    return best_at_each_step[0].data[0].detach().cpu().item()  # return the best candidate
