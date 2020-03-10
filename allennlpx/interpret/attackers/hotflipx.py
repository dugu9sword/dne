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

from allennlpx.interpret.attackers.attacker import (DEFAULT_IGNORE_TOKENS,
                                                    Attacker)
from luna import cast_list


class HotFlipX(Attacker):
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
                         grad_input_field: str = 'grad_input_1') -> JsonDict:

        raw_instance = self.predictor.json_to_labeled_instances(inputs)[0]
        raw_text_field: TextField = raw_instance[field_to_change]  # type: ignore
        raw_tokens = list(map(lambda x: x.text, raw_instance[field_to_change].tokens))

        adv_instance = deepcopy(raw_instance)

        # Gets a list of the fields that we want to check to see if they change.
        adv_tokens = list(map(lambda x: x.text, adv_instance[field_to_change].tokens))
        grads, outputs = self.predictor.get_gradients([adv_instance])

        # ignore any token that is in the ignore_tokens list by setting the token to already flipped
        ignored_positions: List[int] = []
        for index, token in enumerate(adv_tokens):
            if token in self.ignore_tokens:
                ignored_positions.append(index)
                
        successful=False
        while True:
            # Compute L2 norm of all grads.
            grad = grads[grad_input_field][0]  # first dim is batch
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
            input_tokens = adv_text_field._indexed_tokens["tokens"]
            token_vid = input_tokens[token_sid]
            new_token_vid = _first_order_taylor(
                torch.from_numpy(grad[token_sid]).to(self.model_device),
                self.token_embedding,  # type: ignore
                token_vid
            )
            # flip token
            new_token = Token(self.vocab._index_to_token["tokens"][new_token_vid])  # type: ignore
            adv_text_field.tokens[token_sid] = new_token
            adv_instance.indexed = False

            # Get model predictions on current_instance, and then label the instances
            grads, outputs = self.predictor.get_gradients([adv_instance])  # predictions
            for key, output in outputs.items():
                outputs[key] = cast_list(outputs[key])

            # add labels to current_instances
            current_instance_labeled = self.predictor.predictions_to_labeled_instances(adv_instance,
                                                                                        outputs)[0]
            
#             print(allenutil.as_sentence(current_instance_labeled))
            
            # if the prediction has changed, then stop
            if current_instance_labeled[field_to_attack] != raw_instance[field_to_attack]:
#                 print("succ")
                successful = True
                break
        
        adv_tokens = list(map(lambda x: x.text, adv_instance[field_to_change].tokens))

        return sanitize({"adv": adv_tokens,
                         "raw": raw_tokens,
                         "outputs": outputs,
                         "success": 1 if successful else 0})

# @pysnooper.snoop()
def _first_order_taylor(grad,
                        embedding_matrix,
                        token_idx: int) -> int:
    """
    The below code is based on
    https://github.com/pmichel31415/translate/blob/paul/pytorch_translate/
    research/adversarial/adversaries/brute_force_adversary.py

    Replaces the current token_idx with another token_idx to increase the loss. In particular, this
    function uses the grad, alongside the embedding_matrix to select the token that maximizes the
    first-order taylor approximation of the loss.
    """
    word_embeds = torch.nn.functional.embedding(torch.LongTensor([token_idx]).to(embedding_matrix.device),
                                                embedding_matrix)
    word_embeds = word_embeds.detach().unsqueeze(0)
    grad = grad.unsqueeze(0).unsqueeze(0)
    # solves equation (3) here https://arxiv.org/abs/1903.06620
    new_embed_dot_grad = torch.einsum("bij,kj->bik", (grad, embedding_matrix))
    prev_embed_dot_grad = torch.einsum("bij,bij->bi", (grad, word_embeds)).unsqueeze(-1)
    neg_dir_dot_grad = -1 * (prev_embed_dot_grad - new_embed_dot_grad)
    _, best_at_each_step = neg_dir_dot_grad.max(2)
    return best_at_each_step[0].data[0].detach().cpu().item()  # return the best candidate
