# pylint: disable=protected-access
from copy import deepcopy
from typing import List
from functools import lru_cache

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
from allennlpx.interpret.attackers.policies import (CandidatePolicy,
                                                    EmbeddingPolicy,
                                                    UnconstrainedPolicy,
                                                    SpecifiedPolicy,
                                                    SynonymPolicy)


class HotFlip(Attacker):
    def __init__(self, 
             predictor, 
             *,
             policy: CandidatePolicy= None,
             **kwargs,
            ):
        super().__init__(predictor, **kwargs)
        self.policy = policy
        
    @lru_cache(maxsize=None)
    def special_mask(self, word):
        if isinstance(self.policy, EmbeddingPolicy):
            good = self.neariest_neighbours(word, 
                                            self.policy.measure, 
                                            self.policy.topk, 
                                            self.policy.rho)
        elif isinstance(self.policy, SynonymPolicy):
            good = self.synom_searcher.search(word)
        elif isinstance(self.policy, SpecifiedPolicy):
            good = self.policy.words
        all_words = self.vocab.get_index_to_token_vocabulary().values()
        mask = torch.zeros([len(all_words)])
        if len(good) > 0:
            idx = [self.vocab.get_token_index(ele) for ele in good]
            mask.scatter_(0, torch.tensor(idx), 1)
        mask = ~mask.bool().to(self.model_device)
        mask &= self.general_mask()
        return mask
    
    @lru_cache(maxsize=None)
    def general_mask(self):
        all_words = self.vocab.get_index_to_token_vocabulary().values()
        idx = [self.vocab.get_token_index(ele) for ele in self.forbidden_tokens]
        mask = torch.zeros([len(all_words)]).scatter_(0, torch.tensor(idx), 1)
        mask = mask.bool().to(self.model_device)
        return mask
        
    
    def attack_from_json(self,
                         inputs: JsonDict = None,
                         field_to_change: str = 'tokens',
                         field_to_attack: str = 'label',
                         grad_input_field: str = 'grad_input_1') -> JsonDict:
        # Not use spacy to tokenize it since we use the tokenizer provided 
        # by the model.
        raw_instance = self.predictor.json_to_labeled_instances(inputs)[0]
        raw_tokens = list(map(lambda x: x.text, raw_instance[field_to_change].tokens))
        max_change_num = self.max_change_num(len(raw_tokens))

        adv_instance = deepcopy(raw_instance)

        # Gets a list of the fields that we want to check to see if they change.
        adv_text_field: TextField = adv_instance[field_to_change]  # type: ignore
        grads, outputs = self.predictor.get_gradients([adv_instance])

        # ignore any token that is in the ignore_tokens list by setting the token to already flipped
        ignored_positions: List[int] = []
        for index, token in enumerate(adv_text_field.tokens):
            if token.text in self.ignore_tokens:
                ignored_positions.append(index)
                
        successful=False
        change_num = 0
        while change_num < max_change_num:
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
            input_token_ids = adv_text_field._indexed_tokens["tokens"]["tokens"]
            token_vid = input_token_ids[token_sid]
            
            cur_word = self.vocab._index_to_token['tokens'][token_vid]
            cur_embed = self.token_embedding[token_vid]
            cur_grad = torch.from_numpy(grad[token_sid]).to(self.model_device)
            new_embed_dot_grad = self.token_embedding @ cur_grad
            cur_embed_dot_grad = cur_embed @ cur_grad
            direction = new_embed_dot_grad - cur_embed_dot_grad
            if isinstance(self.policy, UnconstrainedPolicy):
                mask = self.general_mask()
            else:
                mask = self.special_mask(cur_word)
            direction.masked_fill_(mask, -19260817.)
            new_token_vid = torch.argmax(direction).item()
            
            # flip token
            new_token = Token(self.vocab._index_to_token["tokens"][new_token_vid])  # type: ignore
            adv_text_field.tokens[token_sid] = new_token
            adv_instance.indexed = False
            change_num += 1

            # Get model predictions on current_instance, and then label the instances
            grads, _ = self.predictor.get_gradients([adv_instance])  # predictions

            # add labels to current_instances
            result = self.predictor._model.forward_on_instance(adv_instance)
            current_instance_labeled = self.predictor.predictions_to_labeled_instances(
                adv_instance, result)[0]
            
            if current_instance_labeled[field_to_attack] != raw_instance[field_to_attack]:
                successful = True
                break
        
        adv_tokens = list(map(lambda x: x.text, adv_text_field.tokens))
        return sanitize({"adv": adv_tokens,
                         "raw": raw_tokens,
                         "outputs": outputs,
                         "success": 1 if successful else 0})