import os
import torch
from tqdm import tqdm
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.common.util import prepare_environment
from allennlp.common.params import Params
from allennlpx import allenutil
from allennlpx.training.util import evaluate
from allennlpx.predictors.text_classifier import TextClassifierPredictor
from allennlpx.interpret.attackers.bruteforce import BruteForce
from allennlpx.interpret.attackers.attacker import DEFAULT_IGNORE_TOKENS
from luna import (auto_create, flt2str, log, log_config)
from sst.model import LstmClassifier
from sst.program_args import ProgramArgs
from sst.utils import (get_token_indexer, get_data_reader, get_data, get_iterator, get_best_checkpoint)

# for logging
log_config("log", "cf")

if __name__ == '__main__':
    config = ProgramArgs.parse(True)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_device
    # random seed
    prepare_environment(Params({}))
    # get token indexer
    token_indexer = get_token_indexer(config.pretrain)

    # get data reader
    sub_reader = get_data_reader(token_indexer, subtree=True)
    reader = get_data_reader(token_indexer, subtree=False)
    
    # get dataset 
    sub_train_data, train_data, \
    dev_data, test_data = auto_create(f"{config.cache_prefix}_data",
                                      lambda: get_data(sub_reader, reader, config.train_data_file, config.dev_data_file, config.test_data_file),
                                      True,
                                      config.cache_path)

    # get vocab
    vocab = auto_create(f"{config.cache_prefix}_vocab",
                        lambda: Vocabulary.from_instances(sub_train_data + dev_data + test_data),
                        True,
                        config.cache_path)

    # get model
    model = LstmClassifier(vocab, config).cuda()

    # get iterator
    iterator = get_iterator(config.batch_size)
    iterator.index_with(vocab)

    best_checkpoint_path = get_best_checkpoint(config.saved_path)
    print('load model from {}'.format(best_checkpoint_path))
    model.load_state_dict(torch.load(get_best_checkpoint(config.saved_path)))

    evaluate(model, test_data, iterator, cuda_device=0, batch_weight_key=None)

    for module in model.modules():
        if isinstance(module, TextFieldEmbedder):
            for embed in module._token_embedders.keys():
                module._token_embedders[embed].weight.requires_grad = True

    pos_words = [line.rstrip('\n') for line in open("../sentiment-words/positive-words.txt")]
    neg_words = [line.rstrip('\n') for line in open("../sentiment-words/negative-words.txt")]
    not_words = [line.rstrip('\n') for line in open("../sentiment-words/negation-words.txt")]
    forbidden_words = pos_words + neg_words + not_words + DEFAULT_IGNORE_TOKENS
    
    predictor = TextClassifierPredictor(model.cpu(), reader)
    # attacker = HotFlip(predictor)
    attacker = BruteForce(predictor)
    attacker.initialize()

    data_to_attack = test_data
    total_num = len(data_to_attack)
    # total_num = 20
    succ_num = 0
    src_words = []
    tgt_words = []
    for i in tqdm(range(total_num)):
        raw_text = allenutil.as_sentence(data_to_attack[i])

        result = attacker.attack_from_json({"sentence": raw_text},
                                           ignore_tokens=forbidden_words,
                                           forbidden_tokens=forbidden_words,
                                           max_change_num=5,
                                           search_num=256)
        
        att_text = allenutil.as_sentence(result['att'])

        if result["success"] == 1:
            succ_num += 1
            log("[raw]", raw_text)
            log("\t", flt2str(predictor.predict(raw_text)['probs']))
            log("[att]", att_text)
            log('\t', flt2str(predictor.predict(att_text)['probs']))
            log()

        raw_tokens = result['raw']
        att_tokens = result['att']
        for i in range(len(raw_tokens)):
            if raw_tokens[i] != att_tokens[i]:
                src_words.append(raw_tokens[i])
                tgt_words.append(att_tokens[i])

    print(f'Succ rate {succ_num / total_num * 100}')