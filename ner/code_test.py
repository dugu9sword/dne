import os
from ner.args import ProgramArgs
from ner.task import NERTask
from allennlp.common.util import prepare_environment
from allennlp.common.params import Params
from ner.attacker.bruteforce import BruteForce
from ner.utils.utils import get_named_entity_vocab
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN

if __name__ == '__main__':
    config = ProgramArgs.parse(True)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_device

    # random seed
    prepare_environment(Params({}))

    task = NERTask(config)

    # named_entity_vocab = get_named_entity_vocab(task.vocab, task.train_data + task.dev_data + task.test_data)
    # named_entity_vocab[task.vocab.get_token_index(DEFAULT_PADDING_TOKEN)] = DEFAULT_PADDING_TOKEN
    # named_entity_vocab[task.vocab.get_token_index(DEFAULT_OOV_TOKEN)] = DEFAULT_OOV_TOKEN
    # # attacker = BruteForce(task.vocab, task.data_reader, search_words=config.search_word, search_num=config.search_nums,
    # #                       forbidden_dict=named_entity_vocab)

    length = {}
    attack_length = {}
    for instance in task.test_data:
        sentence_length = len(instance.fields['tokens'])
        if sentence_length in length:
            length[len(instance.fields['tokens'])] += 1
        else:
            length[len(instance.fields['tokens'])] = 1

        # named_entity_index=set()
        # for index, (token, tag) in enumerate(zip(instance.fields['tokens'], instance.fields['tags'])):
        #     if tag != 'O':
        #         if index > 0:
        #             named_entity_index.add(index - 1)
        #         named_entity_index.add(index)
        #         if index < sentence_length - 1:
        #             named_entity_index.add(index + 1)

        # attack_index = [i for i in range(sentence_length) if i not in named_entity_index]    
        # attack_num = min(len(attack_index), config.search_word)
        # if sentence_length == 1:
        #     attack_num = 0
        # 
        # if attack_num in attack_length:
        #     attack_length[attack_num] += 1
        # else:
        #     attack_length[attack_num] = 1

    all = 0
    for key, value in length.items():
        if key <= 5:
            all += value
    all = 0
    for key in sorted(length.keys()):
        print(str(key) + ' ' + str(length[key]))
        all += (key * length[key])
    print(all)
    print(length)
    print(attack_length)
    print(all / len(task.test_data))
# class Test:
#     def __init__(self):
#         self.a = {}
#         self.b = {}
#         self.c = {}
# 
# class ATest(Test):        
#     def __iadd__(self, other):
#         for key, value in other.a.items():
#             if key in self.a:
#                 self.a[key] += value
#             else:
#                 self.a[key] = value
#             
#         for key, value in other.b.items():
#             if key in self.b:
#                 self.b[key] += value
#             else:
#                 self.b[key] = value
#         
#         for key, value in other.c.items():
#             if key in self.c:
#                 self.c[key] += value
#             else:
#                 self.c[key] = value
#     
#     
# a = ATest()
# b = ATest()
# b.a['key'] = 1
# b.b['key1'] = 1
# b.c['key2'] = 1
# a += b