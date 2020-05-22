from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.common.util import JsonDict


def as_sentence(something, field=None):
    if isinstance(something, Instance):
        if field is None:
            field = next(iter(something.fields))
        tokens = something.fields[field].tokens
        tkstrs = list(map(str, tokens))
    elif isinstance(something, TextField):
        tokens = something.tokens
        tkstrs = list(map(str, tokens))
    elif isinstance(something, list):
        if isinstance(something[0], Token):
            tkstrs = list(map(str, something))
        elif isinstance(something[0], str):
            tkstrs = something
        else:
            raise Exception(f'the arg you passes is {type(something[0])}')
    else:
        raise Exception(f'the arg you passes is {type(something)}')
    sent = " ".join(tkstrs)
    # replace word piece
    sent = sent.replace("[CLS]", "", -1)
    sent = sent.replace("[SEP]", "", -1)
    sent = sent.replace(" ##", "", -1)
    sent = sent.strip()
    return sent


def as_json(instance: Instance):
    ret = {}
    for k, v in instance.items():
        if isinstance(v, TextField):
            ret[k] = as_sentence(v)
    return ret


def bert_instance_as_json(instance: Instance):
    ret = {}
    tokens = list(map(str, instance['sent'].tokens))
    assert tokens[0] == '[CLS]'
    sep_index = []
    for tid, token in enumerate(tokens):
        if token == '[SEP]':
            sep_index.append(tid)
    assert len(sep_index) in [1, 2]
    assert sep_index[-1] == len(tokens) - 1
    if len(sep_index) == 1:
        ret['sent'] = as_sentence(tokens[1:-1])
    else:
        ret['sent1'] = as_sentence(tokens[1:sep_index[0]])
        ret['sent2'] = as_sentence(tokens[sep_index[0] + 1: -1])
    return ret

def modified_copy(jsonDict: JsonDict, key, value):
    ret = jsonDict.copy()
    ret[key] = as_sentence(value)
    return ret
