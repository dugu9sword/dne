from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.common.util import JsonDict
from copy import deepcopy
from copy import copy


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


def modified_copy(jsonDict: JsonDict, key, value):
    ret = jsonDict.copy()
    ret[key] = as_sentence(value)
    return ret
