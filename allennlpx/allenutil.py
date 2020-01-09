from allennlp.data.instance import Instance
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import Token

def as_sentence(something, field='tokens'):
    if isinstance(something, Instance):
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
    return " ".join(tkstrs)

    