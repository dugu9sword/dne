import os

import torch


def back_translation():
    os.environ["TORCH_HOME"] = '/disks/sdb/torch_home'

    # List available models
    torch.hub.list('pytorch/fairseq', force_reload=False)  # [..., 'transformer.wmt18.en-de', ... ]

    # # Load the WMT'18 En-De ensemble
    en2de = torch.hub.load(
        'pytorch/fairseq', 'transformer.wmt19.en-de',
        checkpoint_file='model1.pt',
        tokenizer='moses', bpe='fastbpe', verbose=True).cuda()

    de2en = torch.hub.load(
        'pytorch/fairseq', 'transformer.wmt19.de-en',
        checkpoint_file='model1.pt',
        tokenizer='moses', bpe='fastbpe', verbose=True).cuda()
    
    return en2de, de2en
