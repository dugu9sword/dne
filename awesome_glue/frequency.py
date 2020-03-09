
import numpy as np
from allennlp.data import Vocabulary
from tabulate import tabulate


def frequency_analysis(counter, src_words, tgt_words):
    src_freqs = [counter[word] for word in src_words]
    tgt_freqs = [counter[word] for word in tgt_words]

    # num = len(counter)

    # for s, t in zip(fsrcs, f_tgts):
    #     if s in counter.most_common(5000)

    print(np.median(src_freqs), np.median(tgt_freqs))

    table = [['SEG', 'HH', 'HL', 'LH', 'LL']]
    for seg in [2000, 4000, 6000, 8000]:
        high = next(zip(*counter.most_common(seg)))
        # low = counter.subtract(high)

        # print(high)

        hh = 0
        hl = 0
        lh = 0
        ll = 0
        for s, t in zip(src_words, tgt_words):
            if s in high:
                if t in high:
                    hh += 1
                else:
                    hl += 1
            else:
                if t in high:
                    lh += 1
                else:
                    ll += 1
        table.append([seg, hh, hl, lh, ll])
    print(tabulate(table))
