import numpy as np
from tabulate import tabulate
from collections import Counter
from allennlp.data import Vocabulary


def analyze_frequency(vocab: Vocabulary):
    counter = Counter(dict(vocab._retained_counter['tokens']))
    num = len(counter)
    freqs = counter.most_common(num)

    for i in range(1, 10):
        before = num // 10 * i
        print(before, '\t', freqs[before][1])
    print()


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
