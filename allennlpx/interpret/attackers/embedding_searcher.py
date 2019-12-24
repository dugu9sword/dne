import torch
from tabulate import tabulate
import numpy as np
from typing import Union


class EmbeddingSearcher:
    def __init__(
        self,
        embed: torch.Tensor,
        word2idx: callable,
        idx2word: callable,
    ):
        self.embed = embed
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.faiss_index = None

    def use_faiss_backend(
        self,
        gpu=False,
        ann=False,
        ann_center=10,
        ann_nprob=1,
    ):
        # The method is just a running example of the faiss library which supports
        #   - batch queries
        #   - approximate nearest neighbours
        #   - several distances
        # However, for most of the time, the vanilla pytorch version is enough.
        #
        # Some details:
        # 1. GPU is not always better than CPU, see:
        #       https://github.com/facebookresearch/faiss/wiki/Comparing-GPU-vs-CPU
        #    From my preliminary experiment (vocab size 20000+, query size 1):
        #       when ann=True, you may set gpu=False
        #       when ann=False, you may set gpu=True
        # 2. There are some overheads in the code. The faiss lib is based on numpy,
        #    while in this code there are some conversions between:
        #       [tensor(GPU) <->] tensor <-> numpy [<-> GPU]
        #
        import faiss
        data = self.embed.cpu().numpy()
        dim = self.embed.size(1)
        index = faiss.IndexFlatL2(dim)
        if ann:
            fast_index = faiss.IndexIVFFlat(index, dim, ann_center)
            fast_index.train(data)
            fast_index.nprobe = ann_nprob
            index = fast_index
        index.add(data)
        if gpu:
            res = faiss.StandardGpuResources()  # use a single GPU
            index = faiss.index_cpu_to_gpu(res, 0, index)
        self.faiss_index = index

    def show_embedding_info(self):
        print('*** Statistics of parameters and 2-norm ***')
        show_mean_std(self.embed, 'Param')
        show_mean_std(torch.norm(self.embed, p=2, dim=1), 'Norm')

        print('*** Statistics of distances in a N-nearest neighbourhood ***')
        nbr_num = [5, 10, 20, 50, 100, 200, 500, 10000, 20000]
        dists = {nbr: [] for nbr in nbr_num}
        for ele in cast_list(torch.randint(self.embed.size(0), (50, ))):
            if self.embed[ele].sum() == 0.:
                continue
            idxs, vals = self.find_neighbours(ele, -1, 'euc', False)
            for nbr in nbr_num:
                dists[nbr].append(vals[1:nbr + 1])
        table = []
        for nbr in nbr_num:
            dists[nbr] = torch.cat(dists[nbr])
            table.append([nbr, dists[nbr].mean().item(), dists[nbr].std().item()])
        print(tabulate(table, headers=['N', 'mean', 'std'], floatfmt='.2f'))
        # exit()

        print('*** Statistics of distances in a N-nearest neighbourhood ***\n'
              '    when randomly moving by different step sizes')
        mve_nom = [1, 2, 5, 10, 20, 50]
        nbr_num = [5, 10, 20, 50, 100, 500]
        dists = {mve: {nbr: [] for nbr in nbr_num} for mve in mve_nom}
        cover = {mve: {nbr: [] for nbr in nbr_num} for mve in mve_nom}
        for ele in cast_list(torch.randint(self.embed.size(0), (50, ))):
            if self.embed[ele].sum() == 0.:
                continue
            vect = torch.rand_like(self.embed[ele])
            vect = vect / torch.norm(vect)
            ridxs, rvals = self.find_neighbours(self.embed[ele], -1, 'euc', False)
            for mve in mve_nom:
                idxs, vals = self.find_neighbours(self.embed[ele] + vect * mve, 500, 'euc', False)
                for nbr in nbr_num:
                    dists[mve][nbr].append(vals[1:nbr + 1])
                    cover[mve][nbr].append(compare_idxes(idxs[1:nbr + 1], ridxs[1:nbr + 1]))
        table = []
        for mve in mve_nom:
            row = [mve]
            for nbr in nbr_num:
                dist = torch.cat(dists[mve][nbr])
                row.append(dist.mean().item())
                row.append("{:.1f}%".format(np.mean(cover[mve][nbr]) / nbr * 100))
            table.append(row)

        print(
            tabulate(table,
                     headers=[
                         'Step', 'D-5', 'I-5', 'D-10', 'I-10', 'D-20', 'I-20', 'D-50', 'I-50',
                         'D-100', 'I-100'
                     ],
                     floatfmt='.2f'))

    @torch.no_grad()
    def find_neighbours(self,
                        element: Union[int, str, torch.Tensor],
                        measure='euc',
                        topk=None,
                        rho=None,
                        verbose=False):
        # checking args
        assert (topk is None) ^ (rho is None), "You must set one of topk/rho to be None"
        assert measure in ['euc', 'cos']
        if rho is not None:
            assert (measure == 'euc' and rho > 0) or (
                measure == 'cos' and 0 < rho < 1), "threshold for euc distance must be larger than 0, for cos distance must be between 0 and 1"

        measure_fn = cos_dist if measure == 'cos' else euc_dist
        if isinstance(element, int):
            idx = element
            query_vector = self.embed[idx]
        elif isinstance(element, str):
            idx = self.word2idx(element)
            query_vector = self.embed[idx]
        elif isinstance(element, torch.Tensor):
            query_vector = element
        else:
            raise TypeError('You passed a {}, int/str/torch.Tensor required'.format(type(element)))

        if topk is None:
            _topk = self.embed.size(0)
        else:
            _topk = topk

        if self.faiss_index is None:
            dists = measure_fn(query_vector, self.embed)
            tk_vals, tk_idxs = torch.topk(dists, _topk, largest=False)
        else:
            if measure == 'cos':
                raise Exception("cos still not compatible with faiss (since I am lazy)")
            D, I = self.faiss_index.search(query_vector.unsqueeze(0).cpu().numpy(), _topk)
            tk_vals = torch.tensor(D[0], device=self.embed.device)
            tk_idxs = torch.tensor(I[0], device=self.embed.device)
        if rho is not None:
            mask_idx = tk_vals < rho
            tk_vals = tk_vals[mask_idx]
            tk_idxs = tk_idxs[mask_idx]
            
        if verbose:
            table = []
            print('Neariest neighbours measured by {}, topk={}, rho={}'.format(measure, topk, rho))
            for i in tk_idxs:
                i = i.item()
                table.append([i, self.idx2word(i), dists[i].item()])
            print(tabulate(table))
        return tk_vals, tk_idxs


# torch.tensor([100]).expand(2000, 100):
#   return a view, in memory it's still [100]
# torch.tensor([100]).expand(2000, 1):
#   return a copied tensor
def cos_dist(qry, mem):
    return 1 - cos_sim(qry, mem)


def cos_sim(qry, mem):
    return torch.nn.functional.cosine_similarity(mem, qry.expand(mem.size(0), mem.size(1)), dim=1)


def euc_dist(qry, mem):
    return torch.sqrt((qry - mem).pow(2).sum(dim=1))


def compare_idxes(nbr1, nbr2):
    nbr1 = set(cast_list(nbr1))
    nbr2 = set(cast_list(nbr2))
    inter = nbr1.intersection(nbr2)
    return len(inter)


if __name__ == '__main__':
    from dpattack.libs.luna import fetch_best_ckpt_name, cast_list, show_mean_std, time_record
    from dpattack.utils.parser_helper import load_parser
    from dpattack.utils.vocab import Vocab

    vocab = torch.load("/disks/sdb/zjiehang/zhou_data/ptb/vocab")  # type: Vocab
    parser = load_parser(
        fetch_best_ckpt_name("/disks/sdb/zjiehang/zhou_data/saved_models/word_tag/lzynb"))
    # print(type(vocab))
    esglv = EmbeddingSearcher(embed=vocab.embeddings,
                              idx2word=lambda x: vocab.words[x],
                              word2idx=lambda x: vocab.word_dict[x])

    with time_record():
        esglv.use_faiss_backend(False, True, 10, 1)
        for _ in range(10):
            esglv.find_neighbours(0, 100)

    # esglv.show_embedding_info()
    # esmdl = EmbeddingSearcher(embed=parser.embed.weight,
    #                           idx2word=lambda x: vocab.words[x],
    #                           word2idx=lambda x: vocab.word_dict[x])
    # esmdl.show_embedding_info()

    # esglv.show_embedding_info()
    # es.show_embedding_info()
    #
    # embed = es.embed[es.word2idx('red')]
    # vals, idxs = es.find_neighbours(embed, 20, 'euc', True)
    #
    # embed += 10 * torch.sign(embed)
    # es.find_neighbours(embed, 20, 'euc', True)

    # for word in ['company', 'red', 'happy', 'play', 'down']:
    #     _, euc_idxes = esmdl.find_neighbours(word, 100, 'euc', verbose=True)
    # exit()
    #
    # total = 0
    # for ele in cast_list(torch.randint(len(vocab.words), (100,))):
    #     _, euc_idxes = esmdl.find_neighbours(ele, 20, 'euc', verbose=False)
    #     _, cos_idxes = esglv.find_neighbours(ele, 20, 'cos', verbose=False)
    #     print(vocab.words[ele], '\t\t', compare_idxes(euc_idxes, cos_idxes))
    #     total += compare_idxes(euc_idxes, cos_idxes)
    # print(total, '\n')
    #
    # total = 0
    # for ele in cast_list(torch.randint(len(vocab.words), (100,))):
    #     _, mdl_idxes = esmdl.find_neighbours(ele, 20, 'euc', verbose=False)
    #     _, glv_idxes = esglv.find_neighbours(ele, 20, 'euc', verbose=False)
    #     print(vocab.words[ele], '\t\t', compare_idxes(mdl_idxes, glv_idxes))
    #     total += compare_idxes(mdl_idxes, glv_idxes)
    # print(total, '\n')
