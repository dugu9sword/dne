from pathlib import Path

import faiss
import h5py
import numpy as np
import torch


class H5pyCollector:
    def __init__(self, name, dim):
        super().__init__()

        self.name = name
        self.dim = dim

        if Path(self.name).exists():
            Path(self.name).unlink()
        self._f = h5py.File(self.name, "w")
        self._dset = self._f.create_dataset("data", shape=(0, self.dim),
                                            maxshape=(None, self.dim))  # type: h5py.Dataset

    def close(self):
        self._f.close()

    def collect(self, tensors):
        if isinstance(tensors, torch.Tensor):
            tensors = tensors.cpu().numpy()
        append_size = tensors.shape[0]
        self._dset.resize(self._dset.shape[0] + append_size, axis=0)
        self._dset[-append_size:] = tensors


def build_faiss_index(data_or_name, gpu=True, ann=False, ann_center=10, ann_nprobe=1):
    if isinstance(data_or_name, str):
        data = h5py.File(data_or_name, "r")['data'][:]
    else:
        data = data_or_name
    print(f'BUILDING INDEX WITH DATA {data.shape}...')

    dim = data.shape[1]
    if ann:
        index = faiss.IndexFlatL2(dim)
        # index = faiss.IndexIVFFlat(index, dim, ann_center)
        index = faiss.IndexIVFPQ(index, dim, 512, 8, 8)
        if gpu:
            res = faiss.StandardGpuResources()  # use a single GPU
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.train(data)
        index.add(data)
        index.nprobe = ann_nprobe
    else:
        index = faiss.IndexFlatL2(dim)
        if gpu:
            res = faiss.StandardGpuResources()  # use a single GPU
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(data)
    return index
