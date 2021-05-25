import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


# def raw_iterator(files, sparse_feature_info):
#     for file in files:
#         with open(file, 'r') as f:
#             _ = f.readline()
#             lines = f.readlines()
#         for line in lines:
#             try:
#                 row = np.array(line.strip().split('\t'))
#                 for feat, v in sparse_feature_info.items():
#                     idxs = list(range(*v['index']))
#                     vocab = v['vocab']
#                     row[idxs] = [vocab[row[i]] if row[i] in vocab else vocab["__OOV__"]
#                                  for i in idxs]
#                 row = row.astype("float32")
#                 yield row
#             except:
#                 continue

def raw_iterator(files, sparse_feature_info):
    for file in files:
        f = open(file, 'r')
        _ = f.readline() # column name
        while True:
            line = f.readline()
            if not line: break
            try:
                row = np.array(line.strip().split('\t'))
                for feat, v in sparse_feature_info.items():
                    idxs = list(range(*v['index']))
                    vocab = v['vocab']
                    row[idxs] = [vocab[row[i]] if row[i] in vocab else vocab["__OOV__"]
                                 for i in idxs]
                row = row.astype("float32")
                yield row
            except:
                continue


class RawDataset(IterableDataset):

    def __init__(self, files, sparse_feature_info, label_idx):

        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            rank, nums_workers = worker_info.id, worker_info.nums_workers
        else:
            rank, nums_workers = 0, 1
        worker_files = [files[i] for i in range(rank, len(files), nums_workers)]
        self.iterator = raw_iterator(worker_files, sparse_feature_info)
        self.label_idx = label_idx

    def __iter__(self):
        for x in self.iterator:
            y = np.array([1.]) if x[self.label_idx] > 0 else np.array([0.])
            y = y.astype("float32")
            yield {"features": x, "label": y}