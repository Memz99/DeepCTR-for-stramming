import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


# def raw_iterator(files, sparse_feature_info):
#     loop_items = [(list(range(*v['index'])), v['vocab']) for k, v in sparse_feature_info.items()]
#     ret_idxs = np.array([idxs for idxs, v in loop_items]).flatten()
#     for file in files:
#         with open(file, 'r') as f:
#             _ = f.readline()
#             lines = f.readlines()
#         for line in lines:
#             try:
#                 row = np.array(line.strip().split('\t'))
#                 for idxs, vocab in loop_items:
#                     row[idxs] = [vocab[row[i]] if row[i] in vocab else vocab["__OOV__"]
#                                  for i in idxs]
#                 row = row[ret_idxs].astype("float32")
#                 yield row
#             except:
#                 continue

def raw_iterator(files, encoder_items, ret_idxs, label_info):
    click_idx, show_idx = label_info['now_click_cnt']['index'][0], label_info['now_show_cnt']['index'][0]
    for file in files:
        f = open(file, 'r', encoding="utf-8")
#         _ = f.readline() # column name
        while True:
            line = f.readline()
            if not line:
                break
            try:
                row = np.array(line.strip().split('\t'))
                for idxs, vocab in encoder_items:
                    row[idxs] = [vocab[row[i]] if row[i] in vocab else vocab["__OOV__"]
                                 for i in idxs]
                feat = row[ret_idxs]
                click = float(row[click_idx])
                show = float(row[show_idx])
                if show > 0:
                    label = click / show if show > click else 1
                else:
                    label = 0
                yield feat.astype("float32"), np.array(label).astype("float32")
            except:
                print("error line:", line)
                continue


class RawDataset(IterableDataset):

    def __init__(self, files, encoder_items, ret_idxs, label_info, cycle=False):
        self.files = files
        self.encoder_items = encoder_items
        self.ret_idxs = ret_idxs
        self.iterator = None
        self.label_info = label_info
        self.cycle = cycle

    def reset(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            rank, num_workers = worker_info.id, worker_info.num_workers
        else:
            rank, num_workers = 0, 1
        worker_files = [self.files[i] for i in range(rank, len(self.files), num_workers)]
        self.iterator = raw_iterator(worker_files, self.encoder_items, self.ret_idxs, self.label_info)

    def __iter__(self):
        if self.iterator is None:
            self.reset()
        for x, y in self.iterator:
            yield {"features": x, "label": y}
