import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


def raw_iterator(files, encoder_items, ret_idxs, label_info):
    click_idx, show_idx = label_info['now_click_cnt']['index'][0], label_info['now_show_cnt']['index'][0]
    for file in files:
        f = open(file, 'r', encoding="utf-8")
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

def click_rate(click, show):
    if show > 0:
        label = click / show if show > click else 1
    else:
        label = 0
    return label

def lv_rate(play, show, lv):
    if play + show <= 0:
        raise
    if play > 0:
        label = lv / play if play > lv else 1
    else:
        label = 0
    return label



class TrainDataset(IterableDataset):

    def __init__(self, files, encoder_items, ret_idxs, label_info, cycle=False):
        self.files = files
        self.encoder_items = encoder_items
        self.ret_idxs = ret_idxs
        self.iterator = None
        self.label_info = label_info
        self.cycle = cycle

    @staticmethod
    def _iterator(files, encoder_items, ret_idxs, label_info):
        click_idx = label_info['now_click_cnt']['index'][0]
        show_idx = label_info['now_show_cnt']['index'][0]
        play_idx = label_info['now_play_cnt']['index'][0]
        lv_idx = label_info['now_long_view_cnt']['index'][0]
        for file in files:
            f = open(file, 'r', encoding="utf-8")
            while True:
                line = f.readline()
                if not line:
                    break
                try:
                    row = np.array(line.strip().split('\t'))
                    for idxs, vocab in encoder_items:
                        row[idxs] = [vocab[row[i]] if row[i] in vocab else vocab["__OOV__"]
                                     for i in idxs]
                    feat = row[ret_idxs].astype("float32")
                    if feat[0] <= 2: continue
                    click, show, play, lv = row[[click_idx, show_idx, play_idx, lv_idx]].astype("float32")
                    label = np.array([click_rate(click, show), lv_rate(play, show, lv)]).astype("float32")
                    yield feat, label
                except:
                    print("error line:", line)
                    continue

    def reset(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            rank, num_workers = worker_info.id, worker_info.num_workers
        else:
            rank, num_workers = 0, 1
        worker_files = [self.files[i] for i in range(rank, len(self.files), num_workers)]
        #  print(worker_files)
        self.iterator = self._iterator(worker_files, self.encoder_items, self.ret_idxs, self.label_info)

    def __iter__(self):
        if self.iterator is None:
            self.reset()
        for x, y in self.iterator:
            yield {"features": x, "label": y}


class EvalDataset(IterableDataset):

    def __init__(self, files, encoder_items, ret_idxs, label_info, eval_info, cycle=False):
        self.files = files
        self.encoder_items = encoder_items
        self.ret_idxs = ret_idxs
        self.eval_info = eval_info
        self.label_info = label_info
        self.cycle = cycle
        self.iterator = None

    @staticmethod
    def _iterator(files, encoder_items, ret_idxs, label_info, eval_info):
        indicator_idx_dict = {k: v['index'][0] for k, v in eval_info.items()}
        now_click_idx = label_info['now_click_cnt']['index'][0]
        now_show_idx = label_info['now_show_cnt']['index'][0]
        now_play_idx = label_info['now_play_cnt']['index'][0]
        now_lv_idx = label_info['now_long_view_cnt']['index'][0]
        for file in files:
            f = open(file, 'r', encoding="utf-8")
            while True:
                line = f.readline()
                if not line:
                    break
                # try:
                row = np.array(line.strip().split('\t'))
                for idxs, vocab in encoder_items:
                    row[idxs] = [vocab[row[i]] if row[i] in vocab else vocab["__OOV__"]
                                 for i in idxs]
                feat = row[ret_idxs].astype("float32")
                if feat[0] <= 2: continue
                # click, show, play, lv = row[[click_idx, show_idx, play_idx, lv_idx]].astype("float32")
                indicator = {k: row[v].astype("float32") for k, v in indicator_idx_dict.items()}
                # indicator = {
                #     'click_cnt': click,
                #     'show_cnt': show,
                #     'play_cnt': play,
                #     'long_view_cnt': lv
                # }
                click, show, play, lv = row[[now_click_idx, now_show_idx, now_play_idx, now_lv_idx]].astype("float32")
                label = np.array([click_rate(click, show), lv_rate(play, show, lv)]).astype("float32")
                yield feat, indicator, label
                # except:
                #     print("error line:", line)
                #     continue


    def reset(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            rank, num_workers = worker_info.id, worker_info.num_workers
        else:
            rank, num_workers = 0, 1
        worker_files = [self.files[i] for i in range(rank, len(self.files), num_workers)]
        self.iterator = self._iterator(worker_files,
                                       self.encoder_items,
                                       self.ret_idxs,
                                       self.label_info,
                                       self.eval_info)

    def __iter__(self):
        if self.iterator is None:
            self.reset()
        for x, indicator, y in self.iterator:
            yield {"features": x, "indicator": indicator, "label": y}
