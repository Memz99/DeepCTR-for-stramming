import os
import numpy as np
import pandas as pd

from collections import defaultdict


class Auc(object):
    def __init__(self, num_buckets):
        self._num_buckets = num_buckets
        self._table = np.zeros(shape=[2, self._num_buckets])

    def Reset(self):
        self._table = np.zeros(shape=[2, self._num_buckets])

    def Update(self, labels: np.ndarray, predicts: np.ndarray):
        """
        :param labels: 1-D ndarray
        :param predicts: 1-D ndarray
        :return: None
        """
        labels = labels.astype(np.int)
        predicts = self._num_buckets * predicts

        buckets = np.round(predicts).astype(np.int)
        buckets = np.where(buckets < self._num_buckets,
                           buckets, self._num_buckets - 1)

        for i in range(len(labels)):
            self._table[labels[i], buckets[i]] += 1

    def Compute(self):
        tn = 0
        tp = 0
        area = 0
        for i in range(self._num_buckets):
            new_tn = tn + self._table[0, i]
            new_tp = tp + self._table[1, i]
            # self._table[1, i] * tn + self._table[1, i]*self._table[0, i] / 2
            area += (new_tp - tp) * (tn + new_tn) / 2
            tn = new_tn
            tp = new_tp
        if tp < 1e-3 or tn < 1e-3:
            return -0.5  # 样本全正例，或全负例
        return area / (tn * tp)

class InteralLoss():
    def __init__(self, interals, name, scale=1):
        self.e = {interal: 0. for interal in interals}
        self.name = name
        self.scale = scale

    def update(self, interal, pred, gt, n1, n):
        loss = sum(abs(pred - gt)) * self.scale
        e0 = self.e[interal]
        self.e[interal] = e0 + (loss - n1 * e0) / n

    def log(self, interal):
        return f"{self.name}: {round(self.e[interal], 3):>5.4}"

class InteralMAE():
    def __init__(self, feature_columns, l=300, r=99999999, interal_nums=100, save_path = ""):
        points = np.unique((2 ** np.linspace(np.log2(1+l), np.log2(1+r), interal_nums) - 1).astype(int))
        self.interals = list(zip(points[:-1], points[1:]))

        self.vidx = defaultdict(int)
        for fc in feature_columns:
            if fc.name == "pv": self.vidx['pv'] = fc.index[0]
            if fc.name == "show_cnt_30d": self.vidx['show_cnt'] = fc.index[0]
            if fc.name == "click_cnt_30d": self.vidx['click_cnt'] = fc.index[0]
            if fc.name == "long_view_cnt_30d": self.vidx['long_view_cnt'] = fc.index[0]
            if fc.name == "play_cnt_30d": self.vidx['play_cnt'] = fc.index[0]

        self.n = {interal: 0. for interal in self.interals}

        np.seterr(divide='ignore', invalid='ignore')
        self.e = [
            InteralLoss(self.interals, "MODEL_CTR_E", scale=1),
            InteralLoss(self.interals, "EMP_CTR_E", scale=1),
            InteralLoss(self.interals, "GT_CTR", scale=1),
            InteralLoss(self.interals, "MODEL_LVTR_E", scale=1),
            InteralLoss(self.interals, "EMP_LVTR_E", scale=1),
            InteralLoss(self.interals, "GT_LVTR", scale=1)
            ]

        self.save_path = save_path
        self.ofp = open(os.path.join(save_path, "prediction"), 'w')
        self.ofp.write("\t".join(
            ["pv", "show_cnt", "model_ctr", "emp_ctr", "gt_ctr", "model_lvtr", "emp_lvtr", "gt_lvtr"]) + "\n")

    def update(self, inputs, pred, y):
        v = {key: inputs[:, idx] for key, idx in self.vidx.items()}
        emp = np.stack([v['click_cnt'] / v['show_cnt'], v['long_view_cnt'] / v['play_cnt']], axis=1)
        emp[emp == np.inf] = 0
        for interal in self.interals:
            l, r = interal
            lind = np.logical_and(v['show_cnt'] >= l, v['show_cnt'] < r)
            n1 = sum(lind)
            if n1 > 0:
                self.n[interal] += n1
                _pred, _emp, _gt = pred[lind], emp[lind], y[lind]
                for e in self.e:
                    if e.name == 'MODEL_CTR_E':  e.update(interal, _pred[:, 0], _gt[:, 0], n1, self.n[interal])
                    if e.name == 'EMP_CTR_E':    e.update(interal,  _emp[:, 0], _gt[:, 0], n1, self.n[interal])
                    if e.name == 'GT_CTR':       e.update(interal,   _gt[:, 0],         0, n1, self.n[interal])
                    if e.name == 'MODEL_LVTR_E': e.update(interal, _pred[:, 1], _gt[:, 1], n1, self.n[interal])
                    if e.name == 'EMP_LVTR_E':   e.update(interal,  _emp[:, 1], _gt[:, 1], n1, self.n[interal])
                    if e.name == 'GT_LVTR':      e.update(interal,   _gt[:, 1],         0, n1, self.n[interal])

        lines = ['\t'.join(line) for line in zip(v['pv'].astype(str),
                                                 v['show_cnt'].astype(str),
                                                 pred[:, 0].round(3).astype(str),
                                                 emp[:, 0].round(3).astype(str),
                                                 y[:, 0].round(3).astype(str),
                                                 pred[:, 1].round(3).astype(str),
                                                 emp[:, 1].round(3).astype(str),
                                                 y[:, 1].round(3).astype(str))]
        self.ofp.write('\n'.join(lines))

    def echo(self):
        np.set_printoptions(suppress=True)
        print("SHOW_CNT Interal MAE:")
        for interal in self.interals:
            s = '\t'.join([f"INTERAL: {str(interal):>15}\tN: {self.n[interal]:>10}"] + [e.log(interal) for e in self.e])
            print(s)

    def plot(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        x = [np.mean(np.log2(interal)) for interal in self.interals]
        v = {e.name: [e.e[interal]
             for interal in self.interals]
             for e in self.e}
        ax.plot(x, v['EMP_CTR_E'], c='b', ls='--', lw=0.8, label='EMP_CTR_E')
        ax.plot(x, v['MODEL_CTR_E'], c='b', ls='-', lw=0.8, label='MODEL_CTR_E')
        ax.plot(x, v['EMP_LVTR_E'], c='g', ls='--', lw=0.8, label='EMP_LVTR_E')
        ax.plot(x, v['MODEL_LVTR_E'], c='g', ls='-', lw=0.8, label='MODEL_LVTR_E')
        ax.set_xlabel("log2(1+show_cnt) , 30d")
        ax.set_ylabel("MAE * 100")
        ax.set_title("Model Prediction vs Empirical")
        ax.legend(loc='upper left')
        fig.savefig(os.path.join(self.save_path, "fig.png"), dpi=130)
