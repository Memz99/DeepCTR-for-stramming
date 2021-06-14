import numpy as np
import pandas as pd


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


class InteralMAE():
    def __init__(self, feature_columns, l=300, r=99999999, interal_nums=100):
        points = (2 ** np.linspace(np.log2(l), np.log2(r), interal_nums)).astype(int)
        self.interals = list(zip(points[:-1], points[1:]))

        self.pv_idx = [fc.index[0] for fc in feature_columns]
        for fc in feature_columns:
            if fc.name == "pv": self.pv_idx = fc.index[0]
            if fc.name == "show_cnt": self.show_cnt_idx = fc.index[0]
            if fc.name == "click_cnt": self.click_cnt_idx = fc.index[0]

        self.n = {interal: 0. for interal in self.interals}
        self.e = {interal: 0. for interal in self.interals}
        self.emp_e = {interal: 0. for interal in self.interals}
        self.y = {interal: 0. for interal in self.interals}


    def update(self, inputs, pred, y):
        pv = inputs[:, self.pv_idx]
        shows = inputs[:, self.show_cnt_idx]
        clicks = inputs[:, self.click_cnt_idx]
        emp_pred = np.where(shows > 0, clicks / shows, 0)
        emp_pred = np.where(emp_pred > 1, 1, emp_pred)
        for interal in self.interals:
            l, r = interal
            lind = np.logical_and(pv >= l, pv < r)
            n1 = sum(lind)
            if n1 > 0:
                self.n[interal] += n1

                loss = abs(sum(pred[lind] - y[lind])) * 100
                e0 = self.e[interal]
                self.e[interal] = e0 + (loss - n1 * e0) / self.n[interal]

                loss = abs(sum(emp_pred[lind] - y[lind])) * 100
                e0 = self.emp_e[interal]
                self.emp_e[interal] = e0 + (loss - n1 * e0) / self.n[interal]

                loss = sum(y[lind])
                y0 = self.y[interal]
                self.y[interal] = y0 + (loss - n1 * y0) / self.n[interal]

    def echo(self):
        np.set_printoptions(suppress=True)
        print("PV Interal MAE * 100:")
        for interal in self.interals:
            s = f"{str(interal):>20}\tN: {self.n[interal]:>10}\t\
                MODEL_MAE: {self.e[interal]:>10.7}\t\
                EMPIRICAL_MAE: {self.emp_e[interal]:>10.7}\
                GOURND_TRUE: {self.y[interal]:>5.3}"
            print(s)

    def plot(self, save_path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        x = [np.mean(np.log2(interal)) for interal in self.interals]
        emp_e = [self.emp_e[interal] for interal in self.interals]
        model_e = [self.e[interal] for interal in self.interals]
        ax.plot(x, emp_e, label='emp_e')
        ax.plot(x, model_e, label='model_e')
        ax.set_xlabel("log2(pv) , 30d")
        ax.set_ylabel("MAE * 100")
        ax.set_title("Model Prediction vs Empirical for the next day CLICK_RATE")
        ax.legend(loc='upper right')
        fig.savefig(save_path, dpi=130)
