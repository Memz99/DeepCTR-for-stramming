import json
import csv
import time

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, MinMaxScaler
import numpy as np
import pandas as pd

import torch


def redirect_stdouterr_to_file(log_file):
    import logging
    import absl.logging
    import os
    import sys

    class RedirectLogger(object):
        def __init__(self, logger):
            self._logger = logger
            self._msg = ""

        def write(self, message):
            self._msg = self._msg + message
            while "\n" in self._msg:
                pos = self._msg.find("\n")
                self._logger(self._msg[:pos])
                self._msg = self._msg[pos + 1:]

        def flush(self):
            if self._msg != "":
                self._logger(self._msg)
                self._msg = ""

    # remove absl logging handler
    absl.logging._warn_preinit_stderr = False
    logging.root.removeHandler(absl.logging._absl_handler)

    # redirect to both file and console
    if log_file is not None and os.path.dirname(log_file) != "":
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file, mode="w", encoding='utf-8'), logging.StreamHandler()]
        if log_file is not None
        else [logging.StreamHandler()],
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sys.stderr = RedirectLogger(logging.error)
    sys.stdout = RedirectLogger(logging.info)
    return


class Logger(object):

    def __init__(self, log_steps, task='Train'):
        self.task = task
        self._log_steps = log_steps
        self.pre = time.time()
        self._whole_steps = 0
        self._cleanup()

    def log_info(self, loss, size, epoch):
        now = time.time()
        self._total_loss += loss
        self._total_time += now - self.pre
        self._total_size += size  # size 是 batch_size，即每秒处理多少个item
        self._total_steps += 1
        self._whole_steps += 1
        self.pre = now

        if self._total_steps >= self._log_steps:
            avg_loss = self._total_loss / self._total_steps
            fps = self._total_size / float(self._total_time)
            self._log_to_console(avg_loss, self._total_time, fps, epoch, self._whole_steps)
            self._cleanup()

    def _log_to_console(self, loss, time, fps, epoch, step):
        print(
            "[%s] Epoch: %d\tStep: %d\tLoss: %.5f\tTime: %.2f\tFPS: %d"
            % (self.task, epoch, step, loss, time, fps)
        )

    def _cleanup(self):
        self._total_loss = 0
        self._total_time = 0
        self._total_size = 0
        self._total_steps = 0


class Optimizers(object):
    def __init__(self, step_size=10000):
        self.optims = []
        self.schedulers = []
        self.step_size = step_size

    def add(self, optim):
        self.optims.append(optim)
        self.schedulers.append(torch.optim.lr_scheduler.StepLR(optim, self.step_size, gamma=0.1))

    def zero_grad(self):
        for optim in self.optims:
            optim.zero_grad()

    def step(self):
        for i in range(len(self.optims)):
            self.optims[i].step()
            self.schedulers[i].step()

def save_checkpoint(model, path):
    import os
    import torch
    os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
    torch.save({
        "model": model.state_dict()
    }, path)