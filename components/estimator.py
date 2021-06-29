import os

import torch.nn.functional as F

from components.utils import Logger, save_checkpoint
from components.metrics import InteralMAE


def ctr_loss(y_pred, y):
    # loss_func = F.binary_cross_entropy
    loss_func = F.mse_loss
    return loss_func(y_pred, y, reduction='mean')


def ctlvtr_loss(y_pred, y):
    loss_func = F.mse_loss
    loss = 0
    for col in range(y.shape[1]):
        loss += loss_func(y_pred[:, col], y[:, col])
    return loss


class Estimator(object):

    def __init__(self, model, optims, loader=None, params=None):
        self.model = model
        self.loader = loader
        self.params = params
        self.optims = optims

    def train(self):
        cfg = self.params['cfg']
        logger = Logger(cfg['log_step'], "Train")
        for epoch in range(cfg['epoch']):
            for batch in self.loader:
                inputs, y = batch['features'].to(cfg['device']), batch['label'].to(cfg['device'])

                y_pred = self.model(inputs).squeeze()
                self.optims.zero_grad()
                loss = ctlvtr_loss(y_pred, y)
                loss.backward()
                self.optims.step()
                logger.log_info(loss=loss.item(), size=cfg['train_batch_size'], epoch=epoch)
        save_checkpoint(self.model, os.path.join(cfg['save_path'], 'checkpoint'))

    def eval(self):
        cfg = self.params['cfg']
        loss_class = InteralMAE(indicator_columns=self.params['indicator_columns'],
                                l=0, r=1000000, interal_nums=50,
                                save_path=cfg['save_path'])
        for batch in self.loader:
            inputs, y, indicator = \
                batch['features'].to(cfg['device']), \
                batch['label'].to(cfg['device']),\
                batch['indicator'],
            y_pred = self.model(inputs).squeeze()
            loss_class.update(indicator=indicator,
                              pred=y_pred.cpu().data.numpy(),
                              y=y.cpu().data.numpy())
        loss_class.echo()
        loss_class.plot()
