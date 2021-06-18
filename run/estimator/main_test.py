import os
import sys
import json

sys.path.append('../../')
import csv
import pickle
from absl import flags, app, logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, BufferedShuffleDataset
# from sklearn import metrics as sk_metrics

from components.feature import *
from components.dataset import raw_iterator, RawDataset
from components.deepfm import DeepFM
from components.utils import Logger, redirect_stdouterr_to_file, Optimizers, save_checkpoint
from components.metrics import InteralMAE

FLAGS = flags.FLAGS
flags.DEFINE_boolean('is_local', False, "wheather do eval")
flags.DEFINE_string('do', 'train', "what task the esitimator gonna do. [train, eval, train&eval]")
flags.DEFINE_string('root', "/home/hemingzhi/.jupyter/ctr", "project root")
flags.DEFINE_string('train_path', "", "train splites")
flags.DEFINE_string('data_info_path', "", "vocab load path")
flags.DEFINE_string('train_info_path', "", "train config load path")
flags.DEFINE_string('eval_path', "", "eval splits")
flags.DEFINE_string('checkpoint_load_path', "", "checkpoint load path")
flags.DEFINE_string('save_path', "", "result save path")

flags.DEFINE_integer('epoch', 1, "epoch")
flags.DEFINE_integer('train_batch_size', 320, "batch size")

def parse_config():
    cfg = {}
    cfg['root'] = FLAGS.root
    cfg['is_local'] = FLAGS.is_local
    cfg['do'] = FLAGS.do

    cfg['train_path'] = FLAGS.train_path
    cfg['data_info_path'] = FLAGS.data_info_path
    cfg['train_info_path'] = FLAGS.train_info_path
    cfg['eval_path'] = FLAGS.eval_path
    cfg['checkpoint_load_path'] = FLAGS.checkpoint_load_path
    cfg['save_path'] = FLAGS.save_path

    # train param
    if cfg['is_local']:
        cfg['train_buffer_size'], cfg['eval_buffer_size'], cfg['num_workers'] = 1000, 100, 0
        cfg['train_batch_size'], cfg['eval_batch_size'] = 320, 1280
        cfg['log_step'] = 20
    else:
        cfg['train_buffer_size'], cfg['eval_buffer_size'], cfg['num_workers'] = 1000000, 100000, 7
        cfg['train_batch_size'], cfg['eval_batch_size'] = 320, 64800
        cfg['log_step'] = 2000


    cfg['epoch'] = FLAGS.epoch
    cfg['train_batch_size'] = FLAGS.train_batch_size
    cfg['device'] = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        cfg['device'] = 'cuda:0'
    return cfg


def get_configuration(train_info_path, data_info_path):
    with open(data_info_path, 'rb') as pkl:
        sparse_feature_info = pickle.load(pkl)
        dense_feature_info = pickle.load(pkl)
        label_feature_info = pickle.load(pkl)
    train_info = json.load(open(train_info_path, 'r'),)
    return train_info, sparse_feature_info, dense_feature_info, label_feature_info


def get_dataset(files, encoder_items, ret_idxs, label_feature_columns_info, batch_size=320, buffer_size=1, num_workers=0):
    ds = RawDataset(files, encoder_items, ret_idxs, label_feature_columns_info)  # label encode的过程在dataset中定义
    ds = BufferedShuffleDataset(ds, buffer_size)
    loader = DataLoader(ds, batch_size=batch_size,
                        num_workers=num_workers)
    return loader


def initialize(cfg, train_info, sparse_feature_info, dense_feature_info, label_feature_info):
    # Parse
    sparse_feature_columns_info = dict()
    dense_feature_columns_info = dict()
    label_feature_columns_info = dict()
    encoder_items = []
    ret_idxs = []
    start = 0

    for fname, v in sparse_feature_info.items():
        if fname in train_info['sparse_features']:
            sparse_feature_columns_info[fname] = {'index': (start, start + v['index'][1] - v['index'][0]),
                                                  'vocab_size': len(v['vocab']),
                                                  'is_sparse': v['is_sparse']}
            start += v['index'][1] - v['index'][0]
            idxs = list(range(*v['index']))
            encoder_items.append([idxs, v['vocab']])
            ret_idxs += idxs
    for fname, v in dense_feature_info.items():
        if fname in train_info['dense_features']:
            dense_feature_columns_info[fname] = {'index': (start, start + v['index'][1] - v['index'][0])}
            start += v['index'][1] - v['index'][0]
            idxs = list(range(*v['index']))
            ret_idxs += idxs

    for fname, v in label_feature_info.items():
        if fname in train_info['label_features']:
            label_feature_columns_info[fname] = v
    # Model
    sparse_feature_columns = [SparseFeat(name, v['index'], v['vocab_size'], 8, v['is_sparse'])
                              for name, v in sparse_feature_columns_info.items()]
    dense_feature_columns = [DenseFeat(name, v['index']) for name, v in dense_feature_columns_info.items()]

    model = DeepFM(sparse_feature_columns, dense_feature_columns, class_num=2, device=cfg['device'])

    _ = model.to(cfg['device'])

    optims = Optimizers()
    dense_params = [v for fname, v in model.named_parameters() if "embedding_dict" not in fname]
    embedding_params = [(fname, v) for fname, v in model.named_parameters() if "embedding_dict" in fname]

    sparse_params = []
    sparse_feat = [fname for fname, v in sparse_feature_info.items() if v['is_sparse']]
    for fname, v in embedding_params:
        feat = fname.split('.')[-2]
        if feat in sparse_feat:
            sparse_params.append(v)
        else:
            dense_params.append(v)

    if sparse_params:
        optims.add(torch.optim.SparseAdam(sparse_params))
    if dense_params:
        optims.add(torch.optim.Adam(dense_params))

    # Dataset
    if cfg['do'] == 'train':
        files = sorted([os.path.join(cfg['train_path'], file) for file in os.listdir(cfg['train_path'])])
        loader = get_dataset(files,
                            encoder_items,
                            ret_idxs,
                            label_feature_columns_info,
                            batch_size=cfg['train_batch_size'],
                            buffer_size=cfg['train_buffer_size'],
                            num_workers=cfg['num_workers'])

    if cfg['do'] == 'eval':
        checkpoint = torch.load(cfg['checkpoint_load_path'])
        model.load_state_dict(checkpoint['model'])
        files = sorted([os.path.join(cfg['eval_path'], file) for file in os.listdir(cfg['eval_path'])])
        loader = get_dataset(files,
                            encoder_items,
                            ret_idxs,
                            label_feature_columns_info,
                            batch_size=cfg['eval_batch_size'],
                            buffer_size=cfg['eval_buffer_size'],
                            num_workers=cfg['num_workers'])

    return model, optims, loader


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

def train(cfg, model, optims, loader):
    logger = Logger(cfg['log_step'], "Train")
    for epoch in range(cfg['epoch']):
        for batch in loader:
            inputs, y = batch['features'].to(cfg['device']), batch['label'].squeeze().to(cfg['device'])

            y_pred = model(inputs).squeeze()
            optims.zero_grad()
            loss = ctlvtr_loss(y_pred, y)
            loss.backward()
            optims.step()
            logger.log_info(loss=loss.item(), size=cfg['train_batch_size'], epoch=epoch)
    save_checkpoint(model, os.path.join(cfg['save_path'], 'checkpoint'))

def eval(cfg, model, loader):
    # loss_func = F.mse_loss
    loss_class = InteralMAE([*model.dense_feature_columns, *model.sparse_feature_columns],
                            l=0, r=1000000, interal_nums=50,
                            save_path=cfg['save_path'])
    # logger = Logger(cfg['log_step'], "Test")
    for batch in loader:
        inputs, y = batch['features'].to(cfg['device']), batch['label'].squeeze().to(cfg['device'])
        y_pred = model(inputs).squeeze()
        loss_class.update(inputs.cpu().data.numpy(), y_pred.cpu().data.numpy(), y.cpu().data.numpy())
        # logger.log_info(loss=loss_func(y_pred, y, reduction='mean').item(), size=320, epoch=0)
    loss_class.echo()
    loss_class.plot()


def main(argv):
    cfg = parse_config()
    redirect_stdouterr_to_file(os.path.join(cfg['save_path'], "log"))

    train_info, \
    sparse_feature_info, \
    dense_feature_info, \
    label_feature_info = get_configuration(cfg['train_info_path'], cfg['data_info_path'])

    model, optims, loader = initialize(cfg, train_info, sparse_feature_info, dense_feature_info, label_feature_info)

    if cfg['do'] == 'train':
        train(cfg, model, optims, loader)

    if cfg['do'] == 'eval':
        eval(cfg, model, loader)

if __name__ == "__main__":
    app.run(main)
