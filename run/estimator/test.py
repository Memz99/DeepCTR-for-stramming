import os
import sys

sys.path.append('../../')
import csv
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, BufferedShuffleDataset
# from sklearn import metrics as sk_metrics

from components.feature import *
from components.dataset import raw_iterator, RawDataset
from components.deepfm import DeepFM
from components.utils import Logger, redirect_stdouterr_to_file, Optimizers
from components.metrics import InteralMAE

# root = "/Users/hemingzhi/Documents/Projects/ctr"
root = "/home/hemingzhi/.jupyter/ctr"
table = "xtr_base"
train_date = "20210527_filtered"
test_date = "20210527_filtered"

train_path = f"{root}/data/{table}/{train_date}_train_splits"
test_path = f"{root}/data/{table}"
save_path = f"{root}/result/{table}/{train_date}/checkpoint"
log_path = f"{root}/result/{table}/{train_date}/log"
load_path = ""
info_path = os.path.join(root, f"data/vocab/{table}_{train_date}.pkl")

device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'

local = True
# train param
if local:
    train_buffer_size, test_buffer_size, train_num_workers = 1000, 100, 0
else:
    train_buffer_size, test_buffer_size, train_num_workers = 1000000, 100000, 7


def get_configuration(info_path):
    with open(info_path, 'rb') as pkl:
        sparse_feature_info = pickle.load(pkl)
        dense_feature_info = pickle.load(pkl)
        label_feature_info = pickle.load(pkl)
    return sparse_feature_info, dense_feature_info, label_feature_info


def get_dataset(files, encoder_items, ret_idxs, label_feature_info, buffer_size=1, num_workers=0):
    ds = RawDataset(files, encoder_items, ret_idxs, label_feature_info)  # label encode的过程在dataset中定义
    ds = BufferedShuffleDataset(ds, buffer_size)
    loader = DataLoader(ds, batch_size=320,
                        num_workers=num_workers)
    return loader


def initialize(sparse_feature_info, dense_feature_info, label_feature_info):
    # Define Model
    sparse_feature_columns_info = dict()
    dense_feature_columns_info = dict()
    encoder_items = []
    ret_idxs = []
    start = 0

    for fname, v in sparse_feature_info.items():
        sparse_feature_columns_info[fname] = {'index': (start, start + v['index'][1] - v['index'][0]),
                                              'vocab_size': len(v['vocab']),
                                              'is_sparse': v['is_sparse']}
        start += v['index'][1] - v['index'][0]
        idxs = list(range(*v['index']))
        encoder_items.append([idxs, v['vocab']])
        ret_idxs += idxs
    for fname, v in dense_feature_info.items():
        dense_feature_columns_info[fname] = {'index': (start, start + v['index'][1] - v['index'][0])}
        start += v['index'][1] - v['index'][0]
        idxs = list(range(*v['index']))
        ret_idxs += idxs

    sparse_feature_columns = [SparseFeat(name, v['index'], v['vocab_size'], 8, v['is_sparse'])
                              for name, v in sparse_feature_columns_info.items()]
    dense_feature_columns = [DenseFeat(name, v['index']) for name, v in dense_feature_columns_info.items()]

    model = DeepFM(sparse_feature_columns, dense_feature_columns)

    _ = model.to(device)

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
    return model, optims, sparse_feature_columns_info, dense_feature_columns_info, encoder_items, ret_idxs


def train(model, optims, loader):
    loss_func = F.binary_cross_entropy

    logger = Logger(2000, "Train")
    for epoch in range(2):
        loader.dataset.dataset.reset()
        for batch in loader:
            inputs, y = batch['features'].to(device), batch['label'].squeeze().to(device)

            y_pred = model(inputs).squeeze()
            optims.zero_grad()
            loss = loss_func(y_pred, y, reduction='sum')
            loss.backward()
            optims.step()
            logger.log_info(loss=loss.item(), size=320, epoch=epoch)
    torch.save({
        "model": model.state_dict()
    }, save_path)

def test(model, loader):
    loss_func = F.mse_loss
    loss_class = InteralMAE(model.dense_feature_columns,
                            l=300, r=99999999, interal_nums=20)
    logger = Logger(1000, "Test")
    for batch in loader:
        inputs, y = batch['features'].to(device), batch['label'].squeeze().to(device)
        y_pred = model(inputs).squeeze()
        loss_class.update(inputs.cpu().data.numpy(), y_pred.cpu().data.numpy(), y.cpu().data.numpy())
        logger.log_info(loss=loss_func(y_pred, y, reduction='mean').item(), size=320, epoch=0)
    loss_class.echo()


if __name__ == "__main__":
    redirect_stdouterr_to_file(log_path)
    sparse_feature_info, dense_feature_info, label_feature_info = get_configuration(info_path)
    train_files = [os.path.join(train_path, file) for file in os.listdir(train_path)]

    model, optims, \
    sparse_feature_columns_info, \
    dense_feature_columns_info, \
    encoder_items, ret_idxs = initialize(sparse_feature_info, dense_feature_info, label_feature_info)

    if load_path:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model'])

    train_loader = get_dataset(train_files,
                               encoder_items,
                               ret_idxs,
                               label_feature_info,
                               buffer_size=train_buffer_size,
                               num_workers=train_num_workers)
    train(model, optims, train_loader)

    test_files = [os.path.join(test_path, f"{test_date}_test")]
    test_loader = get_dataset(test_files,
                              encoder_items,
                              ret_idxs,
                              label_feature_info,
                              buffer_size=test_buffer_size,
                              num_workers=1)
    test(model, test_loader)
