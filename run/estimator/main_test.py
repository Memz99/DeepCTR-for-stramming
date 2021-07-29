import os
import sys
import json

sys.path.append('../../')
import pickle
from absl import flags, app

import torch

from torch.utils.data import DataLoader, BufferedShuffleDataset
from components.feature import *
from components.dataset import TrainDataset, EvalDataset
from components.deepfm import DeepFM
from components.utils import Logger, redirect_stdouterr_to_file, Optimizers, save_checkpoint
from components.estimator import Estimator


FLAGS = flags.FLAGS
flags.DEFINE_boolean('is_local', False, "wheather do eval")
flags.DEFINE_string('do', 'train', "what task the esitimator gonna do. [train, eval, train&eval]")
flags.DEFINE_string('root', "/home/hemingzhi/.jupyter/ctr", "project root")
flags.DEFINE_string('train_path', "", "train splites")
flags.DEFINE_string('data_info_path', "", "vocab load path")
flags.DEFINE_string('model_info_path', "", "model config load path")
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
    cfg['eval_path'] = FLAGS.eval_path
    cfg['checkpoint_load_path'] = FLAGS.checkpoint_load_path
    cfg['save_path'] = FLAGS.save_path

    # configuration
    cfg['feature_info'] = {}
    cfg['feature_info']['model_info'] = json.load(open(FLAGS.model_info_path, 'r'))
    cfg['feature_info']['data_feature_info'] = json.load(open(FLAGS.data_info_path, 'r'))

    # train param
    if cfg['is_local']:
        cfg['train_buffer_size'], cfg['eval_buffer_size'], cfg['num_workers'] = 1000, 100, 0
        cfg['train_batch_size'], cfg['eval_batch_size'] = 320, 1280
        cfg['log_step'] = 20
    else:
        cfg['train_buffer_size'], cfg['eval_buffer_size'], cfg['num_workers'] = 1000000, 100000, 7
        cfg['train_batch_size'], cfg['eval_batch_size'] = 320, 64800
        cfg['train_batch_size'] = FLAGS.train_batch_size
        cfg['log_step'] = 2000

    cfg['epoch'] = FLAGS.epoch
    cfg['device'] = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        cfg['device'] = 'cuda:0'

    return cfg


def get_dataset(cfg):
    model_info = cfg['feature_info']['model_info']
    sparse_feature_info = cfg['feature_info']['data_feature_info']['sparse_feature_info']
    dense_feature_info = cfg['feature_info']['data_feature_info']['dense_feature_info']
    label_feature_info = cfg['feature_info']['data_feature_info']['label_feature_info']

    # Parse
    label_feature_columns_info = {}
    encoder_items = []
    ret_idxs = []

    for fname, v in sparse_feature_info.items():
        if fname in model_info['sparse_features']:
            vocab = pickle.load(open(v['vocab_load_path'], 'rb'))
            idxs = list(range(*v['index']))
            encoder_items.append([idxs, vocab])
            ret_idxs += idxs

    for fname, v in dense_feature_info.items():
        if fname in model_info['dense_features']:
            idxs = list(range(*v['index']))
            ret_idxs += idxs

    for fname, v in label_feature_info.items():
        if fname in model_info['label_features']:
            label_feature_columns_info[fname] = v

    if cfg['do'] == 'train':
        files = sorted([os.path.join(cfg['train_path'], file) for file in os.listdir(cfg['train_path'])])
        ds = TrainDataset(files, encoder_items, ret_idxs, label_feature_columns_info)  # label encode的过程在dataset中定义
        ds = BufferedShuffleDataset(ds, cfg['train_buffer_size'])
        loader = DataLoader(ds, batch_size=cfg['train_batch_size'], num_workers=cfg['num_workers'])

    if cfg['do'] == 'eval':
        eval_indicator_columns_info = {}
        for fname, v in {**sparse_feature_info, **dense_feature_info, **label_feature_info}.items():
            if fname in model_info['eval_features']:
                eval_indicator_columns_info[fname] = v

        files = sorted([os.path.join(cfg['eval_path'], file) for file in os.listdir(cfg['eval_path'])])
        ds = EvalDataset(files, encoder_items, ret_idxs,
                         label_feature_columns_info, eval_indicator_columns_info)  # eval indicator用于评估使用
        ds = BufferedShuffleDataset(ds, cfg['eval_buffer_size'])
        loader = DataLoader(ds, batch_size=cfg['eval_batch_size'], num_workers=cfg['num_workers'])

    return loader


def initialize(cfg):
    sparse_feature_info = cfg['feature_info']['data_feature_info']['sparse_feature_info']
    dense_feature_info = cfg['feature_info']['data_feature_info']['dense_feature_info']
    label_feature_info = cfg['feature_info']['data_feature_info']['label_feature_info']

    model_info = cfg['feature_info']['model_info']

    # Parse
    sparse_feature_columns_info = dict()
    dense_feature_columns_info = dict()
    label_feature_columns_info = dict()
    start = 0

    for fname, v in sparse_feature_info.items():
        if fname in model_info['sparse_features']:
            vocab = pickle.load(open(v['vocab_load_path'], 'rb'))
            sparse_feature_columns_info[fname] = {'index': (start, start + v['index'][1] - v['index'][0]),
                                                  'vocab_size': len(vocab),
                                                  'is_sparse': v['is_sparse'],
                                                  'group': v['group']}
            start += v['index'][1] - v['index'][0]
    for fname, v in dense_feature_info.items():
        if fname in model_info['dense_features']:
            dense_feature_columns_info[fname] = {'index': (start, start + v['index'][1] - v['index'][0]),
                                                 'group': v['group']}
            start += v['index'][1] - v['index'][0]

    for fname, v in label_feature_info.items():
        if fname in model_info['label_features']:
            label_feature_columns_info[fname] = v

    # Model
    sparse_feature_columns = [SparseFeat(name=name,
                                         index=v['index'],
                                         vocabulary_size=v['vocab_size'],
                                         embedding_dim=8,
                                         sparse_embedding=v['is_sparse'],
                                         group=v['group'])
                              for name, v in sparse_feature_columns_info.items()]
    dense_feature_columns = [DenseFeat(name=name,
                                       index=v['index'],
                                       group=v['group'])
                             for name, v in dense_feature_columns_info.items()]

    model = DeepFM(sparse_feature_columns, dense_feature_columns, class_num=2, device=cfg['device'])

    _ = model.to(cfg['device'])

    params = {
        'cfg': cfg,
    }

    if cfg['do'] == "train":
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

    if cfg['do'] == 'eval':
        eval_indicator_columns_info = {}
        for fname, v in {**sparse_feature_info, **dense_feature_info, **label_feature_info}.items():
            if fname in model_info['eval_features']:
                eval_indicator_columns_info[fname] = v

        checkpoint = torch.load(cfg['checkpoint_load_path'])
        model.load_state_dict(checkpoint['model'])
        optims = None
        params['indicator_columns'] = eval_indicator_columns_info

    estimator = Estimator(model=model, optims=optims, params=params)
    return estimator


def main(argv):
    cfg = parse_config()
    redirect_stdouterr_to_file(os.path.join(cfg['save_path'], "log"))

    estimator = initialize(cfg)
    loader = get_dataset(cfg)
    estimator.loader = loader

    if cfg['do'] == 'train':
        estimator.train()
        # train(cfg, model, optims, loader)

    if cfg['do'] == 'eval':
        estimator.params['eval_info'] = loader.dataset.dataset.eval_info
        estimator.eval()
        # eval(cfg, model, loader)

if __name__ == "__main__":
    app.run(main)
