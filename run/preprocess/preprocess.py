import os
import csv
import json

import numpy as np
import pandas as pd

from tqdm import tqdm
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string('root', "/home/hemingzhi/.jupyter/ctr", "project root")
flags.DEFINE_string('data_file', "", "data file path")
flags.DEFINE_string('data_config', "", "data config")
flags.DEFINE_string('vocab_save_path', "", "vocab save path")
flags.DEFINE_integer('frag_size', 20000, "file frag size")

root = "/Users/hemingzhi/Documents/Projects/ctr" # root = "/home/hemingzhi/.jupyter/ctr"
table = "xtr_base"
date = "20210608_filtered"

config_path = os.path.join(root, 'run/preprocess/configs/xtr_base_no_emb.json')
vocab_path = os.path.join(root, "data", "vocab")

def parse_config():
    config = json.load(open(FLAGS.data_config, 'r'))
    cfg = {
        "data_file": FLAGS.data_file,
        "sparse_features": config['sparse_features'],
        "dense_features": config['dense_features'],
        "label_features": config['label_features'],
        "sparse_embedding_features": config['sparse_embedding_features'],
        "frag_size": FLAGS.frag_size,
        "vocab_save_path": FLAGS.vocab_save_path
    }
    return cfg

def read_and_fill(data_file, sparse_features=None, usecols=None):
    print("reading csv...")
    data = pd.read_csv(data_file, sep="\t", dtype={feat: str for feat in sparse_features},
               error_bad_lines=False, quoting=csv.QUOTE_NONE, encoding='utf-8',
               low_memory=False, usecols=usecols)
    for feat in usecols:
        if feat in sparse_features:
            data[feat] = data[feat].fillna("__UNK__")
        else:
            data[feat] = data[feat].fillna(0)
    print("reading done!")
    return data

def split_dataset(data_file, frag_size, sparse_features, dense_features, label_features):
    print("spliting...")
    from sklearn.model_selection import train_test_split

    data = read_and_fill(data_file,
                  sparse_features=sparse_features,
                  usecols=sparse_features+dense_features+label_features)
    data, eval = train_test_split(data, train_size=0.9)
    data.to_csv(data_file+"_train", index=False, sep='\t')
    split_files(data, data_file + "_train", frag_size)
    split_files(eval, data_file + "_eval", frag_size)
    eval = None
    data, val = train_test_split(data, train_size=0.95)
    print("spliting done!")
    return data  # return 95% of train set to do encoding


def split_files(df, data_file, frag_size):
    print("deviding...")
    os.makedirs(data_file + "_splits", exist_ok=True)

    v = df.values
    i = 0
    cnt = 0
    fi = open(f"{data_file}_splits/frag_{i}", 'w', encoding='utf-8')
    for line in tqdm(v):
        row = '\t'.join(line.astype(str)) + '\n'
        if cnt == frag_size:
            i += 1
            fi.close()
            fi = open(f"{data_file}_splits/frag_{i}", 'w', encoding='utf-8')
            cnt = 0
        fi.write(row)
        cnt += 1
    fi.close()
    print("done!")


def encode_and_save(data,
                    sparse_features,
                    dense_features,
                    label_features,
                    sparse_embedding_features,
                    vocab_save_path):
    print("encoding...")
    import pickle
    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, MinMaxScaler
    
    sparse_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    sparse_encoder.fit(data[sparse_features])
    
    sparse_feature_info = {}

    feat2idx = {feat: (i, i+1) for i, feat in enumerate(data.columns)}

    for fname, word_list in zip(sparse_features, sparse_encoder.categories_):
        vocab = {word: i for i, word in enumerate(np.concatenate((word_list, ["__OOV__"])))}
        sparse_feature_info[fname] = {'index': feat2idx[fname],
                                      'vocab': vocab,
                                      'is_sparse': False if fname not in sparse_embedding_features else True}
    dense_feature_info = {}
    for fname in dense_features:
        dense_feature_info[fname] = {'index': feat2idx[fname]}
    
    # Label信息记录，在dataset构造的部分再做encoder
    label_feature_info = {}
    for fname in label_features:
        label_feature_info[fname] = {'index': feat2idx[fname]}
    
    output = open(vocab_save_path, 'wb')
    pickle.dump(sparse_feature_info, output, -1)
    pickle.dump(dense_feature_info, output, -1)
    pickle.dump(label_feature_info, output, -1)
    pickle.dump(feat2idx, output, -1)
    output.close()
    print("encoding done!")


def directly_preprocess(cfg):

    data_file = cfg['data_file']
    frag_size = cfg['frag_size']
    os.makedirs(data_file + "_splits", exist_ok=True)

    print("deviding...")
    f = open(cfg['data_file'], 'r')
    columns = f.readline().split('\t')

    v = df.values
    i = 0
    cnt = 0
    fi = open(f"{data_file}_splits/frag_{i}", 'w', encoding='utf-8')
    for line in tqdm(v):
        row = '\t'.join(line.astype(str)) + '\n'
        if cnt == frag_size:
            i += 1
            fi.close()
            fi = open(f"{data_file}_splits/frag_{i}", 'w', encoding='utf-8')
            cnt = 0
        fi.write(row)
        cnt += 1
    fi.close()
    print("done!")


def main(argv):
    import time
    start = time.time()
    cfg = parse_config()
    
    train = split_dataset(cfg['data_file'],
                          cfg['frag_size'],
                          cfg['sparse_features'],
                          cfg['dense_features'],
                          cfg['label_features'])

    encode_and_save(train,
                    cfg['sparse_features'],
                    cfg['dense_features'],
                    cfg['label_features'],
                    cfg['sparse_embedding_features'],
                    cfg['vocab_save_path'])
    print("cost:", time.time() - start)

if __name__ == '__main__':
    # app.run(main)
    cfg = {
        "data_file": "../../data/xtr_v2/20210608",
        "frag_size": 2000
    }
    directly_preprocess(cfg)