import os
import csv
import json

import numpy as np
import pandas as pd

from tqdm import tqdm

# root = "/Users/hemingzhi/Documents/Projects/ctr"
root = "/home/hemingzhi/.jupyter/ctr"
table = "xtr_base"
date = "20210608_filtered"
frag_size = 20000

config_path = os.path.join(root, 'run/preprocess/configs/xtr_base_no_emb.json')
vocab_path = os.path.join(root, "data", "vocab")

config = json.load(open(config_path, 'r'))
sparse_features = config['sparse_features']
dense_features = config['dense_features']
label_features = config['label_features']
sparse_embedding_features = config['sparse_embedding_features']

def get_info(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        head = f.readline()
    head = [feat.strip() for feat in head.split('\t')]
    feat2idx = {feat: (i, i+1) for i, feat in enumerate(head)}
    return head, feat2idx

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

def split_dataset(data_file):
    print("spliting...")
    from sklearn.model_selection import train_test_split

    data = read_and_fill(data_file,
                  sparse_features=sparse_features,
                  usecols=sparse_features+dense_features+label_features)
    data, eval = train_test_split(data, train_size=0.9)
    data.to_csv(data_file+"_train", index=False, sep='\t')
    split_files(data, data_file + "_train")
    split_files(eval, data_file + "_eval")
    eval = None
    data, val = train_test_split(data, train_size=0.95)
    print("spliting done!")
    return data  # return 95% of train set to do encoding


def split_files(df, data_file):
    print("deviding...")
    os.makedirs(data_file + "_splits", exist_ok=True)
    # f = open(data_file, "r", encoding='utf-8')
    # f.readline() # column name
    # i = 0
    # end = False
    # while not end:
    #     fi = open(f"{data_file}_splits/{date}_{i}", 'w', encoding='utf-8')
    #     for _ in range(frag_size):
    #         row = f.readline()
    #         if row != "":
    #             fi.write(row)
    #         else:
    #             end = True
    #             break
    #     fi.close()
    #     i += 1
    # f.close()
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


def encode_and_save(data):
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
    
    output = open(f'{vocab_path}/{table}_{date}.pkl', 'wb')
    pickle.dump(sparse_feature_info, output, -1)
    pickle.dump(dense_feature_info, output, -1)
    pickle.dump(label_feature_info, output, -1)
    pickle.dump(feat2idx, output, -1)
    output.close()
    print("encoding done!")


if __name__ == '__main__':
    import time
    start = time.time()
    data_file = os.path.join(root, "data", table, date)
    
    head, feat2idx = get_info(data_file)
    
    train = split_dataset(data_file)

    encode_and_save(train)
    print("cost:", time.time() - start)
