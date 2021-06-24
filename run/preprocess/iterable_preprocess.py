import os
import csv
import json
import heapq
import pickle

from tqdm import tqdm
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_integer('is_encode', 0, "")
flags.DEFINE_integer('is_config', 0, "")
flags.DEFINE_integer('is_split', 0, "")
flags.DEFINE_string('data_file', "", "")
flags.DEFINE_string('feat_info_json', "", "feature info for the specific dataset")
flags.DEFINE_string('vocab_save_path', "", "vocab save path")
flags.DEFINE_string('config_save_path', "", "config save path")
# encode
flags.DEFINE_float('vocab_dataset_size', 0.85, "use how big proportions dataset to make vocab")
flags.DEFINE_integer('topk', 0, "vocab only contains the topk word")
# split
flags.DEFINE_integer('frag_size', 20000, "file frag size")
flags.DEFINE_float('train_dataset_size', 0.9, "file frag size")
flags.DEFINE_float('eval_dataset_size', 0.1, "file frag size")


def parse_config():
    cfg = {
        "is_encode": True,
        "is_config": True,
        "is_split": True,
        "data_file": "/Users/hemingzhi/Documents/Projects/ctr/data/xtr_v2/20210608",
        "feat_info": {
            "sparse": {
                "keyword": {"is_sparse": True},
                "item_id": {"is_sparse": True},
            },
            "dense": {
                "show_cnt_30d": {"type": "value"},
                "query_embed": {"type": "array"},
                "photo_embed": {"type": "array"},
            },
            "label":{
                "now_show_cnt": {},
                "now_click_cnt": {}
            },
        },
        "encode_info": {
            "topk": 10
        },
        "vocab_save_path": "/Users/hemingzhi/Documents/Projects/ctr/data/vocab/xtr_v2",
        "config_save_path": "/Users/hemingzhi/Documents/Projects/ctr/data/configs/xtr_v2",
        "split": {
            "train_dataset_size": 0.8,
            "eval_dataset_size": 0.2,
            "frag_size": 500,
            "array": ["query_embed", "photo_embed"]
        }
    }
    cfg = {}
    cfg['is_encode'] = FLAGS.is_encode
    cfg['is_config'] = FLAGS.is_config
    cfg['is_split'] = FLAGS.is_split
    cfg['data_file'] = FLAGS.data_file
    cfg['feat_info'] = json.load(open(FLAGS.feat_info_json, 'r'))
    cfg['vocab_save_path'] = FLAGS.vocab_save_path
    cfg['config_save_path'] = FLAGS.config_save_path
    cfg["encode"] = {
        "vocab_dataset_size": FLAGS.vocab_dataset_size,
        "topk": FLAGS.topk
    }
    cfg["split"] = {
        "train_dataset_size": FLAGS.train_dataset_size,
        "eval_dataset_size": FLAGS.eval_dataset_size,
        "frag_size": FLAGS.frag_size,
    }
    return cfg

class TopkHeap(object):

    def __init__(self, k):
        self.data = list()
        self.k = k

    def push(self, elem):
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem > topk_small:
                heapq.heapreplace(self.data, elem)

    def topk(self, sort=True):
        data = [heapq.heappop(self.data) for x in range(len(self.data))]
        if sort:
            data.sort(reverse=True)
        return data


class VocabContainer(object):

    def __init__(self, topk):
        if topk is None:
            self.vocabs = []
        else:
            self.vocabs = TopkHeap(topk)

    def add(self, fid, count):
        pair = [count, fid]
        if isinstance(self.vocabs, TopkHeap):
            self.vocabs.push(pair)
        else:
            self.vocabs.append(pair)

    def get_fid(self):
        if isinstance(self.vocabs, TopkHeap):
            vocabs = self.vocabs.topk(sort=True)
        else:
            vocabs = self.vocabs
        return [fid for _, fid in vocabs]


def get_feat2idx(cfg):
    dense_info = cfg['feat_info']['dense']
    with open(cfg['data_file'], 'r') as f:
        columns = f.readline().rstrip('\n').split('\t')
        row = f.readline().split('\t')

        feat2idx = {}
        start = 0
        for feat, val in zip(columns, row):
            L = 1
            if dense_info.get(feat, None) and dense_info[feat]['type'] == 'array':
                L = len(val[1:-1].split(','))
            feat2idx[feat] = (start, start+L)
            start += L
    return feat2idx


def encode_and_save(cfg, feat2idx, lens):
    # statistics frequency
    from collections import defaultdict
    f = open(cfg['data_file'], 'r')
    sparse2idx = {feat: feat2idx[feat][0] for feat in cfg['feat_info']['sparse']}
    feat2count = {feat: defaultdict(int) for feat in cfg['feat_info']['sparse']}
    count, limit = 0, int(lens*cfg['encode']['vocab_dataset_size'])
    while True:
        if count >= limit:
            break
        line = f.readline()
        if not line:
            break
        items = line.rstrip('\n').split('\t')
        for feat, idx in sparse2idx.items():
            fid = items[idx]
            feat2count[feat][fid] += 1
        count += 1
    # make vocab
    topk = cfg['encode']['topk']
    vc = {feat: VocabContainer(topk=topk) for feat in sparse2idx}
    for feat, dic in feat2count.items():
        container = vc[feat]
        for fid, count in dic.items():
            container.add(fid, count)
    idx2fid = {feat: c.get_fid() + ["__OOV__"] for feat, c in vc.items()}
    fid2idx = {feat: {fid: i for i, fid in enumerate(fids)} for feat, fids in idx2fid.items()}

    os.makedirs(cfg['vocab_save_path'], exist_ok=True)
    for feat, fids in idx2fid.items():
        output = open(f"{cfg['vocab_save_path']}/{feat}", 'wb')
        pickle.dump(fid2idx[feat], output, -1)

def save_configs(cfg, feat2idx):
    feat_info = cfg['feat_info']
    sparse_feature_info = {}
    dense_feature_info = {}
    label_feature_info = {}

    for fname in feat_info['sparse']:
        sparse_feature_info[fname] = {
            'index': feat2idx[fname],
            'vocab_load_path': f"{cfg['vocab_save_path']}/{fname}",
            'is_sparse': feat_info['sparse'][fname]['is_sparse']
        }
    for fname in feat_info['dense']:
        dense_feature_info[fname] = {'index': feat2idx[fname]}
    for fname in feat_info['label']:
        label_feature_info[fname] = {'index': feat2idx[fname]}

    data_feature_info = {
        "sparse_feature_info": sparse_feature_info,
        "dense_feature_info": dense_feature_info,
        "label_feature_info": label_feature_info
    }

    os.makedirs('/'.join(cfg['config_save_path'].split('/')[:-1]), exist_ok=True)
    json.dump(data_feature_info, open(f"{cfg['config_save_path']}", 'w'), indent=2)


def count_lines(data_file):
    with open(data_file, 'rb') as f:
        count = 0
        while True:
            data = f.read(0x400000)
            if not data:
                break
            count += data.count(b'\n')
    return count


def split_dataset(cfg, feat2idx, lens):

    def do_split(data_file, dataset_size, task):
        save_dir= f"{data_file}_{task}_splits"
        if os.path.exists(save_dir):
            os.system(f"rm -r {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

        count = 0
        frag_cnt = 0
        fout = None
        while True:
            line = f.readline()
            if count >= dataset_size or not line:
                break
            if count % frag_size == 0:
                if fout: fout.close()
                frag_cnt += 1
                fout = open(f"{save_dir}/frag_{frag_cnt}", 'w')
            if array_feat:
                items = line.rstrip().split('\t')
                for feat in array_feat:
                    l, r = feat2idx[feat]
                    items = items[:l] + items[l][1:-1].split(',') + items[l+1:]
                    line = '\t'.join(items) + '\n'
            fout.write(line)
            count += 1
        fout.close()

    split_cfg = cfg['split']
    array_feat = [feat for feat, v in cfg['feat_info']['dense'].items() if v['type'] == 'array']
    frag_size = split_cfg['frag_size']

    f = open(cfg['data_file'], 'r')
    columns = f.readline().rstrip('\n').split('\t')
    origin_column_idx = {col: i for i, col in enumerate(columns)}

    do_split(cfg['data_file'], int(lens*split_cfg['train_dataset_size']), "train")
    do_split(cfg['data_file'], int(lens*split_cfg['eval_dataset_size']), "eval")

def main(argv):
    import time
    cfg = parse_config()
    feat2idx = get_feat2idx(cfg)
    lens = count_lines(cfg['data_file'])
    if cfg['is_encode']:
        start = time.time()
        print("encoding...")
        encode_and_save(cfg, feat2idx, lens)
        print(f"done, cost:{(time.time() - start) / 60} min")
    if cfg['is_config']:
        save_configs(cfg, feat2idx)
    if cfg['is_split']:
        start = time.time()
        print("spliting...")
        split_dataset(cfg, feat2idx, lens)
        print(f"done, cost:{(time.time() - start) / 60} min")


if __name__ == "__main__":
    app.run(main)