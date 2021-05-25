import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch.nn.functional as F

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, MinMaxScaler
from sklearn import metrics as sk_metrics
import numpy as np
import pandas as pd

from components.feature import *
from components.dataset import raw_iterator, RawDataset
from components.deepfm import DeepFM


feat2idx =  {'user_id': (0, 1),
             'keyword': (1, 2),
             'sequence_keyword': (2, 3),
             'search_source': (3, 4),
             'session_id': (4, 5),
             'item_id': (5, 6),
             'show_cnt': (6, 7),
             'click_cnt': (7, 8),
             'play_cnt': (8, 9),
             'like_cnt': (9, 10),
             'follow_cnt': (10, 11),
             'long_view_cnt': (11, 12),
             'short_view_cnt': (12, 13),
             'first_click': (13, 14),
             'last_click': (14, 15),
             'first_view': (15, 16),
             'last_view': (16, 17),
             'skip': (17, 18),
             'exam': (18, 19),
             'play_duration': (19, 20),
             'slide_show': (20, 21),
             'slide_click': (21, 22),
             'pos': (22, 23),
             'atlas_view_cnt': (23, 24),
             'download_cnt': (24, 25),
             'feed_model': (25, 26),
             'p_date': (26, 27),
             'product': (27, 28)}


sparse_features = ['user_id', 'keyword', 'sequence_keyword', 'search_source', 'session_id', 'item_id',
                   'first_click', 'last_click', 'first_view', 'last_view',
                   'pos', 'feed_model', 'p_date', 'product']

dense_features = ['show_cnt', 'click_cnt', 'play_cnt', 'like_cnt', 'follow_cnt', 'long_view_cnt',
                  'short_view_cnt', 'slide_show', 'slide_click', 'atlas_view_cnt']

data = pd.read_csv("data/raw/20210517", sep="\t", dtype={feat: str for feat in sparse_features})
## 缺失值处理
for feat in sparse_features + dense_features:
    if feat in sparse_features:
        data[feat] = data[feat].fillna("")
    else:
        data[feat] = data[feat].fillna(0)

## 离散特征编码
sparse_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
data[sparse_features] = sparse_encoder.fit_transform(data[sparse_features])

sparse_feature_info = {}
for fname, word_list in zip(sparse_features, sparse_encoder.categories_):
    vocab = {word: i for i, word in enumerate(np.concatenate((word_list, ["__UNK__"])))}
    sparse_feature_info[fname] = (feat2idx[fname], vocab)
dense_feature_info = {}
for fname in dense_features:
    dense_feature_info[fname] = feat2idx[fname]
# ------------------------------------------------------------------------------------------

files = ["data/raw/20210517"]
ds = RawDataset(files, sparse_feature_info, feat2idx["click_cnt"][0])
loader = DataLoader(ds, batch_size=32, num_workers=3)

sparse_feature_columns = [SparseFeat(name, index, len(vocab), 4, False)
                          for name, (index, vocab) in sparse_feature_info.items()]
dense_feature_columns = [DenseFeat(name, index) for name, index in dense_feature_info.items()]

model = DeepFM(sparse_feature_columns, dense_feature_columns)

loss_func = F.binary_cross_entropy
optim = torch.optim.Adam(model.parameters())  # 0.001
metric = sk_metrics.roc_auc_score

i = 0
for epoch in range(5):
    for batch in loader:
        inputs, y = batch['features'], batch['label'].squeeze()

        y_pred = model(inputs).squeeze()
        optim.zero_grad()
        loss = loss_func(y_pred, y,reduction='sum')
        loss.backward()
        optim.step()
        i += 1
        if i % 200 == 0:
            print("loss:", loss.item())

    auc = metric(y.cpu().unsqueeze(dim=-1).data.numpy(), y_pred.cpu().unsqueeze(dim=-1).data.numpy())
    print(f"Eopch {epoch} auc: {auc}")