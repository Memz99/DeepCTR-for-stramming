import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm

def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    # for varlen sparse features, {embedding_name: nn.EmbeddingBag}
    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(
            feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=feat.sparse_embedding)
         for feat in feature_columns}
    )

    # for feat in varlen_sparse_feature_columns:
    #     embedding_dict[feat.embedding_name] = nn.EmbeddingBag(
    #         feat.dimension, embedding_size, sparse=sparse, mode=feat.combiner)

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict.to(device)

def activation_layer(act_name):
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer

class DNN(nn.Module):

    def __init__(self, inputs_dim, hidden_units, activation='relu', dropout_rate=0, init_std=0.0001):
        super(DNN, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, inputs):
        x = inputs
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            x = self.activation_layers[i](x)
            x = self.dropout(x)
        return x


class Linear(nn.Module):
    def __init__(self, sparse_feature_columns, dense_feature_columns,
                 init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.sparse_feature_columns = sparse_feature_columns
        self.dense_feature_columns = dense_feature_columns

        self.embedding_dict = create_embedding_matrix(sparse_feature_columns)

        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1).to(
                device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, inputs):

        dense_value_list = [inputs[:, feat.index[0]:feat.index[1]]
                          for feat in self.dense_feature_columns]

        sparse_embedding_list = [
            self.embedding_dict[feat.name]( # 取出 embedding 表
                inputs[:, feat.index[0]:feat.index[1]].long())
            for feat in self.sparse_feature_columns]

#         sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
#             X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
#             feat in self.sparse_feature_columns]

#         dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
#                             self.dense_feature_columns]

        linear_logit = torch.zeros([inputs.shape[0], 1]).to(dense_value_list[0].device)
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit

        return linear_logit


class FM(nn.Module):

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term


class DeepFM(nn.Module):
    def __init__(self, sparse_feature_columns, dense_feature_columns,
                 dnn_hidden_units=(256, 128, 1), dnn_activation='relu', dnn_dropout=0,
                 init_std=0.0001, seed=1173, device='cpu'):
        super(DeepFM, self).__init__()
        torch.manual_seed(seed)
        self.sparse_feature_columns = sparse_feature_columns
        self.dense_feature_columns = dense_feature_columns

        self.embedding_dict = create_embedding_matrix(sparse_feature_columns)
        # 正则项...

        # DNN
        dnn_input_dim = sum(feat.dimension for feat in sparse_feature_columns + dense_feature_columns)
        self.dnn = DNN(dnn_input_dim, dnn_hidden_units,
               activation=dnn_activation, dropout_rate=dnn_dropout)

        # FM
        self.linear = Linear(sparse_feature_columns, dense_feature_columns)
        self.fm = FM()

        # Output
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.to(device)

    def split_tensor(self, inputs):

        dense_value_list = [inputs[:, feat.index[0]:feat.index[1]]
                          for feat in self.dense_feature_columns]

        sparse_embedding_list = [
            self.embedding_dict[feat.name]( # 取出 embedding 表
                inputs[:, feat.index[0]:feat.index[1]].long())
            for feat in self.sparse_feature_columns]

        return sparse_embedding_list, dense_value_list

    def forward(self, inputs):
        sparse_embedding_list, dense_value_list = self.split_tensor(inputs)
        logit = self.linear(inputs)

        if len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)

        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1)
        if sparse_embedding_list:
            sparse_dnn_input = torch.flatten(
                torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
            dnn_input = torch.cat((sparse_dnn_input, dense_dnn_input), dim=-1)
        else:
            dnn_input = dense_dnn_input
        dnn_logit = self.dnn(dnn_input)
        logit += dnn_logit

        y_pred = torch.sigmoid(logit + self.bias)
        return y_pred