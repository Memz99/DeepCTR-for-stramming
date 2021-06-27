import torch
import torch.nn as nn

from collections import defaultdict
from components.feature import Group

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
        self.bn = torch.nn.BatchNorm1d(inputs_dim)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_units[i]) for i in range(1, len(hidden_units))])
        self.activation_layers = nn.ModuleList(
            [activation_layer(activation) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, inputs):
        x = self.bn(inputs)
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            x = self.bns[i](x)
            x = self.activation_layers[i](x)
            x = self.dropout(x)
        return x


class Linear(nn.Module):
    def __init__(self,
                 sparse_feature_columns,
                 dense_feature_columns,
                 class_num=1,
                 init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.sparse_feature_columns = sparse_feature_columns
        self.dense_feature_columns = dense_feature_columns

        self.embedding_dict = create_embedding_matrix(sparse_feature_columns)

        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        self.dense_bn = torch.nn.BatchNorm1d(sum(fc.dimension for fc in dense_feature_columns))

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(
                                torch.Tensor(
                                    sum(fc.dimension for fc in self.dense_feature_columns),
                                    class_num
                                ).to(device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

        self.bias = nn.Parameter(torch.zeros((class_num,)))
        self.class_num = class_num

    def forward(self, inputs):

        dense_value_list = [inputs[:, feat.index[0]:feat.index[1]]
                          for feat in self.dense_feature_columns]

        sparse_embedding_list = [
            self.embedding_dict[feat.name]( # 取出 embedding 表
                inputs[:, feat.index[0]:feat.index[1]].long())
            for feat in self.sparse_feature_columns]

        # sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
        #     X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
        #     feat in self.sparse_feature_columns]
        #
        # dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
        #                     self.dense_feature_columns]

        linear_logit = torch.zeros([inputs.shape[0], self.class_num]).to(dense_value_list[0].device)
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_tensor = self.dense_bn(torch.cat(dense_value_list, dim=-1))
            dense_value_logit = dense_tensor.matmul(self.weight)
            linear_logit += dense_value_logit
        linear_logit += self.bias
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
    def __init__(self,
                 sparse_feature_columns,
                 dense_feature_columns,
                 dnn_hidden_units=(256, 128, 64),
                 dnn_shared_layers=1,
                 dnn_activation='relu',
                 dnn_dropout=0,
                 class_num=1,
                 init_std=0.0001,
                 seed=1173,
                 device='cpu'):
        super(DeepFM, self).__init__()
        torch.manual_seed(seed)
        self.class_num = class_num
        self.sparse_feature_columns = sparse_feature_columns
        self.dense_feature_columns = dense_feature_columns
        self.group = Group(sparse_feature_columns, dense_feature_columns)

        self.embedding_dict = create_embedding_matrix(sparse_feature_columns)

        # PRE-TRANSFORM

        # FIELD-DNN
        # p_dim = sum(feat.dimension for feat in dense_feature_columns if feat.group == 'p')
        # self.p_dnn = nn.Linear(p_dim, 1)

        # UNION-DNN
        self.dnn_dense_fc = self.group.dense.get_fc('qp', 'p', 'default')
        self.dnn_sparse_fc = []
        dnn_input_dim = sum(feat.dimension for feat in self.dnn_dense_fc + self.dnn_sparse_fc)
        self.shared_dnn = DNN(dnn_input_dim, dnn_hidden_units[:dnn_shared_layers],
               activation=dnn_activation, dropout_rate=dnn_dropout)

        dnn_hidden_units += tuple([1])
        self.private_dnn = nn.ModuleList([
                DNN(dnn_hidden_units[dnn_shared_layers-1], dnn_hidden_units[dnn_shared_layers:],
                    activation=dnn_activation, dropout_rate=dnn_dropout)
                for _ in range(class_num)
        ])

        # FM
        self.linear_dense_fc = self.group.dense.get_fc('qp', 'p', 'default')
        self.linear = Linear(sparse_feature_columns, self.linear_dense_fc)
        self.fm = FM()

        # Output
        self.to(device)

    def pre_transform(self, inputs):

        sparse_embedding_dict = {feat.name:
            self.embedding_dict[feat.name](  # 取出 embedding 表
                inputs[:, feat.index[0]:feat.index[1]].long())
            for feat in self.sparse_feature_columns}

        dense_value_dict = {feat.name: inputs[:, feat.index[0]:feat.index[1]]
                            for feat in self.dense_feature_columns}

        feat_value_dict = {**dense_value_dict, **sparse_embedding_dict}
        return feat_value_dict

    def forward(self, inputs):
        feat_value_dict = self.pre_transform(inputs)
        # DNN
        sparse_embedding_list = [feat_value_dict[fc.name] for fc in self.dnn_sparse_fc]
        dnn_dense_input = torch.flatten(
            torch.cat([feat_value_dict[feat.name] for feat in self.dnn_dense_fc], dim=1),
            start_dim=1
        )
        if sparse_embedding_list:
            sparse_dnn_input = torch.flatten(
                torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
            dnn_input = torch.cat((sparse_dnn_input, dnn_dense_input), dim=-1)
        else:
            dnn_input = dnn_dense_input

        shared_feat = self.shared_dnn(dnn_input)
        logit = []
        for cls in range(self.class_num):
            _logit = self.private_dnn[cls](shared_feat)
            logit.append(_logit)
        logit = torch.cat(logit, dim=-1)

        # FM
        logit += self.linear(inputs)  # terms 1
        if len(sparse_embedding_list) > 0:  # 现在multi-task没考虑fm，要改的话应该有多少个task就建多少组fm
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)

        y_pred = torch.sigmoid(logit)
        return y_pred
