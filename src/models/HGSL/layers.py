from utils.util_funcs import cos_sim
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class Metric_calc_layer(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, nhid))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h):
        return h * self.weight


class HGSL_AdjGenerator(nn.Module):
    """
    Decode neighbors of input graph.
    """

    def __init__(self, n_feat, n_hidden,
                 num_head=4,
                 threshold=0.1, dev=True):
        super(HGSL_AdjGenerator, self).__init__()
        self.threshold = threshold
        self.metric_layer_emb = nn.ModuleList()
        self.metric_layer_feat = nn.ModuleList()
        for i in range(num_head):
            self.metric_layer_feat.append(Metric_calc_layer(n_feat))
            self.metric_layer_emb.append(Metric_calc_layer(n_hidden))
        self.num_head = num_head
        self.dev = dev

    def forward(self, h, mode='emb'):
        """

        Args:
            h: node_num * hidden_dim/feat_dim
            mode: whether h is emb or feat
        Returns:

        """
        # TODO Zero mat, necessary?
        s = torch.zeros((h.shape[0], h.shape[0])).to(self.dev)
        zero_lines = torch.nonzero(torch.sum(h, 1) == 0)
        if len(zero_lines) > 0:
            # raise ValueError('{} zero lines in {}s!\nZero lines:{}'.format(len(zero_lines), mode, zero_lines))
            h[zero_lines, :] += 1e-8
        for i in range(self.num_head):
            if mode == 'feat':  # First time, use feat as emb.
                weighted_h = self.metric_layer_feat[i](h)
            elif mode == 'emb':
                weighted_h = self.metric_layer_emb[i](h)
            s += cos_sim(weighted_h, weighted_h)
            if torch.min(cos_sim(weighted_h, weighted_h) < 0) < 0:
                print('!' * 20 + '\n')
        s /= self.num_head
        # Remove negative values (Otherwise Nans are generated for negative values with power operation
        s = torch.where(s < self.threshold, torch.zeros_like(s), s)
        return s


class GraphConvolution(nn.Module):  # GCN AHW
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.spmm(inputs, self.weight)  # HW in GCN
        output = torch.spmm(adj, support)  # AHW
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
