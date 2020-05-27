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


class IDGL_AdjGenerator(nn.Module):
    """
    Decode neighbors of input graph.
    """

    def __init__(self, n_feat, n_hidden,
                 num_head=4,
                 threshold=0.1, cuda=True):
        super(IDGL_AdjGenerator, self).__init__()
        self.threshold = threshold
        self.metric_layer_emb = nn.ModuleList()
        self.metric_layer_feat = nn.ModuleList()
        for i in range(num_head):
            self.metric_layer_feat.append(Metric_calc_layer(n_feat))
            self.metric_layer_emb.append(Metric_calc_layer(n_hidden))
        self.num_head = num_head
        self.cuda = cuda

    def normalize_adj_torch(self, adj, mode='sc'):
        """Row-normalize sparse matrix"""
        rowsum = torch.sum(adj, 1)
        # r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()  # Abandoned, gen nan for zero values.
        r_inv_sqrt = torch.pow(rowsum + 1e-8, -0.5).flatten()
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        normalized_adj = torch.mm(torch.mm(r_mat_inv_sqrt, adj), r_mat_inv_sqrt)
        return normalized_adj

    def forward(self, h, mode='emb'):
        """

        Args:
            h: node_num * hidden_dim/feat_dim
            mode: whether h is emb or feat
        Returns:

        """
        # TODO Zero mat, necessary?
        if self.cuda:
            s = torch.zeros((h.shape[0], h.shape[0])).cuda()
        else:
            s = torch.zeros((h.shape[0], h.shape[0]))
        # zero_lines = torch.where(torch.sum(h, 1) == 0)[0]
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
        s /= self.num_head
        # Remove negative values (Otherwise Nans are generated for negative values with power operation
        s = torch.where(s < self.threshold, torch.zeros_like(s), s)
        # s = self.normalize_adj_torch(s)
        s = F.normalize(s, dim=1, p=1)  # Row normalization
        return s


class GraphConvolution(nn.Module):  # GCN AHW
    def __init__(self, in_features, out_features, cuda=True, bias=True):
        super(GraphConvolution, self).__init__()
        self.cuda = cuda
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
