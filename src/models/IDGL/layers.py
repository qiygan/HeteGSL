from utils.util_funcs import cos_sim
import math
import torch.nn as nn
from torch.nn.parameter import Parameter


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
                 threshold=0.1):
        super(IDGL_AdjGenerator, self).__init__()
        self.threshold = threshold
        self.metric_layer_emb = nn.ModuleList()
        self.metric_layer_feat = nn.ModuleList()
        for i in range(num_head):
            self.metric_layer_feat.append(Metric_calc_layer(n_feat))
            self.metric_layer_emb.append(Metric_calc_layer(n_hidden))
        self.num_head = num_head

    def normalize_adj_torch(self, adj, mode='sc'):
        """Row-normalize sparse matrix"""
        rowsum = torch.sum(adj, 1)
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
        s = torch.zeros((h.shape[0], h.shape[0])).cuda()
        for i in range(self.num_head):
            if mode == 'feat':  # First time, use feat as emb.
                h_prime = self.metric_layer_feat[i](h)
            elif mode == 'emb':
                h_prime = self.metric_layer_emb[i](h)
            if torch.isnan(h_prime[1, 1]):
                print()
            s += cos_sim(h_prime, h_prime)

        s /= self.num_head
        # Remove negative values (Otherwise Nans are generated for negative values with power operation
        s = torch.where(s < self.threshold, torch.zeros_like(s), s)
        # check whether all positive:  sum(sum(s >= 0)).item() == 7333264
        # s = self.normalize_adj_torch(s)
        s = F.normalize(s, dim=1, p=1)  # Row normalization
        return s


# ! Previous version
#         s = torch.zeros((h.shape[0], h.shape[0])).cuda()
#         for i in range(self.num_head):
#             h_prime = self.metric_layer[i](h)
#             s += cos_sim(h_prime, h_prime)
#         s /= self.num_head
#         rst = torch.where(s < self.threshold, torch.zeros_like(s), s)
#         rst = F.normalize(rst, dim=1, p=1)  # Row normalization
#         return (1 - self.lambda_) * rst + self.lambda_adj


# pylint: disable= no-member, arguments-differ, invalid-name

import torch
import torch.nn as nn
import torch.nn.functional as F


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
