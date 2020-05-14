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


class IDGL_GenAdjLayer(nn.Module):
    """
    Decode neighbors of input graph.
    """

    def __init__(self, nhid,
                 num_head=4,
                 threshold=0.1, #Cora 0.0
                 confidence=0.1):
        super(IDGL_GenAdjLayer, self).__init__()
        self.threshold = threshold
        self.metric_layer = nn.ModuleList()
        for i in range(num_head):
            self.metric_layer.append(Metric_calc_layer(nhid))
        self.num_head = num_head
        self.confidence = confidence

    def forward(self, h, adj):
        """

        Args:
            h: node_num * hidden_dim
            adj: original adj

        Returns:

        """
        s = torch.zeros((h.shape[0],h.shape[0])).cuda()
        for i in range(self.num_head):
            h_prime = self.metric_layer[i](h)
            s += cos_sim(h_prime, h_prime)
        s/= self.num_head
        rst = torch.where(s < self.threshold,torch.zeros_like(s), s )
        rst = F.normalize(rst,dim=1,p=1) # Row normalization
        return self.confidence* rst + (1-self.confidence)*adj


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
