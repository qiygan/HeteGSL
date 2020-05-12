import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GCN_with_emb(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.2):
        super(GCN_with_emb, self).__init__()
        self.normalize = False
        self.attention = False

        self.gc1 = GraphConvolution(nfeat, nhid, bias=False)
        self.gc2 = GraphConvolution(nhid, nclass, bias=False)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
         # Classification Layer
        x2 = self.gc2(x, adj)
        if self.normalize:
            x2 = F.normalize(x2, p=2, dim=1)

        x2 = F.relu(x2)
        return F.log_softmax(x2, dim=1), x


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.2):
        super(GCN, self).__init__()
        self.normalize = False
        self.attention = False

        self.gc1 = GraphConvolution(nfeat, nhid, bias=False)
        self.gc2 = GraphConvolution(nhid, nclass, bias=False)
        self.dropout = dropout

    def forward(self, x, adj, emb_only=False):
        x = self.gc1(x, adj)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        if emb_only:
            return x
        x = F.relu(x)
        return F.log_softmax(x, dim=1)


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
        if len(adj._values()) == 0:
            if self.cuda:
                return torch.zeros(adj.shape[0], self.out_features).cuda()
            else:
                return torch.zeros(adj.shape[0], self.out_features)
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
