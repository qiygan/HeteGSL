from models.IDGL.GCN_with_emb import GCN_with_emb
from models.IDGL.layers import *


class IDGL(nn.Module):
    """
    Decode neighbors of input graph.
    """

    def __init__(self, nfeat, nhid, nclass, num_head=4, threshold=0.0, lamda=0.8):
        super(IDGL, self).__init__()
        self.GCN = GCN_with_emb(nfeat, nhid, nclass, dropout=0.5)
        self.GenAdjLayer = IDGL_GenAdjLayer(nhid, num_head, threshold, confidence=1 - lamda)

    def forward(self, x, adj, h, emb_only=False):
        """

        Args:
            x: input feature
            adj: adj of graph
            h: embedding
            use_ori_adj: use origin adj for the first time
            emb_only: return emb
        Returns:

        """
        if h is None:  # First time, use feat as emb.
            # ! Bug: feat_dim != emb_dim
            # adj_new = self.GenAdjLayer(x, adj)
            adj_new = adj
        else:
            adj_new = self.GenAdjLayer(h, adj)
        logits, h = self.GCN(x, adj_new)
        return logits, h, adj_new


import torch.nn as nn
import torch.nn.functional as F


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
        return F.log_softmax(x2, dim=1), x.detach()
        # ! Note that the detach() operation is vital, 下一个循环中我们要根据adj update的是metric向量而不是网络参数


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
