from models.IDGL.layers import *


class IDGL(nn.Module):
    """
    Decode neighbors of input graph.
    """

    def __init__(self, args, nfeat, nclass, dev):
        super(IDGL, self).__init__()
        # self.GCN = GCN_with_emb(nfeat, args.num_hidden, nclass, dropout=args.dropout)
        self.GCN = GCN2(nfeat, args.num_hidden, nclass, dropout=args.dropout)
        self.GenAdjLayer = IDGL_AdjGenerator(nfeat, args.num_hidden, args.num_head, args.epsilon, dev)
        self.lambda_ = args.lambda_
        self.eta = args.eta

    def forward(self, x, h, adj_ori, adj_feat, mode):
        """

        Args:
            x: input feature
            h: embedding
            adj_ori: adj of graph
            adj_feat: adj generated by feature
            mode: gen adj using 'feat' or 'emb'
        Returns:
            logits: predicted labels
            h: embedding generated by GCN
            adj_sim:
            adj_agg:

        """
        # ! Generate adj
        if mode == 'feat':
            adj_sim = self.GenAdjLayer(x, mode='feat')
            adj_agg = F.normalize(adj_sim, dim=1, p=1)  # Row normalization
            adj_agg = self.lambda_ * adj_ori + (1 - self.lambda_) * adj_agg
        elif mode == 'emb':
            adj_sim = self.GenAdjLayer(h, mode='emb')
            adj_agg = F.normalize(adj_sim, dim=1, p=1)  # Row normalization
            adj_agg = self.lambda_ * adj_ori + (1 - self.lambda_) * adj_agg
            # combine feat and emb sim mat
            adj_agg = self.eta * adj_agg + (1 - self.eta) * adj_feat

        # ! Aggregate using adj_agg
        logits, h = self.GCN(x, adj_agg)
        return logits, h, adj_sim, adj_agg


import torch.nn as nn
import torch.nn.functional as F


class GCN_with_emb(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.2, activation='Relu'):
        super(GCN_with_emb, self).__init__()
        self.normalize = False
        self.attention = False
        if activation == 'Relu':
            self.activation = F.relu
        elif activation == 'Elu':
            self.activation = F.elu
        self.gc1 = GraphConvolution(nfeat, nhid, bias=False)
        self.gc2 = GraphConvolution(nhid, nclass, bias=False)
        self.dropout = dropout

    def forward(self, x, adj):
        # Embedding Layer
        x = self.gc1(x, adj)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        x = self.activation(x)
        emb = x.detach()
        x = F.dropout(x, self.dropout, training=self.training)
        # Classification Layer
        x2 = self.gc2(x, adj)
        if self.normalize:
            x2 = F.normalize(x2, p=2, dim=1)

        x2 = F.relu(x2)
        return F.log_softmax(x2, dim=1), emb


class GCN2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN2, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        emb = x.detach()
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1), emb
