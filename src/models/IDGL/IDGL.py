from models.IDGL.layers import *


class IDGL(nn.Module):
    """
    Decode neighbors of input graph.
    """

    def __init__(self, args, nfeat, nclass):
        super(IDGL, self).__init__()
        self.GCN = GCN_with_emb(nfeat, args.num_hidden, nclass, dropout=args.dropout)
        self.GenAdjLayer = IDGL_AdjGenerator(nfeat, args.num_hidden, args.num_head, args.epsilon)
        self.lambda_ = args.lambda_
        self.eta = args.eta

    def forward(self, x, h, adj, adj_feat, mode):
        """

        Args:
            x: input feature
            adj: adj of graph
            h: embedding
            use_ori_adj: use origin adj for the first time
            emb_only: return emb
        Returns:

        """
        # ! Generate adj
        if mode == 'feat':
            adj_sim = self.GenAdjLayer(x, mode='feat')
            adj_agg = (1 - self.lambda_) * adj_sim + self.lambda_ * adj
        elif mode == 'emb':
            adj_sim = self.GenAdjLayer(h, mode='emb')
            adj_emb = (1 - self.lambda_) * adj_sim + self.lambda_ * adj
            adj_agg = self.eta * adj_emb + (1 - self.eta) * adj_feat

        # ! Aggregate using adj_agg
        logits, h = self.GCN(x, adj_agg)
        return logits, h, adj_sim, adj_agg


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
        emb = x.detach()
        x = F.dropout(x, self.dropout, training=self.training)
        # Classification Layer
        x2 = self.gc2(x, adj)
        if self.normalize:
            x2 = F.normalize(x2, p=2, dim=1)

        x2 = F.relu(x2)
        return F.log_softmax(x2, dim=1), emb
        # ! Note that the detach() operation is vital, 下一个循环中我们要根据adj update的是metric向量而不是网络参数
