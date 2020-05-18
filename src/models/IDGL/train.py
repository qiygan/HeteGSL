import sys
import os
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path.split('/models')[0]
sys.path.append(root_path)
from utils.util_funcs import *

#
# shell_init(server='S5', gpu_id=3)
# shell_init(server='S5', gpu_id=2)
# shell_init(server='S5', gpu_id=4, f_prefix='src/models/IDGL')
shell_init(server='Ali', gpu_id=3, f_prefix='src/models/IDGL')
import argparse
import networkx as nx
import time
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from models.IDGL import IDGL
import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)


def normalize_adj_torch(adj):
    """Row-normalize sparse matrix"""
    rowsum = torch.sum(adj, 1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    normalized_adj = torch.mm(torch.mm(r_mat_inv_sqrt, adj), r_mat_inv_sqrt)
    return normalized_adj


def graph_reg_loss(args, adj, features):
    return torch.Tensor(0).cuda()


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask, adj):
    model.eval()
    with torch.no_grad():
        logits, _ = model.GCN(features, adj)  # Fixme
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


def iter_condition(args, adj_prev, adj_new, ori_adj_norm, t):
    cond1 = False
    if t == 0 or \
            torch.norm(adj_new - adj_prev, p=2) > args.delta * ori_adj_norm:
        cond1 = True
    cond2 = t < args.T
    return cond1 and cond2


def cal_loss(args, cla_loss, logits, train_mask, labels, adj, features):
    l_pred = cla_loss(logits[train_mask], labels[train_mask])
    # l_graph = graph_reg_loss(args, adj, features)
    # loss = l_pred + l_graph
    # return loss
    return l_pred  # Fixme


# load and preprocess dataset
def train_idgl(args):
    data = load_data(args)
    seed_init(seed=args.seed)

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    num_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d 
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        # torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    n_edges = g.number_of_edges()
    # create model
    model = IDGL(args, num_feats, n_classes)

    print(model)
    if cuda:
        model.cuda()
    cla_loss = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    h = None
    adj = g.adjacency_matrix().cuda()
    adj = normalize_adj_torch(adj.to_dense())
    ori_adj_norm = torch.norm(adj, p=2)
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        t, adj_sim_prev = 0, None
        logits, h, adj_sim, adj_feat = model(features, h=None, adj=adj, adj_feat=None, mode='feat')
        loss_adj_feat = cal_loss(args, cla_loss, logits, train_mask, labels, adj_sim, features)
        loss_list = [loss_adj_feat]

        while iter_condition(args, adj_sim_prev, adj_sim, ori_adj_norm, t):
            t += 1
            adj_sim_prev = adj_sim.detach()
            logits, h, adj_sim, adj_agg = model(features, h, adj, adj_feat, mode='emb')
            loss_adj_emb = cal_loss(args, cla_loss, logits, train_mask, labels, adj_sim, features)
            loss_list.append(loss_adj_emb)
        loss = torch.mean(torch.stack(loss_list))
        optimizer.zero_grad()

        # 反向传播时：在求导时开启侦测
        with torch.autograd.detect_anomaly():
            loss.backward()
        # print(model.GenAdjLayer.metric_layer_emb[0].weight) # normal
        optimizer.step()
        # print(model.GenAdjLayer.metric_layer_emb[0].weight) # ! NAN
        # print(model.GenAdjLayer.metric_layer_feat[0].weight) # ! NAN

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        val_acc = evaluate(model, features, labels, val_mask, adj)

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask, adj)
    print("Test Accuracy {:.4f}".format(acc))
    res_dict = {'parameters': args.__dict__, 'res': {'acc': '{:.4f}'.format(acc)}}
    return res_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT_dgl')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--dataset", type=str, default='cora',
                        help="dataset to use")
    parser.add_argument("--num_hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--seed", type=int, default=0,
                        help="training seed")
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help="weight decay")  # Fixme
    # Model configs
    parser.add_argument("--lambda_", type=float, default=0.9,
                        help="ratio of retain the original graph")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--num_head", type=int, default=3,
                        help="number of metric heads, m in paper")
    parser.add_argument("--delta", type=float, default=4e-5)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--out_path', type=str, default='results/IDGL/',
                        help="path of results")
    parser.add_argument('--exp_name', type=str, default='IDGL_Results.txt',
                        help="name of the experiment")
    args = parser.parse_args()

    # !  Debug Target: No graph reg_loss training, 0.8020
    args.T = 0 # Fixme
    args.lambda_= 1
    print(args)
    res_dict = train_idgl(args)
    write_dict(res_dict, args.out_path + args.exp_name)
