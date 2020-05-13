"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""
import sys
# sys.path.append('../')
from utils.util_funcs import shell_init,seed_init

#
# shell_init(server='S5', gpu_id=3)
# shell_init(server='S5', gpu_id=5)
# shell_init(server='S5', gpu_id=4, f_prefix='src/models/IDGL')
import argparse
import networkx as nx
import time
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from models.IDGL import IDGL
import numpy as np
import torch


def graph_reg_loss(adj, features):
    return None


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask, adj=None, h=None):
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(features, adj, h)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


def train_idgl(args):
    # load and preprocess dataset

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
        torch.cuda.set_device(args.gpu)
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
    model = IDGL(num_feats, args.num_hidden, n_classes, args.num_head, args.ori_ratio)

    print(model)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    h = None
    adj = g.adjacency_matrix().cuda()
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits, h, adj_new = model(features, adj, h)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        val_acc = evaluate(model, features, labels, val_mask, adj, h)

        print("Epoch {:05d} | Time(s) {:.4f} | r {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask, adj)
    print("Test Accuracy {:.4f}".format(acc))
    res_dict = {'parameters': args, 'res': acc}
    return res_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT_dgl')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--dataset", type=str, default='cora',
                        help="dataset to use")
    parser.add_argument("--num_head", type=int, default=8,
                        help="number of metric heads")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--seed", type=int, default=0,
                        help="training seed")
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--ori_ratio", type=float, default=0.8,
                        help="ratio of retain the original graph")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--out_path', type=str, default='/results/IDGL/',
                        help="path of results")
    args = parser.parse_args()
    print(args)
    res_dict = train_idgl(args)
