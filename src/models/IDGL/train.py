"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""
from utils.util_funcs import shell_init,seed_init

#
# shell_init(server='S5', gpu_id=3)
shell_init(server='S5', gpu_id=5)
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


def evaluate(model, features, labels, mask, model_name='GAT_dgl', adj=None, h=None):
    model.eval()
    with torch.no_grad():
        if model_name == 'GAT_dgl':
            logits = model(features)
        elif model_name == 'GCN':
            logits = model(features, adj)
        elif model_name == 'IDGL':
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
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = IDGL(num_feats, args.num_hidden, n_classes, args.num_head, args.lamda)

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
        if args.model == 'GAT_dgl':
            logits = model(features)
        elif args.model == 'GCN':
            logits = model(features, adj)
        elif args.model == 'IDGL':
            logits, h, adj_new = model(features, adj, h)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        if args.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        else:
            val_acc = evaluate(model, features, labels, val_mask, args.model, adj, h)

        print("Epoch {:05d} | Time(s) {:.4f} | r {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask, args.model, adj)
    print("Test Accuracy {:.4f}".format(acc))
    res_dict = {'parameters': args, 'res': acc}
    return res_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT_dgl')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")

    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    args = parser.parse_args()
    args.dataset = 'cora'
    dataset = 'cora'
    args.model = 'IDGL'
    args.lr = 0.01
    args.weight_decay = 5e-4
    args.num_head = 4
    args.lamda = 0.8
    args.epochs = 300
    args.seed = 2020
    print(args)
    res_dict = train_idgl(args)
