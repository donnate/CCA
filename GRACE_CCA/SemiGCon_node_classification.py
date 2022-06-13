import sys, os
sys.path.append('../../cca')
import os.path as osp
import argparse
import pandas as pd
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid, Coauthor, Amazon
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import add_self_loops


from models.baseline_models import *
from aug import *
from models.cca import CCA_SSG
from train_utils import *
from similarities import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GRACE')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--split', type=str, default='PublicSplit')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--n_experiments', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=2) #CiteSeer: 1, Rest: 2
parser.add_argument('--channels', type=int, default=512) #512
parser.add_argument('--tau', type=float, default=0.5) #
parser.add_argument('--lr1', type=float, default=1e-3) #
parser.add_argument('--lr2', type=float, default=1e-2)
parser.add_argument('--wd2', type=float, default=1e-4)
parser.add_argument('--drop_rate_edge', type=float, default=0.4)
parser.add_argument('--drop_rate_feat', type=float, default=0.4)
parser.add_argument('--result_file', type=str, default="/SemGCon/results/SemGCon_node_classification")
args = parser.parse_args()

file_path = os.getcwd() + args.result_file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path =  os.getcwd() + '/data'

if args.split == "PublicSplit":
    transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(device)])
    num_per_class  = 20
if args.split == "SupervisedSplit":
    transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(device), T.RandomNodeSplit()])

if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root=path + '/Planetoid', name=args.dataset, transform=transform)
    data = dataset[0]
if args.dataset in ['CS', 'physics']:
    dataset = Coauthor(path, args.dataset, transform=transform)
    data = dataset[0]
if args.dataset in ['computers', 'photo']:
    dataset = Amazon(path, args.dataset, transform=transform)
    data = dataset[0]
print(data)

train_idx = data.train_mask
val_idx = data.val_mask
test_idx = data.test_mask

in_dim = data.num_features
hid_dim = args.channels
out_dim = args.channels
n_layers = args.n_layers

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_class = int(data.y.max().item()) + 1
N = data.num_nodes

class_idx = []
for c in range(num_class):
    index = (data.y == c) * train_idx
    class_idx.append(index)
class_idx = torch.stack(class_idx).bool()
pos_idx = class_idx[data.y]
# class_idx = torch.BoolTensor([class_idx])

##### Train the GRACE model #####
print("=== train GRACE model ===")
results =[]
for exp in range(args.n_experiments):
    hid_dim = [args.channels] * args.n_layers
    model = CCA_SSG(data.num_features, hid_dim,
                    args.channels, use_mlp=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=0)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        dfr = args.drop_rate_feat
        der = args.drop_rate_edge

        new_data1 = random_aug(data, dfr, der)
        new_data2 = random_aug(data, dfr, der)

        z1, z2 = model(new_data1, new_data2)
        # pdb.set_trace()
        loss =  cl_loss_fn(z1, z2, pos_idx,
                       mean = True,
                       tau = args.tau, type='semG', num_per_class=num_per_class)

        loss.backward()
        optimizer.step()

        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))
        # add self loop # data.edge_index

    embeds = model.get_embedding(data)

    train_embs = embeds[train_idx]
    val_embs = embeds[val_idx]
    test_embs = embeds[test_idx]

    label = data.y
    feat = data.x

    train_labels = label[train_idx]
    val_labels = label[val_idx]
    test_labels = label[test_idx]

    train_feat = feat[train_idx]
    val_feat = feat[val_idx]
    test_feat = feat[test_idx]

    ''' Linear Evaluation '''
    logreg = LogReg(train_embs.shape[1], num_class)
    lr2 = args.lr2
    wd2 = 1e-4
    opt = torch.optim.Adam(logreg.parameters(), lr=lr2, weight_decay=wd2)

    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    eval_acc = 0

    for epoch in range(2000):
        logreg.train()
        opt.zero_grad()
        logits = logreg(train_embs)
        preds = torch.argmax(logits, dim=1)
        train_acc = torch.sum(preds == train_labels).float() / train_labels.shape[0]
        loss = loss_fn(logits, train_labels)
        loss.backward()
        opt.step()

        logreg.eval()
        with torch.no_grad():
            val_logits = logreg(val_embs)
            test_logits = logreg(test_embs)

            val_preds = torch.argmax(val_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)

            val_acc = torch.sum(val_preds == val_labels).float() / val_labels.shape[0]
            test_acc = torch.sum(test_preds == test_labels).float() / test_labels.shape[0]

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                if test_acc > eval_acc:
                    eval_acc = test_acc

        print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc, test_acc))
        print('Linear evaluation accuracy:{:.4f}'.format(eval_acc))
    results += [['GRACE', args.dataset, args.epochs, args.n_layers, args.tau, args.lr1, args.lr2, args.wd2, args.channels, args.drop_rate_edge, args.drop_rate_feat, eval_acc]]
    res1 = pd.DataFrame(results, columns=['model', 'dataset', 'epochs', 'layers', 'tau', 'lr1', 'lr2', 'wd2', 'channels', 'drop_edge_rate', 'drop_feat_rate', 'accuracy'])
    res1.to_csv(file_path + "_" + args.dataset +  ".csv", index=False)
