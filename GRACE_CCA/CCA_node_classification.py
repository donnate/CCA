import os
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

from model import *
from aug import *
# from aug_gae import gae_aug # delete + add

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='CCA')
parser.add_argument('--dataset', type=str, default='Photo')
parser.add_argument('--split', type=str, default='RandomSplit')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--n_experiments', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=2) #CiteSeer: 1, Rest: 2
parser.add_argument('--channels', type=int, default=512)
parser.add_argument('--lambd', type=float, default=1e-3)
parser.add_argument('--lr1', type=float, default=1e-3)
parser.add_argument('--lr2', type=float, default=1e-2)
parser.add_argument('--wd2', type=float, default=1e-4)
parser.add_argument('--drop_rate_edge', type=float, default=0.2)
parser.add_argument('--drop_rate_feat', type=float, default=0.2)
parser.add_argument('--result_file', type=str, default="/results/CCA_node_classification")
args = parser.parse_args()

file_path = os.getcwd() + args.result_file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.split == "PublicSplit":
    transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(device)])
if args.split == "RandomSplit":
    transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(device), T.RandomNodeSplit(split="random", 
                                                                                        num_train_per_class = 50,
                                                                                        num_val = 400,
                                                                                        num_test = 3200 )])
if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root='Planetoid', name=args.dataset, transform=transform)
    data = dataset[0]
if args.dataset in ['cs', 'physics']:
    dataset = Coauthor(args.dataset, 'public', transform=transform)
    data = dataset[0]
if args.dataset in ['Computers', 'Photo']:
    dataset = Amazon("/Users/ilgeehong/Desktop/GRACE_CCA/", args.dataset, transform=transform)
    data = dataset[0]

# pdb.set_trace()
train_idx = data.train_mask 
val_idx = data.val_mask 
test_idx = data.test_mask  

in_dim = data.num_features
hid_dim = args.channels
out_dim = args.channels
n_layers = args.n_layers

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_class = int(data.y.max().item()) + 1 #data.num_classes
N = data.num_nodes

##### Train the CCA model #####
print("=== train CCA model ===")
results =[]
for exp in range(args.n_experiments): 
    model = CCA_SSG(in_dim, hid_dim, out_dim, n_layers, use_mlp=False)
    lr1 = args.lr1
    wd1 = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr1, weight_decay=wd1)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        dfr = args.drop_rate_feat
        der = args.drop_rate_edge

        new_data1 = random_aug(data, dfr, der)
        new_data2 = random_aug(data, dfr, der)

        z1, z2 = model(new_data1, new_data2)
        
        c = torch.mm(z1.T, z2)
        c1 = torch.mm(z1.T, z1)
        c2 = torch.mm(z2.T, z2)

        c = c / N
        c1 = c1 / N
        c2 = c2 / N

        loss_inv = - torch.diagonal(c).sum()
        iden = torch.tensor(np.eye(c.shape[0]))
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()

        lambd = args.lambd

        loss = loss_inv + lambd * (loss_dec1 + loss_dec2)

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
    wd2 = args.wd2
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
    results += [['CCA', args.dataset, True, True, args.lr1, args.channels, args.epochs, args.drop_rate_edge, args.drop_rate_feat, eval_acc]]
    res1 = pd.DataFrame(results, columns=['model', 'dataset', 'non-linearity', 'normalize',  'lr', 'channels', 'epoch', 'drop_edge_rate', 'drop_feat_rate', 'accuracy'])
    res1.to_csv(file_path + "_" + args.dataset + ".csv", index=False)

