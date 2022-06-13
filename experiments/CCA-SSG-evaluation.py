#import sys, os
import os.path as osp
import numpy as np
sys.path.append('../../cca')

import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.nn import GAE, VGAE, APPNP
from torch_geometric.transforms import RandomLinkSplit
import torch_geometric.transforms as T
from torch_geometric.utils import (
    add_self_loops,
    negative_sampling,
    remove_self_loops,
     to_dense_adj
)

from models.basicVGNAE import *
from models.DeepVGAEX import *
from models.baseline_models import *
from aug import *
from models.cca import CCA_SSG
from models.ica_gnn import GraphICA, iVGAE, random_permutation
from train_utils import *

if args.split == "PublicSplit":
    transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(device)])
if args.split == "RandomSplit":
    transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(device), T.RandomNodeSplit(split="random",
                                                                                        num_train_per_class = 20,
                                                                                        num_val = 160,
file_path = os.getcwd() + args.result_file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path =  os.getcwd() + '/data'

if args.split == "PublicSplit":
    transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(device)])
    num_per_class  = 20
if args.split == "SupervisedSplit":
    transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(device), T.RandomNodeSplit()])

if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root='data/Planetoid', name=args.dataset, transform=transform)
    data = dataset[0]
if args.dataset in ['CS', 'physics']:
    dataset = Coauthor(path, args.dataset, transform=transform)
    data = dataset[0]
if args.dataset in ['computers', 'photo']:
    dataset = Amazon(path, args.dataset, transform=transform)
    data = dataset[0]

results = []

for training_rate in [0.1, 0.2, 0.4, 0.6, 0.8, 0.85]:
    val_ratio = (1.0 - training_rate) / 3
    test_ratio = (1.0 - training_rate) / 3 * 2
    transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
                                    is_undirected=True, split_labels=True)
    train_data, val_data, test_data = transform(data)
    for lambd in np.logspace(-7, 2, num=1, endpoint=True, base=10.0, dtype=None, axis=0):#np.logspace(-7, 2, num=10, endpoint=True, base=10.0, dtype=None, axis=0):
        for channels in [32, 64, 128, 256, 512]:
            for drop_rate_edge in [0.01, 0.05, 0.1, 0.2, 0,3, 0.4, 0.5, 0.7]:
                out_dim = channels
                hid_dim = [channels] * n_layers
                model = CCA_SSG(in_dim, hid_dim, out_dim, use_mlp=False)
                wd1 = 1e-4
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd1)
                for epoch in range(300):
                    model.train()
                    optimizer.zero_grad()
                    dfr = drop_rate_edge
                    new_data1 = random_aug(data, dfr , drop_rate_edge )
                    new_data2 = random_aug(data, dfr, drop_rate_edge)

                    z1, z2 = model(new_data1, new_data2)

                    c = torch.mm(z1.T, z2)
                    c1 =torch.mm(z1.T, z1)
                    c2 = torch.mm(z2.T, z2)

                    c = c / N
                    c1 = c1 / N
                    c2 = c2 / N

                    loss_inv = -torch.diagonal(c).sum()
                    iden = torch.tensor(np.eye(c.shape[0]))
                    loss_dec1 = (iden - c1).pow(2).sum()
                    loss_dec2 = (iden - c2).pow(2).sum()

                    loss = loss_inv + lambd * (loss_dec1 + loss_dec2)

                    loss.backward()
                    optimizer.step()

                    print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))

                # In[ ]:


                print("=== Evaluation ===")
                data = dataset[0]
                transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
                                            is_undirected=True, split_labels=True)
                train_data, val_data, test_data = transform(dataset[0])
                embeds = model.get_embedding(train_data)
                logreg = LogReg(embeds.shape[1], adj_train.shape[1])
                opt = torch.optim.Adam(logreg.parameters(), lr=args.lr, weight_decay=1e-4)

                _, res = edge_prediction(embeds.detach(), embeds.shape[1], train_data, test_data, val_data,
                                         lr=0.01, wd=1e-4,
                                         patience = args.patience, max_epochs=MAX_EPOCH_EVAL)
                val_ap, val_roc, test_ap, test_roc, train_ap, train_roc = res[-1][1], res[-1][2], res[-1][3], res[-1][4], res[-1][5], res[-1][6]
                _, nodes_res = node_prediction(embeds.detach(), dataset.num_classes, data.y, data.train_mask, data.test_mask,
                                            lr=0.01, wd=1e-4, patience = args.patience, max_epochs=MAX_EPOCH_EVAL)

                acc_train, acc = nodes_res[-1][2], nodes_res[-1][3]

                results += [['CCA', args.dataset, str(args.non_linear), args.normalize, args.lr, out_channels,
                                      training_rate, val_ratio, test_ratio, alpha, train_auc, train_ap,
                                      roc_auc, ap, acc_train, acc, epoch, 0, 0]]
                print(['CCA', args.dataset, str(args.non_linear), args.normalize, args.lr, out_channels,
                                      training_rate, val_ratio, test_ratio, alpha, train_auc, train_ap,
                                      roc_auc, ap, acc_train, acc, epoch, 0, 0])

                res1 = pd.DataFrame(results, columns=['model', 'dataset', 'non-linearity', 'normalize',  'lr', 'channels',
                                                    'train_rate','val_ratio', 'test_ratio', 'alpha',  'train_auc', 'train_ap',
                                                      'test_auc', 'test_ap', 'accuracy_train', 'accuracy_test', 'epoch',
                                                      'drop_edge_rate', 'drop_feat_rate'])
                res1.to_csv(file_path, index=False)
