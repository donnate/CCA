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
from torch_geometric.nn import GAE, VGAE, APPNP
from torch_geometric.transforms import RandomLinkSplit
import torch_geometric.transforms as T
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import add_self_loops

from sklearn.metrics import roc_auc_score, average_precision_score

from models.baseline_models import *
from aug import *
from models.cca import CCA_SSG
from train_utils import *
from similarities import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='VGNAE')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--n_experiments', type=int, default=10)
parser.add_argument('--tau', type=float, default=0.5) #
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--channels', type=int, default=512)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--non_linear', type=bool, default=True)
parser.add_argument('--scaling_factor', type=float, default=1.8)
parser.add_argument('--lr1', type=float, default=0.001)
parser.add_argument('--lr2', type=float, default=0.01)
parser.add_argument('--drop_rate_edge', type=float, default=0.2)
parser.add_argument('--drop_rate_feat', type=float, default=0.2)
parser.add_argument('--result_file', type=str, default="/results/SelfGCon_link_prediction")
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
    dataset = Planetoid(root='data/Planetoid', name=args.dataset, transform=transform)
    data = dataset[0]
if args.dataset in ['CS', 'physics']:
    dataset = Coauthor(path, args.dataset, transform=transform)
    data = dataset[0]
if args.dataset in ['computers', 'photo']:
    dataset = Amazon(path, args.dataset, transform=transform)
    data = dataset[0]

out_dim = args.channels
hid_dim = args.channels
n_layers = args.n_layers
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_dim = data.num_features
N = data.num_nodes


##### Train the CCA model
results =[]
# for training_rate in [0.1, 0.2, 0.4, 0.6, 0.8, 0.85]:
for training_rate in [0.85]:
    val_ratio = (1.0 - training_rate) / 3
    test_ratio = (1.0 - training_rate) / 3 * 2
    for exp in range(args.n_experiments):
        hid_dim = [args.channels] * args.n_layers
        model = CCA_SSG(data.num_features, hid_dim,
                        args.channels, use_mlp=False)
        lr1 = args.lr1
        wd1 = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr1, weight_decay=wd1)
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()

            dfr = args.drop_rate_feat
            der = args.drop_rate_edge

            new_data1 = random_aug(data, dfr, der)
            new_data2 = random_aug(data, dfr, der)

            z1, z2 = model(new_data1, new_data2)
            loss = cl_loss_fn(z1, z2, indices=None, tau=args.tau, type='selfG')

            loss.backward()
            optimizer.step()

            print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))

        print("=== Evaluation ===")
        transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
                                    is_undirected=True, split_labels=True)
        train_data, val_data, test_data = transform(data)
        embeds = model.get_embedding(train_data)
        adj_train = to_dense_adj(train_data.edge_index, max_num_nodes=N)
        adj_train = adj_train[0]
        weight_tensor, norm = compute_loss_para(adj_train)
        logreg = LogReg(embeds.shape[1], adj_train.shape[1])
        opt = torch.optim.Adam(logreg.parameters(),
                               lr=args.lr2, weight_decay=1e-4)

        loss_fn = F.binary_cross_entropy
        output_activation = nn.Sigmoid()

        best_val_roc = 0
        eval_roc = 0
        best_val_ap = 0
        eval_ap = 0

        for epoch in range(args.epochs):
            logreg.train()
            opt.zero_grad()
            logits_temp = logreg(embeds.detach())
            logits = output_activation(torch.mm(logits_temp, logits_temp.t()))
            loss = norm * loss_fn(logits.view(-1), adj_train.view(-1), weight = weight_tensor)
            loss.backward()
            opt.step()
            logreg.eval()
            with torch.no_grad():
                val_roc, val_ap = get_scores(val_data.pos_edge_label_index, val_data.neg_edge_label_index, logits)
                test_roc, test_ap = get_scores(test_data.pos_edge_label_index, test_data.neg_edge_label_index, logits)

                if val_roc >= best_val_roc:
                    best_val_roc = val_roc
                    if test_roc > eval_roc:
                        eval_roc = test_roc

                if val_ap >= best_val_ap:
                    best_val_ap = val_ap
                    if test_ap > eval_ap:
                        eval_ap = test_ap

            print('Epoch:{}, val_ap:{:.4f}, val_roc:{:4f}, test_ap:{:4f}, test_roc:{:4f}'.format(epoch, val_ap, val_roc, test_ap, test_roc))
            print('Linear evaluation AP:{:.4f}'.format(eval_ap))
            print('Linear evaluation ROC:{:.4f}'.format(eval_roc))
        results += [[exp, 'CCA', args.dataset, True, True, args.lr1, args.channels,
                                training_rate, val_ratio, test_ratio, 0, eval_roc, eval_ap, args.epochs, args.drop_rate_edge, args.drop_rate_feat]]
        res1 = pd.DataFrame(results, columns=['exp', 'model', 'dataset', 'non-linearity', 'normalize',  'lr', 'channels',
                                                'train_rate', 'val_ratio',
                                                'test_ratio', 'alpha', 'auc', 'ap', 'epoch', 'drop_edge_rate', 'drop_feat_rate'])
        res1.to_csv(file_path  +  args.model + str(args.non_linear) + "_norm" +  str(args.normalize) +  "_lr"+ str(args.lr1) +
                        '_channels' + str(args.channels) + "_dropedgerate" + str( args.drop_rate_edge) + "_dropfeatrate" + str( args.drop_rate_feat)+
                        ".csv", index=False)
