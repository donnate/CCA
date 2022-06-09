#!/usr/bin/env python
# coding: utf-8
from model import CCA_SSG, LogReg
from aug import random_aug
from dataset import load

import numpy as np
import torch as th
import torch.nn as nn

import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, average_precision_score
from util import mask_test_edges_dgl
import dgl
import torch.nn.functional as F
import pdb

def compute_loss_para(adj):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = th.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm





def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = adj_rec.cpu()
    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


# In[5]:


graph, feat, labels, num_class, train_idx, val_idx, test_idx = load('cora')
adj_orig = graph.adj().to_dense()
train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_dgl(graph, adj_orig)

# create train graph
train_edge_idx = th.tensor(train_edge_idx)
train_graph = dgl.edge_subgraph(graph, train_edge_idx, preserve_nodes=True)

# add self loop
#train_graph = dgl.remove_self_loop(train_graph)
#train_graph = dgl.add_self_loop(train_graph)
#n_edges = train_graph.number_of_edges()
#adj = train_graph.adjacency_matrix().to_dense()

# normalization
#degs = train_graph.in_degrees().float()
#norm = th.pow(degs, -0.5)
#norm[th.isinf(norm)] = 0
#train_graph.ndata['norm'] = norm.unsqueeze(1)


# In[6]:


# graph, feat, labels, num_class, train_idx, val_idx, test_idx = load('cora')

in_dim = feat.shape[1]

hid_dim = 512
out_dim = 512
n_layers = 2

model = CCA_SSG(in_dim, hid_dim, out_dim, n_layers, use_mlp=False)
lr1 = 1e-3
wd1 = 0
optimizer = th.optim.Adam(model.parameters(), lr=lr1, weight_decay=wd1)

N = graph.number_of_nodes()

for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    dfr = 0.2
    der = 0.2

    graph1, feat1 = random_aug(graph, feat, 0.2, 0.2)
    graph2, feat2 = random_aug(graph, feat, 0.2, 0.2)

    graph1 = graph1.add_self_loop()
    graph2 = graph2.add_self_loop()

    z1, z2 = model(graph1, feat1, graph2, feat2)

    c = th.mm(z1.T, z2)
    c1 = th.mm(z1.T, z1)
    c2 = th.mm(z2.T, z2)

    c = c / N
    c1 = c1 / N
    c2 = c2 / N

    loss_inv = -th.diagonal(c).sum()
    iden = th.tensor(np.eye(c.shape[0]))
    loss_dec1 = (iden - c1).pow(2).sum()
    loss_dec2 = (iden - c2).pow(2).sum()

    lambd = 1e-3

    loss = loss_inv + lambd * (loss_dec1 + loss_dec2)

    loss.backward()
    optimizer.step()

    print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))


# In[7]:


print("=== Evaluation ===")
graph = train_graph.remove_self_loop().add_self_loop()
adj = graph.adj().to_dense()

weight_tensor, norm = compute_loss_para(adj)

embeds = model.get_embedding(graph, feat)

# loss_fn = F.binary_cross_entropy
loss_fn = F.binary_cross_entropy
output_activation = nn.Sigmoid()
logreg = LogReg(embeds.shape[1], adj.shape[1])

logits_temp = logreg(embeds)
logits = output_activation(th.mm(logits_temp, logits_temp.t()))

val_roc, val_ap = get_scores(val_edges, val_edges_false, logits)
test_roc, test_ap = get_scores(test_edges, test_edges_false, logits)
print(test_roc, test_ap)


# In[ ]:


print("=== Evaluation ===")
graph = train_graph.remove_self_loop().add_self_loop()
adj = graph.adj().to_dense()

weight_tensor, norm = compute_loss_para(adj)

embeds = model.get_embedding(graph, feat)

''' Linear Evaluation '''
logreg = LogReg(embeds.shape[1], adj.shape[1])
lr2 = 1e-2
wd2 = 1e-4
opt = th.optim.Adam(logreg.parameters(), lr=lr2, weight_decay=wd2)

loss_fn = F.binary_cross_entropy
output_activation = nn.Sigmoid()

best_val_roc = 0
eval_roc = 0
best_val_ap = 0
eval_ap = 0

for epoch in range(2000):
    logreg.train()
    opt.zero_grad()
    logits_temp = logreg(embeds)
    logits = output_activation(th.mm(logits_temp, logits_temp.t()))

    # pdb.set_trace()
    loss = norm*loss_fn(logits.view(-1), adj.view(-1), weight = weight_tensor)
    loss.backward()
    opt.step()

    logreg.eval()
    with th.no_grad():
        val_roc, val_ap = get_scores(val_edges, val_edges_false, logits)
        test_roc, test_ap = get_scores(test_edges, test_edges_false, logits)

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


# In[ ]:
