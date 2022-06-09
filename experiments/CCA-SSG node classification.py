#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


# parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
# parser.add_argument('--use_mlp', action='store_true', default=False, help='Use MLP instead of GNN')


# In[6]:


graph, feat, labels, num_class, train_idx, val_idx, test_idx = load('cora')
graph = graph.remove_self_loop().add_self_loop()
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
graph = graph.remove_self_loop().add_self_loop()

embeds = model.get_embedding(graph, feat)

train_embs = embeds[train_idx]
val_embs = embeds[val_idx]
test_embs = embeds[test_idx]

label = labels

train_labels = label[train_idx]
val_labels = label[val_idx]
test_labels = label[test_idx]

train_feat = feat[train_idx]
val_feat = feat[val_idx]
test_feat = feat[test_idx] 

''' Linear Evaluation '''
logreg = LogReg(train_embs.shape[1], num_class)
lr2 = 1e-2
wd2 = 1e-4
opt = th.optim.Adam(logreg.parameters(), lr=lr2, weight_decay=wd2)

loss_fn = nn.CrossEntropyLoss()

best_val_acc = 0
eval_acc = 0
    
for epoch in range(2000):
    logreg.train()
    opt.zero_grad()
    logits = logreg(train_embs)
    preds = th.argmax(logits, dim=1)
    train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
    loss = loss_fn(logits, train_labels)
    loss.backward()
    opt.step()

    logreg.eval()
    with th.no_grad():
        val_logits = logreg(val_embs)
        test_logits = logreg(test_embs)

        val_preds = th.argmax(val_logits, dim=1)
        test_preds = th.argmax(test_logits, dim=1)

        val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
        test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            if test_acc > eval_acc:
                eval_acc = test_acc

    print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc, test_acc))
    print('Linear evaluation accuracy:{:.4f}'.format(eval_acc))


# In[ ]:




