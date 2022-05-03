#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, _, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)

        x = self.act_fn(x)
        x = self.layer2(x)

        return x
    
    
class MLP2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP2, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn1 = nn.ReLU()
        self.act_fn2 = nn.Softmax()

    def forward(self, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act_fn1(x) 
        x = self.layer2(x)
        # x = self.act_fn2(x)

        return x


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x):

        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](graph, x))
        x = self.convs[-1](graph, x)

        return x

class TOCCA1(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_classes, n_layers, use_mlp = False): # add n_classes
        super().__init__()
        if not use_mlp:
            self.backbone1 = GCN(in_dim, hid_dim, out_dim, n_layers)
            self.backbone2 = GCN(in_dim, hid_dim, out_dim, n_layers)
            # self.backbone3 = MLP2(2*out_dim, hid_dim, n_classes)
            self.backbone3 = MLP2(out_dim, hid_dim, n_classes) # not concat, mean
        else:
            self.backbone1 = MLP(in_dim, hid_dim, out_dim)
            self.backbone2 = MLP(in_dim, hid_dim, out_dim)

    #def get_embedding(self, graph, feat):
    #    out = self.backbone(graph, feat)
    #    return out.detach()

    def forward(self, graph1, feat1, graph2, feat2):
        h1 = self.backbone1(graph1, feat1)
        h2 = self.backbone2(graph2, feat2)

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        # z_temp = torch.cat([z1,z2],1)
        z = torch.add(z1,z2) / 2 # not concat, mean
        
        pred_y = self.backbone3(z)
        
        return z1, z2, pred_y
    
class TOCCA(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_classes, n_layers, use_mlp = False): # add n_classes
        super().__init__()
        if not use_mlp:
            self.backbone1 = GCN(in_dim, hid_dim, out_dim, n_layers)
            self.backbone2 = GCN(in_dim, hid_dim, out_dim, n_layers)
            # self.backbone3 = MLP2(2*out_dim, hid_dim, n_classes)
            self.backbone3 = MLP2(out_dim, hid_dim, n_classes)
        else:
            self.backbone1 = MLP(in_dim, hid_dim, out_dim)
            self.backbone2 = MLP(in_dim, hid_dim, out_dim)

    #def get_embedding(self, graph, feat):
    #    out = self.backbone(graph, feat)
    #    return out.detach()

    def forward(self, graph1, feat1, graph2, feat2):
        h1 = self.backbone1(graph1, feat1)
        h2 = self.backbone1(graph2, feat2)

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        # z = torch.cat([z1,z2],1)
        z = torch.add(z1,z2) / 2 # not concat, mean
        
        pred_y = self.backbone3(z)
        
        return z1, z2, pred_y
    
class TOCCA_link(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_adj, n_layers, use_mlp = False): # add n_classes
        super().__init__()
        if not use_mlp:
            self.backbone1 = GCN(in_dim, hid_dim, out_dim, n_layers)
            self.backbone2 = GCN(in_dim, hid_dim, out_dim, n_layers)
            # self.backbone3 = MLP2(2*out_dim, hid_dim, n_classes)
            self.backbone3 = LogReg(out_dim, n_adj) # not concat, mean
        else:
            self.backbone1 = MLP(in_dim, hid_dim, out_dim)
            self.backbone2 = MLP(in_dim, hid_dim, out_dim)
        
    def forward(self, graph1, feat1, graph2, feat2):
        h1 = self.backbone1(graph1, feat1)
        h2 = self.backbone2(graph2, feat2)

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        # z_temp = torch.cat([z1,z2],1)
        z = torch.add(z1,z2) / 2 # not concat, mean
             
        logits = self.backbone3(z)
        
        return z1, z2, logits
    
    
class TOCCA3(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_classes, n_layers, use_mlp = False): # add n_classes
        super().__init__()
        if not use_mlp:
            self.backbone1 = GCN(in_dim, hid_dim, out_dim, n_layers)
            self.backbone2 = GCN(in_dim, hid_dim, out_dim, n_layers)
            # self.backbone3 = MLP2(2*out_dim, hid_dim, n_classes)
            self.backbone3 = MLP2(out_dim, hid_dim, n_classes)
        else:
            self.backbone1 = MLP(in_dim, hid_dim, out_dim)
            self.backbone2 = MLP(in_dim, hid_dim, out_dim)

    #def get_embedding(self, graph, feat):
    #    out = self.backbone(graph, feat)
    #    return out.detach()

    def forward(self, graph1, feat1, graph2, feat2):
        h1 = self.backbone1(graph1, feat1)
        h2 = self.backbone2(graph2, feat2)

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)
        
        pred1 = self.backbone3(z1)
        pred2 = self.backbone3(z2)
        
        return z1, z2, pred1, pred2
    

class TOCCA4(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_classes, n_layers, use_mlp = False): # add n_classes
        super().__init__()
        if not use_mlp:
            self.backbone1 = GCN(in_dim, hid_dim, out_dim, n_layers)
            self.backbone2 = GCN(in_dim, hid_dim, out_dim, n_layers)
            # self.backbone3 = MLP2(2*out_dim, hid_dim, n_classes)
            self.backbone3 = MLP2(out_dim, hid_dim, n_classes)
        else:
            self.backbone1 = MLP(in_dim, hid_dim, out_dim)
            self.backbone2 = MLP(in_dim, hid_dim, out_dim)

    #def get_embedding(self, graph, feat):
    #    out = self.backbone(graph, feat)
    #    return out.detach()

    def forward(self, graph1, feat1, graph2, feat2):
        h1 = self.backbone1(graph1, feat1)
        h2 = self.backbone1(graph2, feat2)

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)
        
        pred1 = self.backbone3(z1)
        pred2 = self.backbone3(z2)
        
        return z1, z2, pred1, pred2


class TOCCA5(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_classes, n_layers, use_mlp = False): # add n_classes
        super().__init__()
        if not use_mlp:
            self.backbone1 = GCN(in_dim, hid_dim, out_dim, n_layers)
            self.backbone2 = GCN(in_dim, hid_dim, out_dim, n_layers)
            # self.backbone3 = MLP2(2*out_dim, hid_dim, n_classes)
            self.backbone3 = GCN(out_dim, hid_dim, n_classes, n_layers)
        else:
            self.backbone1 = MLP(in_dim, hid_dim, out_dim)
            self.backbone2 = MLP(in_dim, hid_dim, out_dim)

    #def get_embedding(self, graph, feat):
    #    out = self.backbone(graph, feat)
    #    return out.detach()

    def forward(self, graph1, feat1, graph2, feat2):
        h1 = self.backbone1(graph1, feat1)
        h2 = self.backbone1(graph2, feat2)

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)
        
        pred1 = self.backbone3(graph1, z1)
        pred2 = self.backbone3(graph2, z2)
        
        return z1, z2, pred1, pred2