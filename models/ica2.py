import copy
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import APPNP,  GINConv, GCNConv
import numpy as np
from models.baseline_models import GCN, MLP, LogReg


class ICA_GIN(nn.Module):
    def __init__(self, in_dim, hid_dims, out_dim, use_mlp = False,
                 use_graph=True, normalize=True, regularize=True):
        super().__init__()
        self.normalize = normalize
        self.f = MLP(in_dim, hid_dims, hid_dims, n_layers=1,
                     activation='relu', slope=.1, device='cpu',
                     use_bn=False)
        self.layer1 = GINConv(self.f, eps=-1.)
        self.layer2 = GCNConv(hid_dims + in_dim , out_dim)
        self.regularize = regularize
        self.decode = MLP(out_dim, int(out_dim/2), 2,
                          n_layers=2, activation='relu',
                          slope=.1, device='cpu', use_bn=False)

    def get_embedding(self, x, edge_index):
        h1 = self.layer1(x, edge_index)
        if self.regularize:
            z1 = (h1 - h1.mean(0)) / h1.std(0)
        else:
            z1 = h1
        out = self.layer2(torch.cat((x, h1), 1), edge_index)
        return out.detach()

    def forward(self, x, edge_index):
        h1 = self.layer1(x, edge_index)
        if self.regularize:
            z1 = (h1 - h1.mean(0)) / h1.std(0)
        else:
            z1 = h1
        x = self.layer2(torch.cat((x, h1), 1), edge_index)
        z1 = self.decode(x)
        return z1
