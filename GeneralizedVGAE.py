import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE, VGAE, APPNP
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch
import torch_geometric as tg
import pandas as pd
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE, VGAE, APPNP



from aggregation import *

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, norm=True, alpha=0.5, beta=1.0, gnn_type='normal', non_linearity=True):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, alpha=alpha, beta=beta, gnn_type=gnn_type) # cached only for transductive learning
        self.conv2 = GCNConv(2* out_channels, out_channels, alpha=alpha, beta=beta, gnn_type=gnn_type) # cached only for transductive learning
        self.norm = norm
        self.non_linearity = non_linearity

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if self.non_linearity: x = x.relu()
        if self.norm == True:
            x = F.normalize(x, p=2., dim=-1)
            x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, norm=True, alpha=0.5, beta=1.0, gnn_type='normal', non_linearity=True):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, alpha=alpha, beta=beta, gnn_type=gnn_type) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, alpha=alpha, beta=beta, gnn_type=gnn_type)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, alpha=alpha, beta=beta, gnn_type=gnn_type)
        self.norm = norm
        self.non_linearity = non_linearity

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if self.non_linearity: x = x.relu()
        if self.norm==True:
            x = F.normalize(x, p=2, dim=1)
            x = F.dropout(x, p=0.5, training=self.training)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
