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
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE, VGAE, APPNP
import torch_geometric.transforms as T

from models.aggregation import GCNConv

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, alpha=0.5,
                 beta=1.0, gnn_type='normal', non_linearity=nn.ReLU()):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, alpha=alpha,
                             beta=beta, gnn_type=gnn_type)
        self.conv2 = GCNConv(2 * out_channels, out_channels, alpha=alpha,
                             beta=beta, gnn_type=gnn_type) # cached only for transductive learning
        self.normalize = normalize
        self.non_linearity = non_linearity

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if self.non_linearity: x = x.relu()
        if self.non_linearity is not None: x = self.non_linearity(x)
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, alpha=0.5,
                 beta=1.0, gnn_type='normal', non_linearity=nn.ReLU()):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True,
                             alpha=alpha,
                             beta=beta, gnn_type=gnn_type) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True,
                               alpha=alpha,
                               beta=beta, gnn_type=gnn_type)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True,
                                   alpha=alpha,
                                   beta=beta, gnn_type=gnn_type)
        self.normalize = normalize
        self.non_linearity = non_linearity

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if self.normalize ==True: x = F.normalize(x, p=2, dim=1)
        if self.non_linearity is not None: x = self.non_linearity(x)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
