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
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index, model='GNAE', scaling_factor=1.8):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=1, alpha=0)
        self.model = model
        self.scaling_factor = scaling_factor

    def forward(self, x, edge_index,not_prop=0):
        if self.model == 'GNAE':
            x = self.linear1(x)
            x = F.normalize(x,p=2,dim=1)  * self.scaling_factor
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.propagate(x, edge_index)
            return x

        if self.model == 'VGNAE':
            x_ = self.linear1(x)
            x_ = self.propagate(x_, edge_index)

            x = self.linear2(x)
            x = F.normalize(x,p=2,dim=1) * self.scaling_factor
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.propagate(x, edge_index)
            return x, x_

        return x
