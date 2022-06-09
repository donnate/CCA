import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


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


class GCN(nn.Module):
    r""" GCN model
    Args:
        in_dim(int): Size of each input sample
        out_dim (int): Size of each output sample.
        hid_dims (list of ints): Size of the hidden embeddings (per layer)
        normalize (bool, optional): Whether to normalize
            the embeddings at each layer
    """
    def __init__(self, in_dim, hid_dims, out_dim, normalize=Trye):
        super().__init__()

        self.n_layers = len(hid_dims)
        self.convs = nn.ModuleList()
        self.normalize = normalize
        if self.n_layers > 1:
            self.convs.append(GCNConv(in_dim, hid_dims[0]))
            for i in range(self.n_layers -1):
                self.convs.append(GCNConv(hid_dims[i], hid_dims[i+1]))
            self.convs.append(GCNConv(hid_dims[-1], out_dim))
        else:
            self.convs.append(GCNConv(in_dim, hid_dims[0]))
            self.convs.append(GCNConv(hid_dims[0], out_dim))

    def forward(self, x, edge_index):
        if self.n_layers>1:
            for i in range(len(self.convs)-1):
                x  = self.convs[i](x, edge_index)
                if self.normalize: x = F.normalize(x, p=2, dim=1)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
            x = self.convs[-1](x, edge_index)
        else:
            x = self.convs[0](x, edge_index)
            if self.normalize: x = F.normalize(x, p=2, dim=1)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.convs[1](x, edge_index)
        return x


class CCA_SSG(nn.Module):
    r""" CCA model
    Args:
        in_dim(int): Size of each input sample
        out_dim (int): Size of each output sample.
        hid_dim (list of ints): Size of the hidden embeddings (per layer)
        normalize (bool, optional): Whether to normalize
            the embeddings at each layer
    """
    def __init__(self, in_dim, hid_dim, out_dims, n_layers, use_mlp=False, normalize=True):
        super().__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dims, out_dim, normalize)
        else:
            self.backbone = MLP(in_dim, hid_dim, out_dim)

    def get_embedding(self, data1):
        out = self.backbone(data1.x, data1.edge_index)
        return out.detach()

    def forward(self, data1, data2):
        h1 = self.backbone(data1.x, data1.edge_index)
        h2 = self.backbone(data2.x, data2.edge_index)

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        return z1, z2
