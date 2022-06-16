from numbers import Number
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, APPNP

from models.aggregation import GenGCNConv

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)



class GCN(nn.Module): # in_dim, hid_dims, out_dim, normalize=True
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers,
    activation='relu', slope=.1, device='cpu', normalize=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        self.propagate = APPNP(K=1, alpha=0)
        self.normalize = normalize
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'relu':
                self._act_f.append(lambda x: torch.nn.ReLU()(x))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x, edge_index):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self.fc[c](h)
                h = F.dropout(h, p=0.5, training=self.training)
                h = self.propagate(h, edge_index)
                if self.normalize: h = F.normalize(h, p=2, dim=1)
                h = self._act_f[c](h)
        return h


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2,
    activation='relu', slope=.1, device='cpu', use_bn=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        self.use_bn = use_bn
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.activation = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'relu':
                self._act_f.append(lambda x: torch.nn.ReLU()(x))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
                if self.use_bn: h= self.bn(h)
        return h

# class MLP(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, use_bn=True):
#         super(MLP, self).__init__()
#
#         self.layer1 = nn.Linear(nfeat, nhid, bias=True)
#         self.layer2 = nn.Linear(nhid, nclass, bias=True)
#
#         self.bn = nn.BatchNorm1d(nhid)
#         self.use_bn = use_bn
#         self.act_fn = nn.ReLU()
#
#     def forward(self, _, x):
#         x = self.layer1(x)
#         if self.use_bn:
#             x = self.bn(x)
#
#         x = self.act_fn(x)
#         x = self.layer2(x)
#
#         return x
#
#
# class GCN(nn.Module):
#     r""" GCN model
#     Args:
#         in_dim(int): Size of each input sample
#         out_dim (int): Size of each output sample.
#         hid_dims (list of ints): Size of the hidden embeddings (per layer)
#         normalize (bool, optional): Whether to normalize
#             the embeddings at each layer
#     """
#     def __init__(self, in_dim, hid_dims, out_dim, normalize=True):
#         super().__init__()
#
#         self.n_layers = len(hid_dims)
#         self.convs = nn.ModuleList()
#         self.normalize = normalize
#         if self.n_layers > 1:
#             self.convs.append(GCNConv(in_dim, hid_dims[0]))
#             for i in range(self.n_layers -1):
#                 self.convs.append(GCNConv(hid_dims[i], hid_dims[i+1]))
#             self.convs.append(GCNConv(hid_dims[-1], out_dim))
#         else:
#             self.convs.append(GCNConv(in_dim, hid_dims[0]))
#             self.convs.append(GCNConv(hid_dims[0], out_dim))
#
#     def forward(self, x, edge_index):
#         if self.n_layers>1:
#             for i in range(len(self.convs)-1):
#                 x  = self.convs[i](x, edge_index)
#                 if self.normalize: x = F.normalize(x, p=2, dim=1)
#                 x = F.relu(x)
#                 x = F.dropout(x, p=0.5, training=self.training)
#             x = self.convs[-1](x, edge_index)
#         else:
#             x = self.convs[0](x, edge_index)
#             if self.normalize: x = F.normalize(x, p=2, dim=1)
#             x = F.relu(x)
#             x = F.dropout(x, p=0.5, training=self.training)
#             x = self.convs[1](x, edge_index)
#         return x


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, enc_hidden_channels,out_channels,
                 normalize=True, alpha=0.5,
                 beta=1.0, gnn_type='normal', non_linearity=nn.ReLU()):
        super(GCNEncoder, self).__init__()
        self.conv1 = GenGCNConv(in_channels, enc_hidden_channels, alpha=alpha,
                             beta=beta, gnn_type=gnn_type)
        self.conv2 = GenGCNConv(enc_hidden_channels, out_channels, alpha=alpha,
                             beta=beta, gnn_type=gnn_type) # cached only for transductive learning
        self.normalize = normalize
        self.non_linearity = non_linearity

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if self.non_linearity: x = x.relu()
        if self.non_linearity is not None: x = self.non_linearity(x)
        return self.conv2(x, edge_index)



class VGCNEncoder(nn.Module): # in_dim, hid_dims, out_dim, normalize=True
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, scaling=1.0,
    activation='relu', slope=.1, device='cpu', normalize=True):
        super(VGCNEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        self.normalize = normalize
        self.scaling = scaling

        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'relu':
                self._act_f.append(lambda x: torch.nn.ReLU()(x))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [torch.nn.Linear(self.input_dim, self.output_dim),
                        torch.nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.propagate = APPNP(K=1, alpha=0)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x, edge_index):
        h = x
        if self.normalize: h = F.normalize(h, p=2, dim=1) * self.scaling
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                #h = self.fc[c](h)
                mu = self.fc[c](h)
                mu = F.dropout(mu, p=0.5, training=self.training)
                if self.normalize: mu = F.normalize(mu, p=2, dim=1) * self.scaling
                mu = self.propagate(mu, edge_index)


                var = self.fc[c + 1](h)
                var = F.dropout(var, p=0.5, training=self.training)
                if self.normalize: var = F.normalize(var, p=2, dim=1) * self.scaling
                var = self.propagate(var, edge_index)

            else:
                h = self.fc[c](h)
                h = F.dropout(h, p=0.5, training=self.training)
                #if self.normalize: h = F.normalize(h, p=2, dim=1) * self.scaling
                h = self.propagate(h, edge_index)
                h = self._act_f[c](h)
        return mu, var


# class VGCNEncoder(torch.nn.Module):
#     def __init__(self, in_channels, enc_hidden_channels,
#                  out_channels, normalize=True, alpha=0.5,
#                  beta=1.0, gnn_type='normal', non_linearity=nn.ReLU()):
#         super(VGCNEncoder, self).__init__()
#         # self.conv1 = GenGCNConv(in_channels,enc_hidden_channels, cached=True,
#         #                      alpha=alpha,
#         #                      beta=beta, gnn_type=gnn_type) # cached only for transductive learning
#         # self.conv_mu = GenGCNConv(enc_hidden_channels, out_channels, cached=True,
#         #                        alpha=alpha,
#         #                        beta=beta, gnn_type=gnn_type)
#         # self.conv_logstd = GenGCNConv(enc_hidden_channels, out_channels, cached=True,
#         #                            alpha=alpha,
#         #                            beta=beta, gnn_type=gnn_type)
#         self.conv1 = GCNConv(in_channels, enc_hidden_channels, cached=True,) # cached only for transductive learning
#         self.conv_mu = GCNConv(enc_hidden_channels, out_channels, cached=True)
#         self.conv_logstd = GCNConv(enc_hidden_channels, out_channels, cached=True)
#         self.normalize = normalize
#         self.non_linearity = non_linearity
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         if self.normalize ==True: x = F.normalize(x, p=2, dim=1)
#         if self.non_linearity is not None: x = self.non_linearity(x)
#         return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
