import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch_geometric as tg
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops

from models.baseline_models import VGCNEncoder


class DeepVGAE(VGAE):
    def __init__(self, enc_in_channels, enc_hidden_channels, enc_out_channels,
                 n_layers=2, normalize=True, activation='relu'):
        super(DeepVGAE, self).__init__(encoder=VGCNEncoder(enc_in_channels,
                                                           enc_hidden_channels,
                                                           enc_out_channels,
                                                           n_layers,
                                                           normalize=normalize,
                                                           activation=activation),
                                       decoder=InnerProductDecoder())

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_pred = self.decoder.forward_all(z)
        return adj_pred

    def loss(self, x, pos_edge_index, neg_edge_index, **kwargs):
        z = self.encode(x, pos_edge_index)

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()
        kl_loss = 1 / x.size(0) * self.kl_loss()
        return pos_loss + neg_loss + kl_loss


    def single_test(self, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        self.eval()
        with torch.no_grad():
            z = self.encode(x, train_pos_edge_index)
        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score
