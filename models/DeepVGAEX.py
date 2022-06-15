import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops


from models.baseline_models import  VGCNEncoder, MLP


class Decoder_Y(nn.Module):
    def __init__(self, z_dim, h_dims, x_dim, dropout):

        super().__init__()
        neurons = [z_dim, *h_dims]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)
        self.dropout = dropout
        self.reconstruction = nn.Linear(h_dims[-1], x_dim)
        self.x_dim = x_dim
        #self.std = nn.Linear(h_dims[-1], x_dim)
        #self.output_activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden:
            x = layer(x)
            x = F.normalize(x, p=2, dim=1)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return self.reconstruction(x)

    def sample_forward(self, x, sd, M=100):
        z = torch.randn((sd.shape[0],sd.shape[1], M))
        sdev = torch.concat([sd.reshape(sd.shape[0],sd.shape[1], 1)]*M, 2)
        sdev = torch.mul(sdev, z)
        new_input = torch.concat([x.reshape(x.shape[0],x.shape[1], 1)]*M, 2) + sdev
        x = new_input.reshape([-1, x.shape[1]])
        for layer in self.hidden:
            x = layer(x)
            x = F.normalize(x, p=2, dim=1)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.reconstruction(x).reshape([-1, self.x_dim, M])
        return x.mean(2),  x.var(2)


class DeepVGAEX(VGAE):
    def __init__(self, enc_in_channels, enc_hidden_channels, enc_out_channels,
                n_layers, normalize=True,
                h_dims_reconstructiony = [64, 64], y_dim=1, dropout=0.5,
                lambda_y =1., activation='relu'):
        super(DeepVGAEX, self).__init__(encoder=VGCNEncoder(enc_in_channels,
                                                           enc_hidden_channels,
                                                           enc_out_channels,
                                                           n_layers=n_layers,
                                                           normalize=normalize,
                                                           activation=activation),
                                       decoder=InnerProductDecoder())
        self.decoder_y=Decoder_Y(enc_out_channels, h_dims_reconstructiony,
                                 y_dim, dropout)
        self.dropout = dropout
        self.lambda_y = lambda_y

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_pred = self.decoder.forward_all(z)
        return adj_pred

    def predict_y(self, z):
        return self.decoder_y(z)

    def loss(self, x, y, pos_edge_index, neg_edge_index, train_mask):
        z = self.encode(x, pos_edge_index)
        z_x, z_sd = self.encoder(x, pos_edge_index)

        mu_y_hat, logstd =  self.decoder_y.sample_forward(z_x, z_sd)
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()

        # Do not include self-loops in negative samples
        #all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
        #all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp) ### make sure they are there, so we don't sample them using negative sampling

        #neg_edge_index = negative_sampling(all_edge_index_tmp, num_nodes=z.size(0),
        #                                   num_neg_samples=pos_edge_index.size(1))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()

        kl_loss = 1 / x.size(0) * self.kl_loss()
        ### add classification  loss to the thing
        s2 = logstd.exp()**2

        y_loss =  - 0.5 * torch.sum( 2 * logstd[train_mask,:] + torch.mul(torch.square(mu_y_hat[train_mask,:] - y.long()[train_mask,:]), 1.0/s2[train_mask,:]))

        return pos_loss + neg_loss + kl_loss + self.lambda_y * y_loss

    def kl_loss_y(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def single_test(self, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        self.eval()
        with torch.no_grad():
            z = self.encode(x, train_pos_edge_index)
        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score
