#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import init
from dgl.nn import GraphConv
from layer import GaussianSample
from distribution import log_gaussian
from distribution import log_standard_gaussian
from distribution import log_standard_categorical


# In[ ]:


class Encoder(nn.Module):
    def __init__(self, dims, dropout, sample_layer=GaussianSample):
        """
        Inference network with GCN hidden layer
        and stochastic output layer
        Returns latent sample from q_φ(z|a,x)
        and the two parameters of the distribution (µ, log σ²).
        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        """
        super().__init__()

        [x_dim, h_dim, z_dim] = dims
        neurons = [x_dim, *h_dim]
        graphconv_layers = [GraphConv(neurons[i-1], neurons[i], bias=False) for i in range(1, len(neurons))]

        self.dropout = dropout
        self.hidden = nn.ModuleList(graphconv_layers)
        self.sample = sample_layer(h_dim[-1], z_dim)

    def forward(self, g, x):
        for layer in self.hidden:
            x = F.relu(layer(g, x))
            x = F.dropout(x, self.dropout, training=self.training)

        return self.sample(g, x)


class Decoder(nn.Module):
    def __init__(self):
        """
        Generative network given by inner product
        between latent variables with
        logistic sigmoid output layer
        Returns adjacency matrix A from p(A|Y,Z).
        """
        super().__init__()
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.mm(x, x.t())
        return self.output_activation(x)

# In[ ]:


class VariationalGraphAutoencoder(nn.Module):
    def __init__(self, dims):
        """
        Variational Graph Autoencoder [Kipf & Welling 2016] model
        where an inference model parameterized by a two-layer GCN
        and an generative model is given by an inner product
        between latent variables.
        """
        super().__init__()

        [x_dim, z_dim, h_dim] = dims
        self.z_dim = z_dim
        self.flow = None

        self.encoder = Encoder([x_dim, h_dim, z_dim], dropout=0.5)
        self.decoder = Decoder()
        self.kl_divergence = 0

        for m in self.modules():
            if isinstance(m, GraphConv):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        for n in self.modules():
            if isinstance(n, nn.Linear):
                init.xavier_normal_(n.weight.data)
                if n.bias is not None:
                    n.bias.data.zero_()

    def _kld(self, z, q_param, p_param=None):
        """
        Computes the KL-divergence of
        some element z.
        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]
        :param z: sample from q-distribuion
        :param q_param: (mu, log_var) of the q-distribution
        :param p_param: (mu, log_var) of the p-distribution
        :return: KL(q||p)
        """
        (mu, log_var) = q_param

        if self.flow is not None:
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        else:
            qz = log_gaussian(z, mu, log_var)

        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)

        kl = qz - pz

        return kl

    def add_flow(self, flow):
        self.flow = flow

    def forward(self, g, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.
        :param x: features
        :return: adjacency matrix
        """
        z, z_mu, z_log_var = self.encoder(g, x)

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        A = self.decoder(z)

        return A

    def sample(self, z):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: Random normal variable
        :return: generated adjacency matrix A
        """
        A = self.decoder(z)
        return A

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)



class DeepGraphGenerativeModel(VariationalGraphAutoencoder):
    def __init__(self, dims):
        """

        Deep Graph Generative Model inspired from the paper
        'Semi-Supervised Learning with Deep Generative Models'
        (Kingma 2014) in PyTorch.
        The DGGM is a probabilistic
        model that incorporates label information in both
        inference and generation.
        Initialise a new generative model
        :param dims: dimensions of x, y, z and hidden layers.
        """
        [x_dim, self.y_dim, z_dim, h_dim] = dims
        super(DeepGraphGenerativeModel, self).__init__([x_dim, z_dim, h_dim])

        self.encoder = Encoder([x_dim + self.y_dim, h_dim, z_dim], dropout=0.5)
        # self.decoder_X = Decoder_X([z_dim + self.y_dim, list(reversed(h_dim)), x_dim], dropout=0.5)
        self.decoder = Decoder()
        self.classifier = Classifier([x_dim, h_dim[0], self.y_dim], dropout=0.5)

        for m in self.modules():
            if isinstance(m, GraphConv):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        for n in self.modules():
            if isinstance(n, nn.Linear):
                init.xavier_normal_(n.weight.data)
                if n.bias is not None:
                    n.bias.data.zero_()

    def forward(self, g, x, y):
        z, z_mu, z_log_var = self.encoder(g, torch.cat([x, y], dim=1))

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        A = self.decoder(torch.cat([z, y], dim=1))

        # X = self.decoder_X(torch.cat([z, y], dim=1))

        return A #,X

    def classify(self, g, x):
        logits = self.classifier(g, x)
        return logits

    def sample(self, z, y):
        """
        Samples from the Decoder to generate an A.
        :param z: latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """

        A = self.decoder(torch.cat([z, y], dim=1))

        return A

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
            r"""Given latent variables :obj:`z`, computes the binary cross
            entropy loss for positive edges :obj:`pos_edge_index` and negative
            sampled edges.

            Args:
                z (Tensor): The latent space :math:`\mathbf{Z}`.
                pos_edge_index (LongTensor): The positive edges to train against.
                neg_edge_index (LongTensor, optional): The negative edges to train
                    against. If not given, uses negative sampling to calculate
                    negative edges. (default: :obj:`None`)
            """

            pos_loss = -torch.log(
                self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

            # Do not include self-loops in negative samples
            pos_edge_index, _ = remove_self_loops(pos_edge_index)
            pos_edge_index, _ = add_self_loops(pos_edge_index)
            if neg_edge_index is None:
                neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
            neg_loss = -torch.log(1 -
                                  self.decoder(z, neg_edge_index, sigmoid=True) +
                                  EPS).mean()


    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


# In[ ]:


class AuxiliaryDeepGraphGenerativeModel(DeepGraphGenerativeModel):
    def __init__(self, dims):
        """
        Auxiliary Deep Graph Generative Model inspired from the paper
        'Auxiliary Deep Generative Models' (Maaløe 2016) in PyTorch.
        The ADGGM introduces an additional latent variable 'a',
        which enables the model to fit more complex variational distributions.
        :param dims: dimensions of x, y, z, a and hidden layers.
        """
        [x_dim, y_dim, z_dim, a_dim, h_dim] = dims
        super(AuxiliaryDeepGraphGenerativeModel, self).__init__([x_dim, y_dim, z_dim, h_dim])

        # hh_dim = [i*2 for i in h_dim]
        # hh_dim = hh_dim + h_dim

        self.aux_encoder = Encoder([x_dim, h_dim, a_dim], dropout=0.5)
        self.aux_decoder = Encoder([x_dim + z_dim + y_dim, list(reversed(h_dim)), a_dim], dropout=0.5) # adding x_dim
        self.classifier = Classifier([x_dim + a_dim, h_dim[0], y_dim], dropout=0.5)

        self.encoder = Encoder([a_dim + y_dim + x_dim, h_dim, z_dim], dropout=0.5)
        self.decoder = Decoder()

    def forward(self, g, x, y):
        """
        Forward through the model
        :param x: features
        :param y: labels
        :return: adjacency matrix
        """
        q_a, q_a_mu, q_a_log_var = self.aux_encoder(g, x)

        z, z_mu, z_log_var = self.encoder(g, torch.cat([x, y, q_a], dim=1))

        p_a, p_a_mu, p_a_log_var = self.aux_decoder(g, torch.cat([x, y, z], dim=1)) # adding x

        A = self.decoder(torch.cat([y, z, p_a], dim=1))

        a_kl = self._kld(q_a, (q_a_mu, q_a_log_var), (p_a_mu, p_a_log_var))
        z_kl = self._kld(z, (z_mu, z_log_var))

        self.kl_divergence = a_kl + z_kl

        return A

    def classify(self, g, x):

        a, a_mu, a_log_var = self.aux_encoder(g, x)

        logits = self.classifier(g, torch.cat([x, a], dim=1))

        return logits

    def sample(self, g, z, y):
        """
        Samples from the Decoder to generate an A.
        :param z: latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        p_a, p_a_mu, p_a_log_var = self.aux_decoder(g, torch.cat([x, y, z], dim=1))

        # A = self.decoder(p_a)

        return p_a

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
                r"""Given latent variables :obj:`z`, computes the binary cross
                entropy loss for positive edges :obj:`pos_edge_index` and negative
                sampled edges.

                Args:
                    z (Tensor): The latent space :math:`\mathbf{Z}`.
                    pos_edge_index (LongTensor): The positive edges to train against.
                    neg_edge_index (LongTensor, optional): The negative edges to train
                        against. If not given, uses negative sampling to calculate
                        negative edges. (default: :obj:`None`)
                """

                pos_loss = -torch.log(
                    self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

                # Do not include self-loops in negative samples
                pos_edge_index, _ = remove_self_loops(pos_edge_index)
                pos_edge_index, _ = add_self_loops(pos_edge_index)
                if neg_edge_index is None:
                    neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
                neg_loss = -torch.log(1 -
                                      self.decoder(z, neg_edge_index, sigmoid=True) +
                                      EPS).mean()


        def test(self, z, pos_edge_index, neg_edge_index):
            r"""Given latent variables :obj:`z`, positive edges
            :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
            computes area under the ROC curve (AUC) and average precision (AP)
            scores.

            Args:
                z (Tensor): The latent space :math:`\mathbf{Z}`.
                pos_edge_index (LongTensor): The positive edges to evaluate
                    against.
                neg_edge_index (LongTensor): The negative edges to evaluate
                    against.
            """
            from sklearn.metrics import average_precision_score, roc_auc_score

            pos_y = z.new_ones(pos_edge_index.size(1))
            neg_y = z.new_zeros(neg_edge_index.size(1))
            y = torch.cat([pos_y, neg_y], dim=0)

            pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
            neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
            pred = torch.cat([pos_pred, neg_pred], dim=0)

            y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

            return roc_auc_score(y, pred), average_precision_score(y, pred)
