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
    def __init__(self, dims): #, dropout
        
        super().__init__()

        [x_dim, h_dim, z_dim] = dims
        neurons = [x_dim, *h_dim]
        graphconv_layers = [GraphConv(neurons[i-1], neurons[i], bias=False) for i in range(1, len(neurons))] # 
        
        # self.dropout = dropout
        self.hidden = nn.ModuleList(graphconv_layers)
        self.output = GraphConv(h_dim[-1], z_dim)
        
    def forward(self, g, x):
        for layer in self.hidden:
            x = F.relu(layer(g, x))
            # x = F.dropout(x, self.dropout, training=self.training)
        
        return self.output(g, x)


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


class GraphAutoencoder(nn.Module):
    def __init__(self, dims):

        super().__init__()

        [x_dim, z_dim, h_dim] = dims
        self.z_dim = z_dim

        self.encoder = Encoder([x_dim, h_dim, z_dim]) #, dropout=0.5
        self.decoder = Decoder()

    def forward(self, g, x, y=None):
        z = self.encoder(g, x)
        A = self.decoder(z)

        return A