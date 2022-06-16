import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline_models import GCN, MLP

class CCA_SSG(nn.Module):
    r""" CCA model
    Args:
        in_dim(int): Size of each input sample
        out_dim (int): Size of each output sample.
        hid_dim (list of ints): Size of the hidden embeddings (per layer)
        normalize (bool, optional): Whether to normalize
            the embeddings at each layer
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers,
    activation='relu', slope=.1, device='cpu', normalize=True, use_mlp=False):
        super().__init__()
        self.use_mlp = use_mlp
        if not self.use_mlp:
            self.backbone = GCN(input_dim, hidden_dim, output_dim,
                                n_layers, activation=activation, slope=slope,
                                device=device, normalize=normalize)
        else:
            self.backbone = MLP(input_dim, hidden_dim, output_dim,
                                n_layers, activation=activation, slope=slope,
                                device=device, normalize=normalize)

    def get_embedding(self, data1):
        out = self.backbone(data1.x, data1.edge_index)
        return out.detach()

    def forward(self, data1, data2):
        h1 = self.backbone(data1.x, data1.edge_index)
        h2 = self.backbone(data2.x, data2.edge_index)

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        return z1, z2
