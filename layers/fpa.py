import numpy as np
import torch
from torch.nn import Linear,Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class Augmented_GPR_Layer(MessagePassing):
    def __init__(self, K, alpha, **kwargs):
        super(Augmented_GPR_Layer, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha
        TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
        TEMP[-1] = (1 - alpha) ** K
        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, aug=None, eta=0., edge_weight=None):
        edge_index, norm = gcn_norm(edge_index, edge_weight, num_nodes=x.
            size(0), dtype=x.dtype)
        hidden = x * self.temp[0]
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            if aug != None:
                x = x * (1-eta) + aug * eta
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
            self.temp)