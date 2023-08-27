import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.fpa import Augmented_GPR_Layer

class FPA(torch.nn.Module):
    def __init__(self, dataset, data, args):
        super(FPA, self).__init__()
        self.dropout = args.dropout
        self.train_mask = data.train_mask
        self.lin1 = nn.Linear(dataset.num_features, args.hidden_channels)
        self.lin2 = nn.Linear(args.hidden_channels, dataset.num_classes, bias=False)
        self.prop = Augmented_GPR_Layer(args.K, args.alpha)
        self.eta = args.eta
        self.epoch = 1
        self.simp = args.simp
        self.dataname = args.dataset
        self.activation = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        Y = data.y_pred
        x = data.x
        edge_index = data.edge_index
        x = F.dropout(x, self.dropout, training=self.training)
        h1 = F.relu(self.lin1(x))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        if self.epoch <= 100: #for robustness
            self.epoch = self.epoch + 1
            h2 = self.prop(h1, edge_index, self.eta)
            h2 = self.lin2(h2)
        else:
            if ~self.simp: #FPA-master
                aug = self.activation(Y - (F.softmax(self.lin2(self.lin1(x)), dim=1)/np.sqrt(self.lin2.weight.shape[1]))) @ self.lin2.weight
            else: #SFPA
                aug = self.activation(Y * (F.softmax(self.lin2(self.lin1(x)), dim=1)/np.sqrt(self.lin2.weight.shape[1]))) @ self.lin2.weight
            h2 = self.prop(h1, edge_index, aug, self.eta)
            h2 = self.lin2(h2)
        return F.log_softmax(h2, dim=1)


