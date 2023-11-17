import torch
import torch.nn as nn
from layers import GraphCNN, AvgReadout, Discriminator
from layers1 import SecondaryCapsuleLayer, firstCapsuleLayer, ReconstructionNet
import torch.nn.functional as F

import sys
sys.path.append("models/")

class CapsGI(nn.Module):
    def __init__(self, num_layers, nb_nodes, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device):
        super(DGI, self).__init__()
        self.gin = GraphCNN(num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(hidden_dim)
        self.nb_nodes = nb_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.first_capsule = firstCapsuleLayer(number_of_features=self.input_dim,
                                               max_node_num=self.nb_nodes,
                                               capsule_dimensions=128,
                                               disentangle_num=4, 
                                               dropout=0)

        self.hidden_capsule = SecondaryCapsuleLayer(num_iterations=3,  
                                                   num_routes=self.nb_nodes,  
                                                   num_capsules=30, 
                                                   in_channels=128, 
                                                   out_channels=128,
                                                   dropout=0) 

    def forward(self, seq1, seq2, adj, msk, samp_bias1, samp_bias2, lbl):
        criterion = nn.BCEWithLogitsLoss()

        h_1 = torch.unsqueeze(self.gin(seq1, adj), 0)
        h_2 = torch.unsqueeze(self.gin(seq2, adj), 0)

        out = self.first_capsule(seq1, adj)
        out2, c_ij, adj2 = self.hidden_capsule(out, adj)

        c2 = torch.mean(out2, 1)
        c2 = self.sigm(c2)
        c2 = c2.squeeze(2)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        ret = self.disc(c2, h_1, h_2, samp_bias1, samp_bias2)
        loss = criterion(ret, lbl)

        return loss