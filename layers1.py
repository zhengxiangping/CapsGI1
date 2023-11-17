"""CapsGNN layers."""
from functools import partial
import torch
import torch.nn as nn
from torch.autograd import Variable
from denseGCNConv import DenseGCNConv
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from disentangle import linearDisentangle


def squash(input_tensor, dim=-1, epsilon=1e-7):
    squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    scale = squared_norm / (1 + squared_norm)
    unit_vector = input_tensor / safe_norm
    return scale * unit_vector

def sparse2dense(x, new_size, mask):
    out = torch.zeros(new_size).cuda()   # out=torch.Size([128, 492, 128])
    out[mask] = x    # mask = torch.Size([128, 492])  x=torch.Size([9016, 128])
    return out   # out=torch.Size([128, 492, 128])

class firstCapsuleLayer(torch.nn.Module):
    def __init__(self, number_of_features, max_node_num, capsule_dimensions, disentangle_num, dropout):
        super(firstCapsuleLayer, self).__init__()

        self.number_of_features = number_of_features 
        self.max_node_num = max_node_num            
        self.capsule_dimensions = capsule_dimensions  
        self.disentangle_num = disentangle_num 
        self.dropout = nn.Dropout(p=dropout)   
        self.bns_disen = nn.BatchNorm1d(self.number_of_features)
        self.disen = torch.nn.ModuleList()
        for i in range(self.disentangle_num):
            self.disen.append(linearDisentangle(self.number_of_features, self.capsule_dimensions//self.disentangle_num))

    def forward(self, x, adj):
        x_size = x.size()
        out = []
        x = self.bns_disen(x) 
        for i, disen in enumerate(self.disen):
            temp = F.relu(disen(x))  
            temp = self.dropout(temp)
            out.append(temp)
        out = torch.cat(out, dim=-1)  
        out = squash(out)
        x = x.view(1, -1, 64)
        return out   # torch.Size([128, 492, 128])


class SecondaryCapsuleLayer(torch.nn.Module):
    def __init__(self, num_iterations, num_routes, num_capsules, in_channels, out_channels, dropout):
        super(SecondaryCapsuleLayer, self).__init__()
        self.num_prim_cap = num_routes  
        self.num_digit_cap = num_capsules 
        self.in_cap_dim = in_channels
        self.out_cap_dim = out_channels 
        self.dropout = nn.Dropout(p=dropout)
        self.num_iterations = num_iterations 
        self.bn_feat = nn.BatchNorm1d(self.in_cap_dim)
        self.convs = torch.nn.ModuleList()

        for i in range(self.num_digit_cap):  #10
            self.convs.append(DenseGCNConv(self.in_cap_dim, self.out_cap_dim))
                 
    def forward(self, x, adj):
        batch_size = 1  # [bs, num_prim_caps, prim_cap_dim]  #torch.Size([128, 492, 128])  torch.Size([128, 10, 128])
        x = x.view(-1, self.in_cap_dim)  # torch.Size([1280, 128])
        x = self.bn_feat(x)  # torch.Size([1280, 128]
        x = x.view(-1, self.in_cap_dim) # torch.Size([128, 492, 128])
        u_hat = []
        for i, conv in enumerate(self.convs):
            # print('x', adj.shape, x.shape)
            temp = conv(x, adj)  
            temp1 = temp.view(1,-1, self.in_cap_dim)
            u_hat.append(temp1)
        u_hat = torch.stack(u_hat, dim=2).unsqueeze(4)  # u_hat torch.Size([128, 492, 10, 128, 1])    torch.Size([128, 10, 3, 128, 1])
        temp_u_hat = u_hat.detach()  
        b_ij = torch.zeros(batch_size, u_hat.size(1), u_hat.size(2), 1, 1).cuda() 
        x = x.view(1, -1, self.in_cap_dim)
        for i in range(self.num_iterations - 1):
            c_ij = F.softmax(b_ij, dim=2)  
            s_j = (c_ij * temp_u_hat).sum(dim=1, keepdim=True) 
            v = squash(s_j, dim=-2)  #v torch.Size([128, 1, 10, 128, 1])
            u_produce_v = torch.matmul(temp_u_hat.transpose(-1, -2), v) 
            b_ij = b_ij + u_produce_v  
        
        c_ij = F.softmax(b_ij, dim=2)  
        s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)  #u_hat torch.Size([128, 492, 10, 128, 1])
        s_j += torch.mean(x, dim=1)[:,None, None,:,None]  #x = torch.Size([128, 492, 128])
        v = squash(s_j, dim=-2) )
        c_ij = c_ij.squeeze(4).squeeze(3) 
        v = v.squeeze(0)
        c_ij = c_ij.squeeze(0)
        adj = torch.spmm(adj, c_ij)
        return v, c_ij, adj


class ReconstructionNet(nn.Module):
    def __init__(self, n_dim, n_classes, hidden):
        super(ReconstructionNet, self).__init__()
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.fc1 = nn.Linear(n_dim * n_classes, hidden)

    def forward(self, first_capsule, class_capsule, y):
        mask = torch.zeros((class_capsule.size(0), self.n_classes))
        mask = mask.cuda()
        mask.scatter_(1, y.view(-1, 1), 1.)
        mask = mask.unsqueeze(2)
        class_capsule = class_capsule * mask
        class_capsule = class_capsule.view(-1, 1, self.n_dim * self.n_classes)

        N = first_capsule.size(1)
        class_capsule = F.relu(self.fc1(class_capsule))  
        x = first_capsule + class_capsule
        x = torch.matmul(x, torch.transpose(x, 2, 1))
        return x