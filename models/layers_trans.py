import math
import torch
import torch.nn as nn

class AELayer(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(AELayer, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.hid_dims = (in_dims+out_dims)//2

        self.Encoder = nn.Sequential(
            # nn.Linear(self.in_dims, self.out_dims),
            # nn.Sigmoid(),
            nn.Linear(self.in_dims, self.hid_dims),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hid_dims, self.hid_dims),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hid_dims, self.out_dims),
        )

        self.weights = self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                m.bias.data.zero_()

    def forward(self, x):
        encode = self.Encoder(x)
        return encode


class ModalGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, DEVICE):
        super(ModalGraphConvolution, self).__init__()

        self.share_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.share_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.Wtv = nn.Conv1d(in_features, num_nodes, 1)
        self.Wav = nn.Conv1d(in_features, num_nodes, 1)
        self.Wat = nn.Conv1d(in_features, in_features, 1)
        
        self.m_a_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.m_a_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.m_b_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.m_b_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.m_c_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.m_c_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

    def GCN_Share(self, x):
        x = self.share_adj(x.transpose(1, 2))
        x = self.share_weight(x.transpose(1, 2))
        return x

    def GCN_A(self, x):
        x = self.m_a_adj(x.transpose(1, 2))
        x = self.m_a_weight(x.transpose(1, 2))
        return x

    def GCN_B(self, x):
        x = self.m_b_adj(x.transpose(1, 2))
        x = self.m_b_weight(x.transpose(1, 2))
        return x

    def GCN_C(self, x):
        x = self.m_c_adj(x.transpose(1, 2))
        x = self.m_c_weight(x.transpose(1, 2))
        return x
    
    def forward(self, za, zb, zc):
        # GCN
        # Share GCN 3 modal
        left = self.Wtv(za) #C*C
        mid = self.Wav(zc) #C*C
        right = self.Wat(zb).transpose(1, 2) # C*D
        zs = (torch.matmul(torch.matmul(left,mid), right)).transpose(1, 2) 
        hs = self.GCN_Share(zs)

        # # Share GCN 2 modal
        # left = self.Wtv(vc) #C*C
        # right = self.Wat(vb).transpose(1, 2) # C*D
        # z = (torch.matmul(left, right)).transpose(1, 2) 
        # zs = self.GCN_Share(z)
        
        # Modal GCN     
        ha = za+self.GCN_A(za)
        hb = zb+self.GCN_B(zb)     
        hc = zc+self.GCN_C(zc)
        return hs,ha,hb,hc
        



