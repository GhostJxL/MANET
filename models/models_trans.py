from timeit import repeat
import torch
import torch.nn as nn
from .layers_trans import AELayer, ModalGraphConvolution
from .SelfAttention import ScaledDotProductAttention
from einops import rearrange, repeat

class MMDLNetV0(nn.Module):
    def __init__(self, config):
        super(MMDLNetV0, self).__init__()
        self.num_classes = config.num_classes
        self.in_dims = config.in_dims
        self.hid_dims = config.hid_dims
        self.out_dims = config.out_dims
        self.lr = config.learning_rate
        self.device = torch.device('cuda:0')

        self.ClsLoss = nn.BCEWithLogitsLoss()
        self.bottle = 32
        self.PreNet_A = AELayer(in_dims=self.in_dims[0], out_dims=self.hid_dims[0])
        self.PreNet_B = AELayer(in_dims=self.in_dims[0], out_dims=self.hid_dims[0])
        self.PreNet_C = AELayer(in_dims=self.in_dims[1], out_dims=self.hid_dims[1])
        
        self.GCN = ModalGraphConvolution(self.hid_dims[1], self.hid_dims[1], self.num_classes, self.device)
        
        self.gen_fsn = nn.Conv1d(self.hid_dims[1], self.bottle, 1)
        
        self.SAtt_A = ScaledDotProductAttention(d_model=self.num_classes, d_k=self.num_classes, d_v=self.num_classes, h=1)
        self.SAtt_B = ScaledDotProductAttention(d_model=self.num_classes, d_k=self.num_classes, d_v=self.num_classes, h=1)
        self.SAtt_C = ScaledDotProductAttention(d_model=self.num_classes, d_k=self.num_classes, d_v=self.num_classes, h=1)
        self.SAtt_S = ScaledDotProductAttention(d_model=self.num_classes, d_k=self.num_classes, d_v=self.num_classes, h=1)
        
        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        self.last_linear = nn.Conv1d(self.bottle, self.num_classes, 1)
        
    def get_optimizer(self):
        optim = [{'params': self.parameters(), 'lr': self.lr}]
        return optim

    def forward(self, A, B,C, Y_true):
        """
        :param A: [batch_size, dims] visual
        :param B: [batch_size, dims] trajectory
        :param C: [batch_size, dims] audio
        :param Y_true: [batch_size, num_classes]
        :return:
        """
        
        # Modal init encoder
        encode_A = self.PreNet_A(A)
        encode_B = self.PreNet_B(B)
        encode_C = self.PreNet_C(C)
        
        # generate category matrix
        za = encode_A.view(encode_A.size(0), encode_A.size(1), 1)
        za = za.repeat(1, 1, self.num_classes)
        zb = encode_B.view(encode_B.size(0), encode_B.size(1), 1)
        zb = zb.repeat(1, 1, self.num_classes)
        zc = encode_C.view(encode_C.size(0), encode_C.size(1), 1)
        zc = zc.repeat(1, 1, self.num_classes)
          
        # DGCN
        hs, ha, hb, hc = self.GCN(za, zb, zc)
        

        # SSA       
        # init fsn
        dim_b = self.bottle 
        fsn_tokens = self.gen_fsn(ha)
        m = self.hid_dims[0]

        #trajectory
        temp_s = torch.cat((hs, fsn_tokens), dim=1)
        _, hs_l2 = self.SAtt_S(temp_s,temp_s,temp_s)
        fsn_tokens_t1 = hs_l2[:,m:m+dim_b]
        hs = (hs_l2[:,0:m]+hs) 
               
        temp_b = torch.cat((hb, fsn_tokens), dim=1)
        _, hb_l2 = self.SAtt_B(temp_b,temp_b,temp_b)
        fsn_tokens_t2 = hb_l2[:,m:m+dim_b]
        hb = (hb_l2[:,0:m]+hb)
        
        fsn_tokens = fsn_tokens + fsn_tokens_t2 + fsn_tokens_t1
                         
        #audio
        temp_s = torch.cat((hs, fsn_tokens), dim=1)
        _, hs_l2 = self.SAtt_S(temp_s,temp_s,temp_s)
        fsn_tokens_t1 = hs_l2[:,m:m+dim_b]
        hs = (hs_l2[:,0:m]+hs) 
        
        temp_c = torch.cat((hc, fsn_tokens), dim=1)
        _, hc_l2 = self.SAtt_C(temp_c,temp_c,temp_c)
        fsn_tokens_t2 = hc_l2[:,m:m+dim_b]
        hc = (hc_l2[:,0:m]+hc)

        fsn_tokens = fsn_tokens + fsn_tokens_t2 + fsn_tokens_t1         
 
        # visual
        temp_s = torch.cat((hs, fsn_tokens), dim=1)
        _, hs_l2 = self.SAtt_S(temp_s,temp_s,temp_s)
        fsn_tokens_t1 = hs_l2[:,m:m+dim_b]
        hs = (hs_l2[:,0:m]+hs) 
        
        temp_a = torch.cat((ha, fsn_tokens), dim=1)
        _, ha_l2 = self.SAtt_A(temp_a,temp_a,temp_a)
        fsn_tokens_t2 = ha_l2[:,m:m+dim_b]
        ha = (ha_l2[:,0:m]+ha)

        fsn_tokens = fsn_tokens + fsn_tokens_t2 + fsn_tokens_t1
 
        #cls
        score = self.last_linear(fsn_tokens)
        mask_mat = self.mask_mat.detach()  
        score = (score * mask_mat).sum(-1)

        loss_cls = self.ClsLoss(score, Y_true)
        return score, loss_cls
