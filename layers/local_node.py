import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random

def sim(h1, h2):
    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)
    return torch.mm(z1, z2.t())

def contrastive_loss_sim(hc, hp1, hn1, hn2, hn3, hn4, hn5):
    f = lambda x: torch.exp(x)

    p_sim = f(sim(hc, hp1))
    n_sim1 = f(sim(hc, hn1))
    n_sim2 = f(sim(hc, hn2))
    n_sim3 = f(sim(hc, hn3))
    n_sim4 = f(sim(hc, hn4))
    n_sim5 = f(sim(hc, hn5))

    return -torch.log(p_sim.diag() /
                     (p_sim.sum(dim=-1) + n_sim1.sum(dim=-1) - n_sim1.diag()+ 
                      n_sim2.sum(dim=-1) - n_sim2.diag() + n_sim3.sum(dim=-1) - n_sim3.diag() 
                      + n_sim4.sum(dim=-1) - n_sim4.diag()+ n_sim5.sum(dim=-1) - n_sim5.diag()))

class Local_Node(nn.Module):
    def __init__(self):
        super(Local_Node, self).__init__()
      
    def forward(self, h_c, h_p1, h_p2, h_n1, h_n2, h_n3, h_n4, h_n5):

        num_node = 10
        seq_len = h_p1.size(1)-1
        loss_node1 =0
        loss_node2 =0
        for _ in range(num_node):
            node_index = random.randint(1,seq_len)
     
            h_p1_ = h_p1[:,node_index,:]
       
    
            h_n1_ = h_n1[:,node_index,:]
            h_n2_ = h_n2[:,node_index,:]
            h_n3_ = h_n3[:,node_index,:]
            h_n4_ = h_n4[:,node_index,:]
            h_n5_ = h_n5[:,node_index,:]
            loss_node1 += contrastive_loss_sim(h_c, h_p1_, h_n1_, h_n2_, h_n3_, h_n4_, h_n5_)
        
        for _ in range(num_node):
            node_index = random.randint(1,seq_len)
            h_p2_ = h_p2[:,node_index,:]
       
    
            h_n1_ = h_n1[:,node_index,:]
            h_n2_ = h_n2[:,node_index,:]
            h_n3_ = h_n3[:,node_index,:]
            h_n4_ = h_n4[:,node_index,:]
            h_n5_ = h_n5[:,node_index,:]
            loss_node2 += contrastive_loss_sim(h_c, h_p2_, h_n1_, h_n2_, h_n3_, h_n4_, h_n5_)
     
        loss_node_ = 1/(2*num_node)*(loss_node1+loss_node2)

        return loss_node_.mean()