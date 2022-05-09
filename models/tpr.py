import torch
import math
import torch.nn as nn
from layers import Local_Node
from utils import process
import torch.nn.functional as F
from torch.autograd import Variable
from embed import PathEmbedding

class EncoderLSTM(nn.Module):
    def __init__(self, node2vec, temporal2vec, input_size, hidden_size,
                 n_layers=2, dropout=0.5):
        super(EncoderLSTM, self).__init__()

        self.embedding = PathEmbedding(node2vec=node2vec)
        self.temporalembed= nn.Embedding.from_pretrained(temporal2vec)

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout)

    def forward(self, path, ts, hidden=None):


        path_embed_= self.embedding(path)
      
        Temporal =self.temporalembed(ts)
        temporal_ =Temporal.expand_as(path_embed_).contiguous()

        path_embed = torch.add(path_embed_,temporal_)


        path_embed = torch.cat((path_embed, temporal_), 2).transpose(1,0) #seq_len, batch_size,dim

        outputs, hidden = self.lstm(path_embed, hidden)
        outputs = outputs.transpose(0,1)

        return outputs, hidden, path_embed_

class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            return torch.sum(seq * msk, 1) / torch.sum(msk,1) 

def sim(h1, h2):
    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)
    return torch.mm(z1, z2.t())

def contrastive_loss_sim(hc, hp1, hn1, hn2, hn3, hn4, hn5, hn6,hn7,hn8):
    f = lambda x: torch.exp(x)
    p_sim = f(sim(hc, hp1))
    n_sim1 = f(sim(hc, hn1))
    n_sim2 = f(sim(hc, hn2))
    n_sim3 = f(sim(hc, hn3))
    n_sim4 = f(sim(hc, hn4))
    n_sim5 = f(sim(hc, hn5))
    n_sim6 = f(sim(hc, hn6))
    n_sim7 = f(sim(hc, hn7))
    n_sim8 = f(sim(hc, hn8))
    return -torch.log(p_sim.diag() /
                     (p_sim.sum(dim=-1) + n_sim1.sum(dim=-1) - n_sim1.diag()+ 
                      n_sim2.sum(dim=-1) - n_sim2.diag() + n_sim3.sum(dim=-1) - n_sim3.diag() 
                      + n_sim4.sum(dim=-1) - n_sim4.diag() + n_sim5.sum(dim=-1) - n_sim5.diag()
                      + n_sim6.sum(dim=-1) - n_sim6.diag()+ n_sim7.sum(dim=-1) - n_sim7.diag()
                      + n_sim8.sum(dim=-1) - n_sim8.diag()))


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


class TPR(nn.Module):
    def __init__(self, node2vec, temporal2vec, input_size=256, hidden_size= 128, n_layers=1, dropout=0.5):

        super(TPR, self).__init__()

        self.encoderLSTM = EncoderLSTM(node2vec=node2vec,temporal2vec=temporal2vec,input_size=input_size, hidden_size=hidden_size, n_layers=n_layers, dropout=dropout)
        
        self.local_node = Local_Node()

        self.read  = Readout()

    def forward(self, path, path_mask, NP1, NP1_mask, NP2, NP2_mask, NP3, NP3_mask, NP4, NP4_mask, TS, TPS1, TPS2, TNS1, TNS2):


        ##query path 
        path_output, hidden_path, path_ori1 = self.encoderLSTM(path, TS)
        qpath = self.read(path_output, path_mask)

        ##Positve paths
        pt1, _,_ = self.encoderLSTM(path, TPS1)
        pkpath1 = self.read(pt1, path_mask)

        pt2, _,_ = self.encoderLSTM(path, TPS2)
        pkpath2 = self.read(pt2, path_mask)


        ##Negative Paths
        npt1, _,_ = self.encoderLSTM(path, TNS1) ##same path differnt weak label
        nkpath1 = self.read(npt1, path_mask)

        npt2, _,_ = self.encoderLSTM(NP1, TNS1) ##different path differnt weak label
        nkpath2 = self.read(npt2, NP1_mask)

        npt3, _,_ = self.encoderLSTM(NP2, TNS2) ##different path different weak label
        nkpath3 = self.read(npt3, NP2_mask)

        npt4, _, _ = self.encoderLSTM(NP4, TNS2)  ##different path differnt weak label
        nkpath4 = self.read(npt4, NP4_mask)

        npt5, _, _ = self.encoderLSTM(NP3, TS)  ##different path same weak label
        nkpath5 = self.read(npt5, NP3_mask)


        ##path
       
        l1 = contrastive_loss_sim(qpath, pkpath1, nkpath1, nkpath2, nkpath3, nkpath4, nkpath5)
        l2 = contrastive_loss_sim(qpath, pkpath2, nkpath1, nkpath2, nkpath3, nkpath4, nkpath5)
        loss_path = 0.5*(l1+l2)
     
        ##node
        loss_node = self.local_node(qpath, pt1*path_mask, pt2*path_mask, npt1*path_mask, npt2*NP1_mask, npt3*NP2_mask, npt4*NP4_mask, npt5*NP3_mask)

        return loss_path.mean(), loss_node

    def embed(self, path, path_mask, ts):
        ##path 
        path_output, hidden_path, path_ori = self.encoderLSTM(path, ts)
        path_embed = torch.unsqueeze(self.read(path_output, path_mask),1)

        return path_embed.detach()