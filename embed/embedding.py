import torch.nn as nn
import torch


class PathEmbedding(nn.Module):
    
    def __init__(self, node2vec, dropout=0.1):
       
        super().__init__()

        self.path_embed = nn.Embedding.from_pretrained(node2vec) 
        

        # self.rt = nn.Embedding(21, 64) ##road type
        # self.ow = nn.Embedding(2,16) ##one way or not
        # self.lane = nn.Embedding(7,32) ##number of lane
        # self.t_signal = nn.Embedding(2,16) ##traffic signal

        self.dropout = nn.Dropout(p=dropout)


    # def forward(self, Path, Path_RT, Path_OW, Path_Lane, Path_ts):
    #     x = torch.cat([self.path_embed(Path), self.rt(Path_RT),self.ow(Path_OW),self.lane(Path_Lane), self.t_signal(Path_ts)],dim=2)
    #     return self.dropout(x)
    
    def forward(self, Path):
        x = self.path_embed(Path)
        return self.dropout(x)
