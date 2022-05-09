'''
This script handles the training process.
'''

import argparse
import pickle as pkl
import torch
import time
from models import TPR
from utils import process
from torch.utils.data import Dataset, DataLoader
import numpy as np


def parse_option():
    """command-line interface"""
    parser = argparse.ArgumentParser(description="PyTorch Implementation of TPR")
    parser.add_argument('--gpu', type=int, default=0,
                        help='set GPU')
    """testing params"""
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='dim of node embedding (default: 512)')

    args = parser.parse_args()

    return args


print('------Loading dataset-----')

class TestDataset(Dataset):
    def __init__(self):
        path, NP1, NP2, NP3, NP4, TS, TPS1, TPS2, TNS1, TNS2, Time = pkl.load(
    open('./data/data_DT210821_DSame_Aalborg_weakly_train.pkl', 'rb'))

        self.path_mask = torch.FloatTensor(process.mask_enc(path))
        self.path = torch.LongTensor(path)
        self.TS = torch.LongTensor(TS)

        self.len = path.shape[0]

    def __getitem__(self,index):
        return self.path[index], self.path_mask[index], self.TS[index]
    def __len__(self):
        return self.len

#==========test epoch======#
def test_epoch(model, path, path_mask,ts):
    ''' Epoch operation in evaluation phase '''


    with torch.no_grad():

        src_embed = model.embed(path, path_mask, ts)

    return src_embed

if __name__ == '__main__':

    args = parse_option()
    testdataset = TestDataset()
    DataLoader_Test= DataLoader(dataset=testdataset, batch_size = 1, shuffle = False, drop_last = False)

    node2vec = pkl.load(open('./data/road_network_200703_128.pkl','rb')) ###512
    temporal2vec = pkl.load(open('./data/Temporal_Graph_2017_210823.pkl','rb')) ###128
    node2vec = torch.FloatTensor(node2vec)
    temporal2vec = torch.FloatTensor(temporal2vec)

    model = TPR(
            node2vec= node2vec,
            Index2traffic=temporal2vec,
            input_size =args.hidden_size,
            hidden_size = args.hidden_size,
            n_layers = args.n_layers,
            dropout = args.dropout)

    print('Model Loading')
    model.load_state_dict(torch.load('./data/best_train.pkl'))

    print('Starting Testing')
    prediction =[]
    embedding =[]
    for i, testdata in enumerate(DataLoader_Test,0):
        path, path_mask, ts =testdata
        train_losses =[]
        if torch.cuda.is_available():
            print(i)
            torch.cuda.set_device(args.gpu)
            model.cuda()
            path = path.cuda()
            path_mask = path_mask.cuda()
            ts = ts.cuda()

            src_embed = test_epoch(model, path, path_mask, ts)

            embedding.append(src_embed.cpu().numpy())
    pkl.dump(np.array(embedding), open('path_embedding_Aal_notemporal_train.pkl', 'wb'))


