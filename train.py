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


def parse_option():
    """command-line interface"""
    parser = argparse.ArgumentParser(description="PyTorch Implementation of TPR")
    parser.add_argument('--gpu', type=int, default=0,
                        help='set GPU')
    """training params"""
    parser.add_argument('--save_model', default='./data/best_train')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='dim of node embedding (default: 512)')
    parser.add_argument('--epoch_flag', type=int, default=30, help=' early stopping (default: 20)')
    parser.add_argument('--nb_epochs', type=int, default=500,
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--l2_coef', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--dropout', type=float, default=0.4)


    args = parser.parse_args()
    
    return args


print('------Loading dataset-----')

class TrainDataset(Dataset):
    def __init__(self):
        Path, NP1, NP2, NP3, NP4, TS, TPS1, TPS2, TNS1, TNS2, Time=pkl.load(
            open('./data/data_sample_train.pkl','rb'))

    # Path: query path
    # Path_RT:  road type sequence
    # Path_OW: one way sequence
    # Path_Lane: No. Lane sequence
    # Path_TS: Traffic Signal sequence
    # TS: weak label for query path
    # TPS1: positive weak label1
    # TPS2: positive weak label2
    # TNS1: negative weak label1
    # TNS2: negative weak label2
    # NP1-NP4 different pathS
    # Time: travel time


        self.path_mask = torch.FloatTensor(process.mask_enc(Path))

        self.NP1_mask = torch.FloatTensor(process.mask_enc(NP1))
        self.NP2_mask = torch.FloatTensor(process.mask_enc(NP2))
        self.NP3_mask = torch.FloatTensor(process.mask_enc(NP3))
        self.NP4_mask = torch.FloatTensor(process.mask_enc(NP4))

        self.path = torch.LongTensor(Path)
        ####negative path

        self.NP1 = torch.LongTensor(NP1)
        self.NP2 = torch.LongTensor(NP2)
        self.NP3 = torch.LongTensor(NP3)
        self.NP4 = torch.LongTensor(NP4)

        self.TS = torch.LongTensor(TS)
        self.TPS1 = torch.LongTensor(TPS1)
        self.TPS2 = torch.LongTensor(TPS2)
        self.TNS1 = torch.LongTensor(TNS1)
        self.TNS2 = torch.LongTensor(TNS2)

        self.len = Path.shape[0]

    def __getitem__(self,index):
        return self.path[index],self.path_mask[index], self.NP1[index], self.NP1_mask[index], self.NP2[index], self.NP2_mask[index],self.NP3[index], self.NP3_mask[index], self.NP4[index], self.NP4_mask[index], self.TS[index], self.TPS1[index], self.TPS2[index], self.TNS1[index], self.TNS2[index]
    def __len__(self):
        return self.len


class ValDataset(Dataset):
    def __init__(self):
        Path, NP1, NP2, NP3, NP4, TS, TPS1, TPS2, TNS1, TNS2, Time=pkl.load(
            open('./data/data_sample_val.pkl','rb'))

    # Path: query path

    # TS: weak label for query path
    # TPS1: positive weak label1
    # TPS2: positive weak label2
    # TNS1: negative weak label1
    # TNS2: negative weak label2
    # NP1-NP4 different pathS
    # Time: travel time


        self.path_mask = torch.FloatTensor(process.mask_enc(Path))

        self.NP1_mask = torch.FloatTensor(process.mask_enc(NP1))
        self.NP2_mask = torch.FloatTensor(process.mask_enc(NP2))
        self.NP3_mask = torch.FloatTensor(process.mask_enc(NP3))
        self.NP4_mask = torch.FloatTensor(process.mask_enc(NP4))

        self.path = torch.LongTensor(Path)
   
        ####negative path

        self.NP1 = torch.LongTensor(NP1)
        self.NP2 = torch.LongTensor(NP2)
        self.NP3 = torch.LongTensor(NP3)
        self.NP4 = torch.LongTensor(NP4)

        self.TS = torch.LongTensor(TS)
        self.TPS1 = torch.LongTensor(TPS1)
        self.TPS2 = torch.LongTensor(TPS2)
        self.TNS1 = torch.LongTensor(TNS1)
        self.TNS2 = torch.LongTensor(TNS2)

        self.len = Path.shape[0]

    def __getitem__(self,index):
        return self.path[index],self.path_mask[index], self.NP1[index], self.NP1_mask[index], self.NP2[index], self.NP2_mask[index],self.NP3[index], self.NP3_mask[index], self.NP4[index], self.NP4_mask[index], self.TS[index], self.TPS1[index], self.TPS2[index], self.TNS1[index], self.TNS2[index]
    def __len__(self):
        return self.len

#==========train epoch and val epoch======#
def train_epoch(model, path, path_mask, NP1, NP1_mask, NP2, NP2_mask ,NP3, NP3_mask,
                NP4, NP4_mask, TS,TPS1,TPS2,TNS1,TNS2,optimizer):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss = 0

    # forward
    optimizer.zero_grad()
    loss_path, loss_node= model(path,  path_mask, NP1, NP1_mask, NP2, NP2_mask, NP3, NP3_mask, NP4, NP4_mask, TS,TPS1,TPS2,TNS1,TNS2)

    loss = loss_path + loss_node

    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    return total_loss, loss, loss_path, loss_node

def eval_epoch(model, path, path_mask, NP1, NP1_mask, NP2, NP2_mask ,NP3, NP3_mask,
                NP4, NP4_mask, TS,TPS1,TPS2,TNS1,TNS2):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss = 0

    with torch.no_grad():

        # forward
        loss_path, loss_node = model(path, path_mask, NP1, NP1_mask, NP2, NP2_mask, NP3, NP3_mask, NP4, NP4_mask,TS,TPS1,TPS2,TNS1,TNS2)


        loss = loss_path + loss_node
        total_loss += loss.item()

    return total_loss,loss, loss_path, loss_node

if __name__ == '__main__':

    best = 1e9
    args = parse_option()

    save_model =args.save_model
    #========= Loading Dataset =========#
    traindataset = TrainDataset()
    valdataset = ValDataset()
    DataLoader_Train= DataLoader(dataset=traindataset, batch_size = args.batch_size, shuffle = True, drop_last = True)
    DataLoader_Val = DataLoader(dataset=valdataset, batch_size = args.batch_size, shuffle = True, drop_last = True)


    node2vec = pkl.load(open('./data/road_network_200703_128.pkl','rb')) 
    temporal2vec = pkl.load(open('./data/Temporal_Graph_2017_210823.pkl','rb'))
    node2vec = torch.FloatTensor(node2vec)
    temporal2vec = torch.FloatTensor(temporal2vec)

    model = TPR(
            node2vec= node2vec,
            temporal2vec=temporal2vec,
            input_size =args.hidden_size*2,
            hidden_size = args.hidden_size,
            n_layers = args.n_layers,
            dropout = args.dropout)
    
    ##pre-training load
    # pretrained_dict = torch.load('./data/best_train_Aalborg_Sim_2p4n_Len18_CL_Stage17.pkl')
    # model_dict =model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
    
    print('===> Starting Training')
    for epoch in range(args.nb_epochs):

        print('[ Epoch', epoch, ']')

        for i, traindata in enumerate(DataLoader_Train,0):
            path,  path_mask, NP1, NP1_mask, NP2, NP2_mask, NP3, NP3_mask, NP4, NP4_mask, TS, TPS1, TPS2, TNS1, TNS2 = traindata
            train_losses =[]
            if torch.cuda.is_available():
                print('GPU available: Using CUDA')
                torch.cuda.set_device(args.gpu)
                model.cuda()
                path = path.cuda()

                path_mask = path_mask.cuda()
            
                ##negative path
                NP1         = NP1.cuda()
                NP1_mask    = NP1_mask.cuda()
                NP2         = NP2.cuda()
                NP2_mask    = NP2_mask.cuda()
                NP3         = NP3.cuda()
                NP3_mask    = NP3_mask.cuda()
                NP4         = NP4.cuda()
                NP4_mask    = NP4_mask.cuda()

                TS = TS.cuda()
                TPS1 = TPS1.cuda()
                TPS2 =TPS2.cuda()
                TNS1=TNS1.cuda()
                TNS2=TNS2.cuda()

                lbl1 =lbl1.cuda()

                start = time.time()
                train_loss1, loss_train, loss_path, loss_node= train_epoch(model, path, path_mask, NP1, NP1_mask, NP2, NP2_mask, NP3, NP3_mask, NP4, NP4_mask, TS, TPS1, TPS2, TNS1, TNS2, optimizer)
                process.print_performances_multi('Training', epoch, loss_train, loss_path, loss_node, start)
                train_losses +=[train_loss1]

            else:
                print('CPU available: Using CPU')
                device = torch.device("cpu")
                start = time.time()
                train_loss1, loss_train, loss_path, loss_node= train_epoch(model, path, path_mask, NP1, NP1_mask, NP2, NP2_mask, NP3, NP3_mask, NP4, NP4_mask, TS, TPS1, TPS2, TNS1, TNS2, optimizer)
                process.print_performances_multi('Training', epoch, loss_train, loss_path, loss_node, start)
                train_losses += [train_loss1]


        for i, valdata in enumerate(DataLoader_Val,0):
            path, path_mask, NP1, NP1_mask, NP2, NP2_mask, NP3, NP3_mask, NP4, NP4_mask, TS, TPS1, TPS2, TNS1, TNS2=valdata
            valid_losses =[]
            if torch.cuda.is_available():
                print('GPU available: Using CUDA')
                torch.cuda.set_device(args.gpu)
                model.cuda()
                path = path.cuda()
             

                path_mask = path_mask.cuda()
            
                ##negative path
                NP1         = NP1.cuda()
                NP1_mask    = NP1_mask.cuda()
                NP2         = NP2.cuda()
                NP2_mask    = NP2_mask.cuda()
                NP3         = NP3.cuda()
                NP3_mask    = NP3_mask.cuda()
                NP4         = NP4.cuda()
                NP4_mask    = NP4_mask.cuda()

                TS = TS.cuda()
                TPS1 = TPS1.cuda()
                TPS2 =TPS2.cuda()
                TNS1=TNS1.cuda()
                TNS2=TNS2.cuda()
        
               
                start = time.time()
                valid_loss, loss_val, loss_path, loss_node= eval_epoch(model, path,  path_mask, NP1, NP1_mask, NP2, NP2_mask, NP3, NP3_mask, NP4, NP4_mask, TS, TPS1, TPS2, TNS1, TNS2)
                process.print_performances_multi('Validation',epoch, loss_val, loss_path, loss_node, start)

                valid_losses += [valid_loss]

                if loss_val < best:
                    best = loss_val
                    model_name = save_model + '.pkl'
                    torch.save(model.state_dict(),model_name)
            else:
                device = torch.device("cpu")
                print('CPU available: Using CPU')
                start = time.time()
                valid_loss,loss_val, loss_path, loss_node= eval_epoch(model, path, path_mask, NP1, NP1_mask, NP2, NP2_mask, NP3, NP3_mask, NP4, NP4_mask, TS, TPS1, TPS2, TNS1, TNS2)
                process.print_performances_multi('Validation', epoch, loss_val, loss_path, loss_node, start)

                valid_losses += [valid_loss]

                if loss_val < best:
                    best = loss_val
                    model_name = save_model + '.pkl'
                    torch.save(model.state_dict(), model_name)
    print('====>Finishing Training')
