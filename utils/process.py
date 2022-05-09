import torch
import numpy as np
import math
import time

def LogitsLoss(l1,l2):
    b_xent = torch.nn.BCEWithLogitsLoss()
    loss_ = b_xent(l1,l2)
    return loss_


def cal_performance(res_pos,res_neg):
    
    loss = mi_loss_jsd(res_pos,res_neg)

    return loss

def mi_loss_jsd(pos,neg):
    e_pos = torch.mean(sp_func(-pos))
    e_neg = torch.mean(torch.mean(sp_func(neg),0))
    return e_pos + e_neg

def sp_func(arg):
    return torch.log(1+torch.exp(arg))


def mask_enc(input):
    src_mask = []
    for i in range(len(input)):
        submask = []
        for j in range(len(input[i])):
            subsubmask =[]
            if input[i][j] == 8497:
                subsubmask.append(0)
            else:
                subsubmask.append(1)
            submask.append(subsubmask)
        src_mask.append(submask)
    return src_mask


def print_performances_single(header, epoch, loss, start_time):
    print('  - {header:12} epoch: {epoch: d}, ppl1: {ppl1: 8.5f}, '\
            'elapse: {elapse: 3.3f} '.format(
                header=f"({header})", epoch=epoch, ppl1=loss,
            elapse=(time.time()-start_time)/60))

def print_performances_multi(header, epoch, loss, loss1, loss2, start_time):
    print('  - {header:12} epoch: {epoch: d}, loss: {loss: 8.5f}, loss_path:{loss_path: 8.5f}, loss_node:{loss_node: 8.5f}, '\
            'elapse: {elapse: 3.3f} '.format(header=f"({header})", epoch=epoch, loss=loss, loss_path=loss1, loss_node = loss2,
                elapse=(time.time()-start_time)/60))
