# -*- encoding=utf-8 -*-
import torch
import myconfig
import numpy as np


def get_device():
    device = ['cpu', 'cuda'][torch.cuda.is_available() and myconfig.USE_GPU_TRAIN]
    use_gpu=False
    if device=='cuda':
        use_gpu=True
    print(f"using {device}")
    return device,use_gpu

def concat_path(str1,str2):
    if str1[-1]!='/':
        str1=str1+'/'

    if str2[0]=='/':
        str2=str2[1:]
    return str1+str2


def to_shaped_tensor(x,variable='x'):
    """
    x: seq_len*feats*batches
    return x: seq_len*batches*feats

    """
    if variable=='x':
        tt=[]
        # s:feats*batches
        for s in x:
            t=[]
            for f in s:
                
                t.append(f.numpy())
            tt.append(np.array(t).T)

        return np.array(tt)
        # return np.array(tt).reshape(myconfig.SEQUENCE_LENGTH,myconfig.GRU_BATCH_SIZE,-1)
    else:
        t=[]
        for s in x:
            
            t.append(s.numpy())
        return np.array(t)