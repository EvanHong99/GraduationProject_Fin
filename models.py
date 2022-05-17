# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import random
import logging
from sklearn.model_selection import cross_val_score
import torch
import warnings
from tensorboardX import SummaryWriter
from abc import *
from data_loader import *
from torch.nn.init import orthogonal_
import myconfig
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from mytools import get_device


device,use_gpu = get_device()

class GRUNet(nn.Module):
    """
    预测股票价格
    """
    def __init__(self, input_size, hidden_size, output_size, n_layers=1,batch_first=False,structure_str=None,drop_out=myconfig.GRU_DROPOUT):
        super(GRUNet, self).__init__()
        if structure_str is None:
            self.hidden_size = hidden_size
            self.n_layers = n_layers
            
            self.gru = nn.GRU(input_size, hidden_size, n_layers,batch_first=batch_first,dropout =drop_out).to(device)  # 输入维度、输出维度、层数、bidirectional用来说明是单向还是双向
            self.fc = nn.Linear(hidden_size, output_size).to(device)
            # self.dropout = nn.Dropout(p=0.05)
            """   RNN通常意义上是不能使用dropout的，因为RNN的权重存在累乘效应，如果使用dropout的话，会破坏RNN的学习过程。

            但是，Google Brain在15年专门发表了一篇文章研究这个：recurrent neural network regularization

            他们在非循环阶段使用了dropout，改善了过拟合的现象
            ————————————————
            版权声明：本文为CSDN博主「喝粥也会胖的唐僧」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
            原文链接：https://blog.csdn.net/zhou_438/article/details/108577209
            """
            # self.tanh=nn.Tanh().to(device)
        else:
            raise NotImplementedError


    def forward(self, _x):
        # print("pred"*20,_x)
        x, _ = self.gru(_x)
        # print("pred"*20,x)
        s, b, h = x.shape
        x = x.reshape(s * b, h)
        # x = self.dropout(x)
        x = self.fc(x)
        x = x.reshape(s,b)
        # x=self.tanh(x)
        return x


if __name__ == '__main__':
    pass