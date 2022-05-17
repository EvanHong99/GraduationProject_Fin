# -*- coding=utf-8 -*-
# @File     : data_loader.py
# @Time     : 2022/2/27 22:49
# @Author   : EvanHong
# @Email    : 939778128@qq.com

from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import myconfig
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from mytools import get_device
from copy import deepcopy

device,use_gpu=get_device()


class RNNDataLoader(Dataset):
    """
    gru data loader

    需要先将数据存储成需要的格式，再由该类读取

    """

    def __init__(self, file_path:list,y_type:str='tag',n_steps_ahead=myconfig.N_STEPS_AHEAD,sequence_length=myconfig.SEQUENCE_LENGTH,use_cols=myconfig.USE_COLS,norm_cols=myconfig.NORM_COLS,use_real=myconfig.USE_REAL):
        """
        ,open,close,high,low,volume,money,predicted,open_pct,close_pct,high_pct,low_pct
        2016-01-05,9.85,9.97,10.55,9.7,41932750.0,422708448.0,1.2121212121212122,-0.15883859948761747,-0.07513914656771792,-0.1081994928148774,-0.1001855287569573


        :param file_path:
        :param y_type: list['tag','logits']
            加载情感分类数据的类型，只完成了tag，todo logits
        """
        assert len(file_path)==2
        self.n_steps_ahead=n_steps_ahead
        self.sequence_length=sequence_length
        self.norm_cols=norm_cols
        self.use_cols=use_cols
        self.use_real=use_real

        
        dropped=['tic']
        # dropped=['Unnamed: 0','tic']
        self.stockdata = pd.read_csv(file_path[0], header=0,index_col='date').drop(columns=dropped)
        self.stockalldata = pd.read_csv(file_path[0].replace('_train','_alldata'), header=0,index_col='date').drop(columns=dropped)
        self.stockdata1 = pd.read_csv(file_path[1], header=0,index_col='date').drop(columns=dropped)
        # 这里不能转换成datetime，因为dataloader不允许
        # stockdata.index=pd.to_datetime(stockdata.index)
        self.y_list=self.stockdata1[myconfig.GRU_PRED_TARGET].values.tolist()
        self.pred_dates=self.stockdata1.index.tolist()

        # 需要在整数据集上进行归一化，否则确实会受到每个区域单独归一化时导致模型输入和输出不匹配，即在valid上归一化后，模型会以为回到了train的某一个很前面的时刻来进行预测，因此预测的股价会较低
        # 哪怕进行shuffle也无法改善这个问题，因为train的时候股价最高就30，模型没见过世面推不出会涨那么高
        t=self.stockalldata[norm_cols]
        self.stockdata[norm_cols]=(t-t.min())/(t.max()-t.min()+1e-8)
        # 加入y的数据
        if self.use_real:
            self.stockdata.loc[:,'real']=(np.array(self.y_list)/10).tolist()
        self.x_list =self.stockdata[use_cols].values.tolist()


    def __get_all__(self,df=False):
        if df:
            return self.stockdata,self.stockdata1,self.stockalldata
        return self.x_list,self.y_list,self.pred_dates

    def __getitem__(self, index):
        # self.dates可能是str类型
        # return self.x_list[index:index+self.sequence_length], self.y_list[index:index+self.sequence_length],self.pred_dates[index:index+self.sequence_length]
        return self.x_list[index:index+self.sequence_length], self.y_list[index+self.n_steps_ahead+self.sequence_length-1],self.pred_dates[index+self.n_steps_ahead+self.sequence_length-1]


    def __len__(self):
        return len(self.y_list)-(self.n_steps_ahead+self.sequence_length-1)


if __name__ == '__main__':
    # mldl = MLDataLoader('002709')
    # print(len(mldl.df_alldata.columns))
    pass