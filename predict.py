# -*- encoding=utf-8 -*-
# from cProfile import run
# from ctypes import Union
# from operator import index
# from black import out
import torch
import myconfig
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_loader import RNNDataLoader
from torch import nn
from mytools import *
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
from tensorboardX import SummaryWriter
from abc import *
from data_loader import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from models import *
import matplotlib.pyplot as plt
from copy import deepcopy

device,use_gpu = get_device()



class BasePredictor(object):

    def __init__(self, stock_code, exchange: str, start_date, end_date):
        self.stock_code = stock_code
        self.exchange = exchange
        self.start_date = start_date
        self.end_date = end_date

    @abstractmethod
    def align_data(self, stock_binary_data, title_data, n_days_ahead=1):
        pass

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def eval(self):
        pass


class StockPredictor(BasePredictor):
    """

    Steps:
        1. Analyse the sentiment contained in sentences
        2. train model to match sentiment and stock price trend
    """
    def __init__(self,root_path):
        with open(concat_path(root_path,'config.json'),'r',encoding='utf-8') as fw:
            print("parsing args")
            self.cfg=json.load(fw)
            print(self.cfg)
        self.model=None
        self.has_load_model=False

    def load_data(self,file_path=None):
        # 必须保证先加载模型，因为需要parse args
        assert self.model is not None and self.has_load_model
        # print
        self.file_path=file_path
        if file_path is None:
            self.file_path=self.cfg['GRU_TEST_PATH']
        dl=RNNDataLoader(self.file_path,'tag',
                self.cfg['N_STEPS_AHEAD'],
                self.cfg['SEQUENCE_LENGTH'],
                self.cfg['USE_COLS'],
                self.cfg['NORM_COLS'],
                self.cfg['USE_REAL'],
                )
        self.test_loader=DataLoader(dl,drop_last =False,batch_size=self.cfg['GRU_BATCH_SIZE'],shuffle=False)
        # print(len(self.test_loader))
        # self.data=pd.read_csv(self.cfg['GRU_TEST_PATH'][1],header=0,index_col=0)[self.cfg['USE_COLS']]
        # self.alldata=pd.read_csv(self.cfg['GRU_ALLDATA_PATH'],header=0,index_col=0)[self.cfg['USE_COLS']]
        # self.data.index=pd.to_datetime(self.data.index)
        # present_date=self.data.index[self.cfg['SEQUENCE_LENGTH']-1:-self.cfg['N_STEPS_AHEAD']]
        # pred_date=self.data.index[self.cfg['SEQUENCE_LENGTH']-1+self.cfg['N_STEPS_AHEAD']:]
        # real_y=deepcopy(self.data[self.cfg['GRU_PRED_TARGET']].iloc[self.cfg['SEQUENCE_LENGTH']-1+self.cfg['N_STEPS_AHEAD']:])
        # self.real=pd.DataFrame({
        #     "present_date":present_date,
        #     "pred_date":pred_date,
        #     "real_y":real_y,
        # })
        # self.real=self.real.set_index(['present_date'])
        self.has_load_model=False

    def norm_data(self):
        on=self.cfg['NORM_COLS']
        # t=self.alldata[on]
        # # 在全数据集上进行归一化
        # self.data[on]=(self.data[on]-t.min())/(t.max()-t.min())
        t=self.data[on]
        # 在全数据集上进行归一化
        self.data[on]=(t-t.min())/(t.max()-t.min()+1e-8)

    # def parse_args(self,model_path:str):
    #     # /content/drive/MyDrive/HYF/Graduation_CS/models/trained_GRU/GRUmodel_2022-03-28-09-21-12_clean_data_train.csv_STMTFalse_NSA1_SL1__0.0005981374415569007/GRUmodel.pth
    #     p=model_path.split('/')[-2].split('_')[-5:-2] #['STMTFalse', 'NSA1', 'SL1']
    #     self.use_sentiment=True if 'True' in p[0] else False
    #     self.n_steps_ahead=int(p[1][3:])
    #     self.sequence_length=int(p[2][2:])
    #     print(f"args: stmt {self.use_sentiment} n steps ahead {self.n_steps_ahead} seq len {self.sequence_length}")


    def load_model(self,root_path=None):
        # path 为根目录
        features=len(self.cfg['USE_COLS'])
        self.model=GRUNet(input_size=features,hidden_size=self.cfg['GRU_HIDDEN_SIZE'],output_size=1,n_layers=self.cfg['GRU_LAYERS'],drop_out=0)
        self.model.load_state_dict(torch.load(concat_path(root_path,'GRUmodel.pth'),map_location=torch.device(device)),strict=True)
        self.has_load_model=True
        print("finish loading model")

    def pred(self):
        """_summary_

        Args:
            input (torch.Tensor): 3d tensor seq_len*batches*feats
        """
        pred=[]
        real=[]
        date=[]
        for i,(x,y,d) in enumerate(self.test_loader):
            x=to_shaped_tensor(x,'x')
            x=torch.as_tensor(x).float().to(device)

            output=self.model(x)
            # print(output)
            pred.extend(output.data.cpu().numpy()[0][:self.cfg['GRU_BATCH_SIZE']])
            real.extend(y.data.cpu().numpy()[:self.cfg['GRU_BATCH_SIZE']])
            # print(y)
            # print(d)
            date.extend(list(d)[:self.cfg['GRU_BATCH_SIZE']])
        # print(pred)
        # print(real)
        print(date)
        res=pd.DataFrame(data={
            "real":real,
            "pred":pred,
        },index=date)
        print(res)

        res.index=pd.to_datetime(res.index)
        return res



if __name__=="__main__":
    # print(0)
    pass
