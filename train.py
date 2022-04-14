# -*- coding: utf-8 -*-
# @File     : train.py
# @Time     : 2022/2/27 22:47
# @Author   : EvanHong
# @Email    : 939778128@qq.com
from abc import ABC,abstractmethod
import torch
import time
import myconfig
from data_loader import RNNDataLoader
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from models import GRUNet
from mytools import get_device
import os
import matplotlib.pyplot as plt
import json


# 检测当前环境设备
device,use_gpu = get_device()


class BaseTrainer(object):
    def __init__(self) -> None:
        """加载data loader、初始化网络、加载预处理模型
        """
        self.loss_history = [1e20]
        self.iter = 0
        # 使用tensorboard
        self.writer = None
        pass
    
    @abstractmethod
    def train_one_epoch(self):
        raise NotImplementedError


    @abstractmethod
    def test_one_epoch(self):
        raise NotImplementedError

    @abstractmethod
    def save_model(self):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError

'''
1ok
2ok
3ok
4ok
5random init
6no
7ok

    4.1 LSTM 参数调优

（1）数据准备、预处理

神经网络具有很强的学习能力（也很容易过拟合），更适用于大规模数据集，因此需要准备大量、高质量并且带有干净标签的数据。
从激活函数（sigmoid、tanh）可以看出，模型的输出绝对值一般在 0~1 之间，因此需要对数据进行归一化处理。常见的方法有：

1、Min-Max Normalization： [公式]

2、Average Normalization： [公式]

3、log function： [公式]

1、2 属于线性归一化，缺点是当有新数据加入时，可能导致 max 和 min 的变化，需要重新定义。3 属于非线性归一化，经常用在数据分化比较大的场景，有些数值很大，有些很小。

与归一化相近的概念是标准化：

Z-score规范化： [公式]

什么时候用归一化？什么时候用标准化？

1、如果对输出结果范围有要求，用归一化。

2、如果数据较为稳定，不存在极端的最大最小值，用归一化。

3、如果数据存在异常值和较多噪音，用标准化，可以间接通过中心化避免异常值和极端值的影响。

（2）批处理

神经网络一般不会一个一个的训练样本，通常采用 minibatch 的方法一次训练一批数据，但不要使用过大的批处理，因为有可能导致低效和过拟合。

（3）梯度归一化、梯度剪裁

因为采用了批处理，因此计算出来梯度之后，要除以 minibatch 的数量。如果训练 RNN 或者LSTM，务必保证 gradient 的 norm 被约束在 5、10、15（前提还是要先归一化gradient），这一点在 RNN 和 LSTM 中很重要。在训练过程中，最好可以检查下梯度。

（4）学习率

学习率是一个非常重要的参数，学习率太大将会导致训练过程非常不稳定甚至失败。太小将影响训练速度，通常设置为 0.1~0.001。

（5）权值初始化

初始化参数对结果的影响至关重要，常见的初始化方法包括：

1、常量初始化：把权值或者偏置初始化为一个自定义的常数。

2、高斯分布初始化：需要给定高斯函数的均值与标准差。

3、xavier 初始化：对于均值为 0，方差为（1 / 输入的个数） 的均匀分布，如果我们更注重前向传播，可以选择 fan_in，即正向传播的输入个数；如果更注重反向传播，可以选择 fan_out, 因为在反向传播的时候，fan_out 就是神经元的输入个数；如果两者都考虑，就选 average = (fan_in + fan_out) /2。对于 Relu 激活函数，xavier 初始化很适合。

在权值初始化的时候，可以多尝试几种方法。此外，如果使用 LSTM 来解决长时依赖的问题，遗忘门初始化 bias 的时候要大一点（大于 1.0）。

（6）dropout

dropout 通过在训练的时候屏蔽部分神经元的方式，使网络最终具有较好的效果，相比于普通训练，需要花费更多的时间。记得在测试的时候关闭 dropout。LSTM 的 dropout 只出现在同一时刻多层隐层之间，不会出现在不同时刻之间（如果 dropout 跨越不同时刻，将导致随时间传递的状态信息丢失）。

（7）提前终止

在训练的过程中，通常训练误差随着时间推移逐渐减小，而验证误差先减小后增大。期望的训练效果是：在训练集和验证集的效果都很好。训练集和验证集的 loss 都在下降，并且差不多在同个地方稳定下来。采用提前终止的方法，可以有效防止过拟合。

'''
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

class TrainGRU(BaseTrainer):
    """用来调用data loader和模型进行训练

    Args:
        object (_type_): _description_
    """
    def __init__(self,train_path:list,test_path:list,epoch,batch_size,lr,momentum,save_loss_threshold,model_save_path) -> None:
        super(TrainGRU,self).__init__()
        self.train_path=train_path
        self.test_path=test_path #这里的test是指valid
        self.epoch=epoch
        self.batch_num=batch_size
        self.save_loss_threshold=save_loss_threshold
        self.model_save_path=model_save_path
        self.lr=lr
        self.momentum=momentum
        print(f"GRU_BATCH_SIZE {myconfig.GRU_BATCH_SIZE}")
        # 不应该shuffle因为是时间序列
        dl=RNNDataLoader(self.train_path,'tag',myconfig.N_STEPS_AHEAD,myconfig.SEQUENCE_LENGTH)
        # todo
        self.train_loader=DataLoader(dl,drop_last =False,batch_size=myconfig.GRU_BATCH_SIZE,shuffle=myconfig.SHUFFLE_TRAIN)
        dl1=RNNDataLoader(self.test_path,'tag',myconfig.N_STEPS_AHEAD,myconfig.SEQUENCE_LENGTH)
        self.test_loader=DataLoader(dl1,drop_last =False,batch_size=myconfig.GRU_BATCH_SIZE,shuffle=False)

        features=len(dl.use_cols)
        print(f"features {features}")
        self.model=GRUNet(input_size=features*myconfig.SEQUENCE_LENGTH,hidden_size=myconfig.GRU_HIDDEN_SIZE,output_size=1,n_layers=myconfig.GRU_LAYERS)
        # self.model=GRUNet(input_size=features*myconfig.SEQUENCE_LENGTH,hidden_size=myconfig.GRU_HIDDEN_SIZE,output_size=1,n_layers=myconfig.GRU_LAYERS)
        self.model=self.model.to(device)
        self.loss_fn=nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.writer=SummaryWriter(comment=f"_GRU_{'_'.join([i.split('/')[-1] for i in self.train_path])}_NSA{myconfig.N_STEPS_AHEAD}_SL{myconfig.SEQUENCE_LENGTH}_SFL_{myconfig.SHUFFLE_TRAIN}")
        self.loss_history=[]

        self.prt=3

    def train_one_epoch(self,epoch):
        # print(f'train--Epoch: {epoch + 1}')
        start_time = time.time()
        size = len(self.train_loader)

        print_step = 100
        writer_step = 10
        self.model.train()
        pred=[]
        real=[]
        # dates=[]
        for i,(x,y,date) in enumerate(self.train_loader):
            self.iter += myconfig.GRU_BATCH_SIZE
            x=to_shaped_tensor(x,'x')# x seq_len*batches*feats
            x=torch.as_tensor(x).float().to(device)
            y=to_shaped_tensor(y,'y')
            y=torch.as_tensor(y).float().to(device)#y: seq_len*batches

            # output is a value
            output=self.model(x)
            loss=self.loss_fn(output,y)
            # if i >size-3 :
            #     print("train shape ",x.shape)
            #     print(f"x {x}")
            #     print(f"y {y}")
            #     print(f'output.data.cpu().numpy() {output.data.cpu().numpy()}')
            # pred.append(output.data.cpu().numpy()[0][0])
            # real.append(y.data.cpu().numpy()[0])
            pred.extend(output.data.cpu().numpy()[0][:myconfig.GRU_BATCH_SIZE])
            real.extend(y.data.cpu().numpy()[:myconfig.GRU_BATCH_SIZE])


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.iter % writer_step == 0:
                self.writer.add_scalar(tag='train_loss', scalar_value=loss.mean(), global_step=epoch)
                # self.writer.add_scalar(tag='train_loss', scalar_value=loss.mean(), global_step=epoch)

            if i % print_step == 0:
                print("train epoch:[%d|%d] [%d|%d] loss:%f" % (
                epoch + 1, self.epoch, i, len(self.train_loader), loss.mean()))
        plt.plot(real)
        plt.plot(pred)
        plt.show()
        # print(f'train epoch{epoch} ' + "time:%.3f" % (time.time() - start_time))

    def test_one_epoch(self,epoch):
        # print(f'test-Epoch: {epoch + 1}')
        start_time = time.time()
        self.model.eval()
        total = 0
        test_loss = 0
        num_batches = len(self.test_loader)
        # num_batches 34, num_batches*batch_size==length of data
        # print(f'num_batches {num_batches}')
        predicted=[]
        real=[]
        dates=[]
        with torch.no_grad():
            # 在数据集上完整test一遍
            for i,(x,y,date) in enumerate(self.test_loader):
         
                x=to_shaped_tensor(x,'x')# x seq_len*batches*feats
                x=torch.as_tensor(x).float().to(device)
                y=to_shaped_tensor(y,'y')
                y=torch.as_tensor(y).float().to(device)#y: seq_len*batches

                output=self.model(x) # input must have 3 dimensions  
                loss=self.loss_fn(output,y)
                test_loss+=loss
                # no batch
                # predicted.append(output.data.cpu().numpy()[0][0])
                # # real.append(y.data.cpu().numpy()[0][0])
                # real.append(y.data.cpu().numpy()[0])
                # dates.append(date[0])
                # with batch
                predicted.extend(output.data.cpu().numpy()[0][:myconfig.GRU_BATCH_SIZE])
                real.extend(y.data.cpu().numpy()[:myconfig.GRU_BATCH_SIZE])
                dates.extend(list(date)[:myconfig.GRU_BATCH_SIZE])
            # except:
            #     print(f"i {ii*myconfig.GRU_BATCH_SIZE}")
            #     print(predicted.tolist())
        # print(f'test epoch{epoch} ' + "time:%.3f" % (time.time() - start_time))
        test_loss /= num_batches
        print("test loss:%.7f" % (test_loss))
        self.writer.add_scalar(tag='test_loss'+'_'+myconfig.GRU_PRED_TARGET, scalar_value=test_loss, global_step=epoch)

        return test_loss,predicted,real,dates


    def save_model(self,loss):
        """
        保存模型
        :param model_save_path:
        :return:
        """
        # 保存最新的模型
        print("saving model")
        if self.model_save_path[-1]=='/':
            self.model_save_path=self.model_save_path[:-1]
        path = self.model_save_path + "/GRUmodel_"+str(time.strftime("%Y-%m-%d-%H-%M-%S")+f"_{'_'.join([i.split('/')[-1] for i in self.train_path])}_NSA{myconfig.N_STEPS_AHEAD}_SL{myconfig.SEQUENCE_LENGTH}__{loss}")
        os.mkdir(path)
        torch.save(self.model, path+"/model.pt")
        torch.save(self.model.state_dict(),path+f'/GRUmodel.pth')
        with open(path+"/config.json",'w',encoding='utf-8') as fw:
            a={
            "model":str(self.model)
            ,"SHUFFLE_TRAIN":myconfig.SHUFFLE_TRAIN
            ,"GRU_TRAIN_PATH":myconfig.GRU_TRAIN_PATH
            ,"GRU_VALID_PATH":myconfig.GRU_VALID_PATH
            ,"GRU_TEST_PATH":myconfig.GRU_TEST_PATH
            ,"INPUT_STOCK":myconfig.INPUT_STOCK
            ,"TARGET_STOCK":myconfig.TARGET_STOCK
            ,"GRU_STOCK_CLOSE_ALLDATA_PATH":myconfig.GRU_STOCK_CLOSE_ALLDATA_PATH
            ,"GRU_ALLDATA_PATH":myconfig.GRU_ALLDATA_PATH
            ,"SEQUENCE_LENGTH":myconfig.SEQUENCE_LENGTH
            ,"N_STEPS_AHEAD":myconfig.N_STEPS_AHEAD
            ,"GRU_BATCH_SIZE":myconfig.GRU_BATCH_SIZE
            ,"GRU_HIDDEN_SIZE":myconfig.GRU_HIDDEN_SIZE 
            ,"GRU_LAYERS":myconfig.GRU_LAYERS 
            ,"GRU_LR":myconfig.GRU_LR 
            ,"GRU_MOMENTUM":myconfig.GRU_MOMENTUM
            ,"GRU_SAVE_LOSS_THRESHOLD":myconfig.GRU_SAVE_LOSS_THRESHOLD
            ,"USE_COLS":myconfig.USE_COLS
            ,"NORM_COLS":myconfig.NORM_COLS,
            "USE_REAL":myconfig.USE_REAL,
            'GRU_PRED_TARGET':myconfig.GRU_PRED_TARGET
            }
            json.dump(a,fw)

        print('-' * 100)
        print(f'model save to {path}')
        print('-' * 100)


    def run(self,epoch=None):
        """
        多次迭代训练
        :return:
        """
        if epoch is None:
            epoch=self.epoch
        print(f'begin train ... {epoch} use sentiment {myconfig.USE_SENTIMENT}')
        print(self.model)
        scheduler=ReduceLROnPlateau(self.optimizer,patience=5,verbose=True)
        """test origin model，确保模型是在优化"""
        # test_loss=self.test_one_epoch(0)
        # print(f"origin test loss {test_loss}")
        test_loss=0
        for e in range(epoch):
            print("-"*100)
            self.train_one_epoch(e)
            test_loss,predicted,real,dates=self.test_one_epoch(e)

            # if len(self.acc)==1 or (acc > 1.01*max(self.acc) and acc > self.save_acc_threshold):
            # self.loss_history.append(test_loss)
            # if (test_loss < 0.99*max(self.loss_history[:-1]) and test_loss < self.save_loss_threshold):
            #     self.save_model()
            scheduler.step(test_loss)
            for param_group in self.optimizer.param_groups:
                print(f"learning rate {param_group['lr']}")
                self.writer.add_scalar(tag='lr', scalar_value=param_group['lr'], global_step=e)
        self.save_model(test_loss)
        return predicted,real,dates
