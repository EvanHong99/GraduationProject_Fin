# -*- coding=utf-8 -*-
# @File     : myconfig.py
# @Time     : 2022/3/9 20:31
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : pairs_trading
# @Description:

USE_GPU_TRAIN=True

TRAIN_PROP=0.9
VALID_PROP=0.05
TEST_PROP=0.05

THRESHOLD=0.95


# GRU stock price predict train settings
GRU_ALLDATA_PATH='./data/train/processed_full.csv'
INPUT_STOCK='600036'
TARGET_STOCK='002142'
GRU_STOCK_CLOSE_ALLDATA_PATH=[f'./data/train/{INPUT_STOCK}_train.csv',f'./data/train/{TARGET_STOCK}_train.csv']
GRU_TRAIN_PATH=[f'./data/train/{INPUT_STOCK}_train.csv',f'./data/train/{TARGET_STOCK}_train.csv']
GRU_VALID_PATH=[f'./data/train/{INPUT_STOCK}_valid.csv',f'./data/train/{TARGET_STOCK}_valid.csv']
GRU_TEST_PATH=[f'./data/train/{INPUT_STOCK}_test.csv',f'./data/train/{TARGET_STOCK}_test.csv']

USE_REAL=True
SHUFFLE_TRAIN=False
SEQUENCE_LENGTH=1 #更改需要重新生成数据
N_STEPS_AHEAD=1 #更改需要重新生成数据
GRU_BATCH_SIZE=8
GRU_HIDDEN_SIZE = 32 # 隐藏层 32
GRU_LAYERS = 2 # RNN的层数 2
GRU_LR = 1e-3
GRU_MOMENTUM=0.5
GRU_SAVE_LOSS_THRESHOLD = 1e8
GRU_DROPOUT=0.01

GRU_EPOCH = 40
USE_SENTIMENT=False

# GRU stock price predict settings
# data
#without pred sentiment and pct changes，百分比变动有负数，最好不要归一化
GRU_PRED_TARGET='close'
# GRU_PRED_TARGET='close_pct'
# GRU_PRED_TARGET='close_change'
['open_pct']
NORM_COLS=['open', 'close', 'high', 'low',
        'volume', 'money', 
       'macd','boll_ub','boll_lb','rsi_30','cci_30','dx_30','close_30_sma','close_60_sma']

['open_pct']
USE_COLS=NORM_COLS

if USE_REAL:
    USE_COLS=USE_COLS+['real']
if USE_SENTIMENT:
    NORM_COLS=["predicted"]+NORM_COLS
    USE_COLS=["predicted"]+USE_COLS

GRU_TRAINED_MODEL_ROOT='./models/trained_GRU/'
GRU_TRAINED_MODEL_PATH=GRU_TRAINED_MODEL_ROOT+"GRUmodel_2022-04-13-09-29-17_600036_train.csv_002142_train.csv_NSA1_SL1__14.743639945983887"
