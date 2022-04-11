import random
import json
from gym import spaces
from gym.spaces import Discrete,Box,Dict
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
"""最大账户余额"""
MAX_NUM_SHARES = 2147483647
"""最大股票数量"""
MAX_SHARE_PRICE = 5000
"""最高股价"""
MAX_OPEN_POSITIONS = 5
"""
敞口头寸 (open position) 
指尚未对冲或交割的头寸，即持仓者承诺要买入或卖出某些未履约的商品，或买入或卖出没有相反方向相配的商品。
"""
MAX_STEPS = 20000
"""最多操作数量"""
INITIAL_ACCOUNT_BALANCE = 10000
"""初始账户余额"""

SHARES_PER_HAND=100#一手100股


class AgentState(object):
    """模拟交易员的状态

    Args:
        object (_type_): _description_
    """
    def __init__(self,money,) -> None:
        self.money=money


class PairsTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self,df):
        super(PairsTradingEnv, self).__init__()
        self.df = df

        
        self.action_space = spaces.Dict({
            "action":spaces.Discrete(3, start=-1),
            "shares":spaces.Discrete(10000,start=0),#一次最多买进10000手
        })

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

        self.profit_list=[]
        ...
    def step(self, action):
        # TODO
        ...
    def reset(self):
        # TODO
        ...
    def render(self, mode='human'):
        # TODO
        ...
    def close(self):
        # TODO