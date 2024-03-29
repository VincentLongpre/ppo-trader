import numpy as np
import pandas as pd
import gym
from gym import spaces

NB_STOCK = 30
INITIAL_ASSET = 1000000

HMAX_NORMALIZE = 100
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4

class StockEnv(gym.Env):
    def __init__(self, dataframe, day=0, turbulance_tresh=0) -> None:
        self.day = day
        self.dataframe = dataframe
        self.turbulance_tresh = turbulance_tresh
        self.terminal = False 

        # self.data = dataframe.iloc[day]
        self.data = dataframe.loc[day,:]

        self.action_space = spaces.Box(-1, 1, (NB_STOCK,))
        self.observation_space = spaces.Box(0, np.inf, (7 * NB_STOCK + 1,))

        self.state = [INITIAL_ASSET] \
            + self.data.close.values.tolist() \
            + [0] * NB_STOCK \
            + self.data.macd.values.tolist() \
            + self.data.rsi.values.tolist() \
            + self.data.cci.values.tolist() \
            + self.data.adx.values.tolist() \
            + self.data.vix.values.tolist()

        self.reward = 0
        self.cost = 0
        self.turbulance = 0
        self.nb_trades = 0

        self.asset_mem = [INITIAL_ASSET]
        self.reward_mem = []

    def _sell(self):
        pass
    
    def _buy(self):
        pass
    
    def step(self):
        pass
    
    def reset(self):
        self.asset_memory = [INITIAL_ASSET]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        #initiate state
        self.state = [INITIAL_ASSET] + \
                      self.data.adjcp.values.tolist() + \
                      [0]*NB_STOCK + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist() + \
                      self.data.vix.values.tolist()
        # iteration += 1 
        return self.state