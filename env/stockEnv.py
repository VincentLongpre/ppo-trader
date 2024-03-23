import numpy as np
import pandas as pd
import gym
from gym import spaces

NB_STOCK = 30
INITIAL_ASSET = 1000000

class StockEnv(gym.Env):
    def __init__(self, dataframe, day=0, turbulance_tresh=0) -> None:
        self.day = day
        self.dataframe = dataframe
        self.turbulance_tresh = turbulance_tresh
        self.terminal = False 

        self.data = dataframe.iloc[day]

        self.action_space = spaces.Box(-1, 1, (NB_STOCK,))
        self.observation_space = spaces.Box(0, np.inf, (6 * NB_STOCK + 1,))

        self.state = [INITIAL_ASSET] \
            + self.data.adjcp.values.tolist() \
            + [0] * NB_STOCK \
            + self.data.macd.values.tolist() \
            + self.data.rsi.values.tolist() \
            + self.data.cci.values.tolist() \
            + self.data.adx.values.tolist()

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
        pass