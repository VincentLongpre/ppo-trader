import numpy as np
import gym
from gym import spaces

NB_STOCK = 30
INITIAL_ASSET = 1000000
MAX_HOLD = 100
TRANSACTION_FEE_PERCENT = 0.01
REWARD_SCALING = 1e-4

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

    def _execute_action(self, actions):
        actions = actions * MAX_HOLD
        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

        # Sell stocks
        for index in sell_index:
            if actions[index] < 0:
                amount = min(abs(actions[index]), self.state[index + NB_STOCK + 1])
                self.state[0] += self.state[index + 1] * amount * (1 - TRANSACTION_FEE_PERCENT)
                self.state[index+NB_STOCK + 1] -= amount
                self.cost += self.state[index + 1] * amount * TRANSACTION_FEE_PERCENT
                self.trades += 1

        # Buy stocks
        for index in buy_index:
            if actions[index] > 0:
                available_amount = self.state[0] // self.state[index + 1]
                amount = min(available_amount, actions[index])
                self.state[0] -= self.state[index + 1] * amount * (1 + TRANSACTION_FEE_PERCENT)
                self.state[index + NB_STOCK + 1] += amount
                self.cost += self.state[index + 1] * amount * TRANSACTION_FEE_PERCENT
                self.trades += 1

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # Create graphs and have fun
            pass

        else:
            begin_total_asset = self.state[0]+ \
                                sum(np.array(self.state[1:(NB_STOCK + 1)]) * np.array(self.state[(NB_STOCK + 1):(NB_STOCK * 2 + 1)]))

            self._execute_action(actions)

            self.day += 1
            self.data = self.df.loc[self.day,:]
            self.state = [self.state[0]] + \
                         self.data.adjcp.values.tolist() + \
                         list(self.state[(NB_STOCK + 1):(NB_STOCK * 2 + 1)]) + \
                         self.data.macd.values.tolist() + \
                         self.data.rsi.values.tolist() + \
                         self.data.cci.values.tolist() + \
                         self.data.adx.values.tolist()

            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(NB_STOCK + 1)]) * np.array(self.state[(NB_STOCK + 1):(NB_STOCK * 2 + 1)]))
            self.asset_memory.append(end_total_asset)

            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * REWARD_SCALING

        return self.state, self.reward, self.terminal, {}
    
    def reset(self):
        self.asset_memory = [INITIAL_ASSET]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []

        self.state = [INITIAL_ASSET] \
            + self.data.adjcp.values.tolist() \
            + [0] * NB_STOCK \
            + self.data.macd.values.tolist() \
            + self.data.rsi.values.tolist() \
            + self.data.cci.values.tolist() \
            + self.data.adx.values.tolist()
        
        return self.state