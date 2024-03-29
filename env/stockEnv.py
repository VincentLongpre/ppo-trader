import numpy as np
import pandas as pd
import gymnasium as gym
from gym.utils import seeding
from gym import spaces
import matplotlib
import matplotlib.pyplot as plt
import pickle

class StockEnv(gym.Env):

    def __init__(self, df, day=0, turbulence_threshold=140,
                 hmax=100, initial_account_balance=1000000, stock_dim=30, transaction_fee_percent=0.001,
                 reward_scaling=1e-4, seed = 42, split='train'):

        self.day = day
        self.df = df
        self.action_space = spaces.Box(low=-1, high=1, shape=(stock_dim,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(181,))

        self.hmax = hmax
        self.initial_account_balance = initial_account_balance
        self.stock_dim = stock_dim
        self.transaction_fee_percent = transaction_fee_percent
        self.reward_scaling = reward_scaling

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state
        self.state = [initial_account_balance] + \
                     self.data.adjcp.values.tolist() + \
                     [0] * stock_dim + \
                     self.data.macd.values.tolist() + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist()
        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        # memorize all the total balance change
        self.asset_memory = [initial_account_balance]
        self.rewards_memory = []

        np.random.seed(seed)

        # self.iteration = iteration
        self.split = split

    def sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.turbulence < self.turbulence_threshold:
            if self.state[index + self.stock_dim + 1] > 0:
                # update balance
                self.state[0] += \
                    self.state[index + 1] * min(abs(action), self.state[index + self.stock_dim + 1]) * \
                    (1 - self.transaction_fee_percent)

                self.state[index + self.stock_dim + 1] -= min(abs(action), self.state[index + self.stock_dim + 1])
                self.cost += self.state[index + 1] * min(abs(action), self.state[index + self.stock_dim + 1]) * \
                             self.transaction_fee_percent
                self.trades += 1
            else:
                pass
        else:
            # if turbulence goes over threshold, just clear out all positions
            if self.state[index + self.stock_dim + 1] > 0:
                # update balance
                self.state[0] += self.state[index + 1] * self.state[index + self.stock_dim + 1] * \
                                 (1 - self.transaction_fee_percent)
                self.state[index + self.stock_dim + 1] = 0
                self.cost += self.state[index + 1] * self.state[index + self.stock_dim + 1] * \
                             self.transaction_fee_percent
                self.trades += 1
            else:
                pass

    def buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        if (self.split != 'train') and (self.turbulence < self.turbulence_threshold):
            available_amount = self.state[0] // self.state[index + 1]

            # print('available_amount:{}'.format(available_amount))

            # update balance
            self.state[0] -= self.state[index + 1] * min(available_amount, action) * \
                             (1 + self.transaction_fee_percent)

            self.state[index + self.stock_dim + 1] += min(available_amount, action)

            self.cost += self.state[index + 1] * min(available_amount, action) * \
                         self.transaction_fee_percent
            self.trades += 1
        else:
            # if turbulence goes over threshold, just stop buying
            pass

    def step(self, actions):

        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            plt.plot(self.asset_memory, 'r')
            # plt.savefig('results/account_value_validation_{}.png'.format(self.iteration))
            plt.close()
            df_total_value = pd.DataFrame(self.asset_memory)
            # df_total_value.to_csv('results/account_value_validation_{}.csv'.format(self.iteration))
            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(self.stock_dim + 1)]) * np.array(
                                  self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]))

            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
                     df_total_value['daily_return'].std()

            return self.state, self.reward, self.terminal, {}

        else:

            actions = actions * self.hmax

            actions = (actions.astype(int))

            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-self.hmax] * self.stock_dim)
            begin_total_asset = self.state[0] + \
                                sum(np.array(self.state[1:(self.stock_dim + 1)]) * np.array(
                                    self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]))

            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self.sell_stock(index, actions[index])

            for index in buy_index:
                self.buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.turbulence = self.data['turbulence']

            self.state = [self.state[0]] + \
                         self.data.adjcp.values.tolist() + \
                         list(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]) + \
                         self.data.macd.values.tolist() + \
                         self.data.rsi.values.tolist() + \
                         self.data.cci.values.tolist() + \
                         self.data.adx.values.tolist()

            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(self.stock_dim + 1)]) * np.array(
                                  self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]))
            self.asset_memory.append(end_total_asset)

            self.reward = end_total_asset - begin_total_asset

            self.rewards_memory.append(self.reward)

            self.reward = self.reward * self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_account_balance]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        # initiate state
        self.state = [self.initial_account_balance] + \
                     self.data.adjcp.values.tolist() + \
                     [0] * self.stock_dim + \
                     self.data.macd.values.tolist() + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist()

        return self.state