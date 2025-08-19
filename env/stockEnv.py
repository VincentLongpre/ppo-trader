import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

class StockEnv(gym.Env):
    def __init__(self, dataframe, **kwargs):
        super(StockEnv, self).__init__()

        self.dataframe = dataframe
        self.terminal = False

        # Default configs
        config_defaults = dict(
            day=0,
            env_type="train",
            hmax=100,
            initial_balance=1_000_000,
            nb_stock=500,
            reward_scaling=1.0,
            seed=42,
            transaction_fee=0.001,
            turbulence_threshold=140,
            history_length=30,
            action_penalty=0.001,  # penalty for extreme actions
            reward_norm_eps=1e-8   # for reward normalization
        )
        config_defaults.update(kwargs)
        for key, value in config_defaults.items():
            setattr(self, key, value)

        # Action & observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.nb_stock,), dtype=np.float32)
        state_dim = 1 + self.nb_stock * 9 + 1
        obs_shape = (self.history_length, state_dim) if self.history_length > 1 else (state_dim,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        # Initialize
        self.reward = 0
        self.cost = 0
        self.turbulence = 0
        self.trades = 0
        self.asset_memory = [self.initial_balance]
        self.rewards_memory = []
        self.return_history = []

        self.reset(seed=self.seed)

    # ---------------------------
    # State / Observation
    # ---------------------------
    def _get_state(self):
        state = [self.state[0]] + \
                self.data.adjcp.values.tolist() + \
                list(self.state[(self.nb_stock + 1):(self.nb_stock * 2 + 1)]) + \
                self.data.macd.values.tolist() + \
                self.data.rsi.values.tolist() + \
                self.data.cci.values.tolist() + \
                self.data.adx.values.tolist() + \
                self.data.volatility.values.tolist() + \
                self.data.ma10.values.tolist() + \
                self.data.ma50.values.tolist() + \
                [self.turbulence]
        return np.array(state, dtype=np.float32)

    def _get_observation(self):
        if self.history_length > 1:
            history = self.state_history[-self.history_length:]
            pad_len = self.history_length - len(history)
            if pad_len > 0:
                zero_pad = np.zeros((pad_len, history[0].shape[0]), dtype=np.float32)
                history = np.vstack([zero_pad, history])
            return np.array(history, dtype=np.float32)
        else:
            return self.state_history[-1]

    # ---------------------------
    # Action execution
    # ---------------------------
    def _execute_action(self, actions):
        actions = np.clip(actions, -1, 1)
        real_prices = np.expm1(self.data.adjcp.values)

        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

        if self.turbulence < self.turbulence_threshold or self.env_type == 'train':
            # SELL
            for idx in sell_index:
                if actions[idx] < 0:
                    amount = min(abs(actions[idx]) * self.hmax, self.state[idx + self.nb_stock + 1])
                    self.state[0] += real_prices[idx] * amount * (1 - self.transaction_fee)
                    self.state[idx + self.nb_stock + 1] -= amount
                    self.cost += real_prices[idx] * amount * self.transaction_fee
                    self.trades += 1
            # BUY
            for idx in buy_index:
                if actions[idx] > 0:
                    available_amount = self.state[0] // real_prices[idx]
                    amount = min(available_amount, actions[idx] * self.hmax)
                    self.state[0] -= real_prices[idx] * amount * (1 + self.transaction_fee)
                    self.state[idx + self.nb_stock + 1] += amount
                    self.cost += real_prices[idx] * amount * self.transaction_fee
                    self.trades += 1
        else:
            # High turbulence → sell all
            for idx in range(len(actions)):
                amount = self.state[idx + self.nb_stock + 1]
                self.state[0] += real_prices[idx] * amount * (1 - self.transaction_fee)
                self.state[idx + self.nb_stock + 1] = 0
                self.cost += real_prices[idx] * amount * self.transaction_fee
                self.trades += 1

    # ---------------------------
    # Step
    # ---------------------------
    def step(self, actions):
        real_prices = np.expm1(self.data.adjcp.values)
        begin_total_asset = self.state[0] + sum(
            real_prices * np.array(self.state[(self.nb_stock + 1):(self.nb_stock * 2 + 1)]))

        # Terminal check
        self.terminal = self.day >= len(self.dataframe.index.unique()) - 1
        if self.terminal:
            # Compute terminal Sharpe reward
            reward = (begin_total_asset - self.initial_balance)/self.initial_balance

            return self._get_observation(), reward, True, False, {}

        # Execute actions
        self._execute_action(actions)

        # Advance day safely
        self.day += 1
        self.data = self.dataframe.loc[self.day, :]
        self.turbulence = float(self.data['turbulence'].values[0])

        # Update state
        self.state = [self.state[0]] + \
                     self.data.adjcp.values.tolist() + \
                     list(self.state[(self.nb_stock + 1):(self.nb_stock * 2 + 1)]) + \
                     self.data.macd.values.tolist() + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist() + \
                     self.data.volatility.values.tolist() + \
                     self.data.ma10.values.tolist() + \
                     self.data.ma50.values.tolist()
        state_vec = self._get_state()
        self.state_history.append(state_vec)

        # Track asset & stepwise reward
        end_total_asset = self.state[0] + sum(
            real_prices * np.array(self.state[(self.nb_stock + 1):(self.nb_stock * 2 + 1)]))
        self.asset_memory.append(end_total_asset)
        step_return = (end_total_asset - begin_total_asset)

        # Dense step reward (normalized)
        reward = step_return * self.reward_scaling - self.action_penalty * np.sum(np.square(actions))
        self.rewards_memory.append(reward)

        return self._get_observation(), reward, self.terminal, False, {}

    # ---------------------------
    # Reset
    # ---------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.asset_memory = [self.initial_balance]
        self.return_history = []
        self.day = 0
        self.data = self.dataframe.loc[self.day, :]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.turbulence = 0

        self.state = [self.initial_balance] + \
                     self.data.adjcp.values.tolist() + \
                     [0] * self.nb_stock + \
                     self.data.macd.values.tolist() + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist() + \
                     self.data.volatility.values.tolist() + \
                     self.data.ma10.values.tolist() + \
                     self.data.ma50.values.tolist()
        state_vec = self._get_state()
        self.state_history = [state_vec]

        return self._get_observation(), {}

    # ---------------------------
    # Seed
    # ---------------------------
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]