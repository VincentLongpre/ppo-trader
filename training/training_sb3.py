import pandas as pd
from preprocessor.utils import data_split
import numpy as np
from env import stockEnv_sb3 as stockEnv
from stable_baselines3 import PPO
import gym
import torch
import yaml

np.random.seed(52)

dataset = pd.read_csv("./data/processed_dataset.csv")
dataset = data_split(dataset, '2013-01-01', '2014-01-01')

with open("./configs/params_PPO.yaml") as f:
    agent_cfg = yaml.safe_load(f)

with open("./configs/params_env.yaml") as f:
    env_cfg = yaml.safe_load(f)

# env_name = 'Pendulum-v1' # 'CartPole-v1' # 'MountainCar-v0'
# env = gym.make(env_name)
env = stockEnv.StockEnv(dataset, **env_cfg)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)