import pandas as pd
from preprocessor.utils import data_split
import numpy as np
from env import stockEnv
from agent import PPO_GAE as PPO
import torch
import yaml

np.random.seed(52)

dataset = pd.read_csv("./data/processed_dataset.csv")
dataset = data_split(dataset, '2013-01-01', '2014-01-01')

with open("./configs/params_env.yaml") as f:
    env_cfg = yaml.safe_load(f)

with open("./configs/params_PPO.yaml") as f:
    agent_cfg = yaml.safe_load(f)

env = stockEnv.StockEnv(dataset, **env_cfg)

agent_class = PPO.MCPPO ## PPO.MCPPO
# policy_class = PPO.FeedForwardNN
policy_class_actor = PPO.FeedForwardNN_Actor
policy_class_critic = PPO.FeedForwardNN_Critic

torch.autograd.set_detect_anomaly(True)

reward_arr_train = PPO.hyperparams_run_gradient(agent_class, policy_class_actor, policy_class_critic, env, **agent_cfg)