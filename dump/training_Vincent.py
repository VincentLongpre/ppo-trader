import pandas as pd
import numpy as np
import stockEnv
import PPO_GAE as PPO
import torch
import yaml
import os

def data_split(dataset, start, end):
    dataset = dataset[(dataset.date >= start) & (dataset.date < end)]
    dataset = dataset.sort_values(['date','ticker'], ignore_index=True)
    dataset.index = dataset.date.factorize()[0]
    return dataset

np.random.seed(52)

print(os.listdir())
dataset = pd.read_csv("processed_dataset.csv")
dataset = data_split(dataset, '2013-01-01', '2014-01-01')

with open("configs/env_configs.yaml") as f:
    env_cfg = yaml.safe_load(f)

with open("configs/ppo_configs.yaml") as f:
    agent_cfg = yaml.safe_load(f)

agent_cfg['learning_rates'] = [0.0003]

env = stockEnv.StockEnv(dataset, **env_cfg)

agent_class = PPO.MCPPO
# policy_class = PPO.FeedForwardNN
policy_class_actor = PPO.FeedForwardNN_Actor
policy_class_critic = PPO.FeedForwardNN_Critic

torch.autograd.set_detect_anomaly(True)

reward_arr_train = PPO.hyperparams_run_gradient(agent_class, policy_class_actor, policy_class_critic, env, **agent_cfg)