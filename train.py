import pandas as pd
import yaml
from utils.create_dataset import data_split
from agent.PPO import MCPPO, FeedForwardNN, hyperparams_run_gradient
from env.stockEnv import StockEnv
from stable_baselines3 import PPO

if __name__ == "__main__":
    dataset = pd.read_csv("processed_dataset.csv")
    dataset = data_split(dataset, '2013-01-01', '2014-01-01')

    with open("configs/ppo_configs.yaml", 'r') as f:
        ppo_configs = yaml.safe_load(f)

    with open("configs/env_configs.yaml", 'r') as f:
        env_configs = yaml.safe_load(f)

    env = StockEnv(dataset, **env_configs)

    hyperparams_run_gradient(MCPPO, FeedForwardNN, env, **ppo_configs)