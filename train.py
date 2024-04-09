import pandas as pd
import yaml
from utils.create_dataset import data_split
from agent.Original_PPO import PPO, FeedForwardNN
from utils.run_episode import hyperparams_run_gradient
from env.stockEnv import StockEnv
from stable_baselines3 import PPO as BPPO

if __name__ == "__main__":
    dataset = pd.read_csv("processed_dataset.csv")
    dataset = data_split(dataset, '2013-01-01', '2014-01-01')

    with open("configs/ppo_configs.yaml", 'r') as f:
        ppo_configs = yaml.safe_load(f)

    with open("configs/env_configs.yaml", 'r') as f:
        env_configs = yaml.safe_load(f)

    env = StockEnv(dataset, **env_configs)

    ppo_configs['learning_rates'] = [0.0003]

    hyperparams_run_gradient(PPO, FeedForwardNN, env, **ppo_configs)
    # model.learn(total_timesteps=50000, progress_bar=True)