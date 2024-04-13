import pandas as pd
import yaml
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from utils.create_dataset import data_split
from agent.PPO import PPO, FeedForwardNN
from env.stockEnv import StockEnv
from utils.run_episode import episode
from stable_baselines3 import PPO as BPPO

if __name__ == "__main__":
    dataset = pd.read_csv("processed_dataset.csv")
    dataset = data_split(dataset, '2014-01-01', '2015-01-01')

    with open("configs/ppo_configs.yaml", 'r') as f:
        ppo_configs = yaml.safe_load(f)

    with open("configs/env_configs.yaml", 'r') as f:
        env_configs = yaml.safe_load(f)

    model_path = "models/"

    env = StockEnv(dataset, **env_configs)

    rewards = []
    for file in os.listdir(model_path + "our_ppo"):
        with open(file, 'r') as f:
            agent = pkl.load(f)
        
        rewards.append(episode(agent, 1, testing=False))

