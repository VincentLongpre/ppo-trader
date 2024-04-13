import pandas as pd
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from utils.create_dataset import data_split
from stable_baselines3.common.monitor import Monitor
from agent.Original_PPO import PPO, FeedForwardNN
from utils.run_episode import hyperparams_run_gradient, run_trials
from env.stockEnv import StockEnv
from stable_baselines3 import PPO as BPPO

def plot_learning_curves(save_path):
    ppo_returns = []
    mcppo_returns = []
    
    for file in os.listdir(save_path):
        file_path = os.path.join(save_path, file)

        if file.endswith('.csv'):
            df = pd.read_csv(file_path, skiprows=1)
            ppo_returns.append(df['r'].values.tolist())

        elif file.endswith('.json'):
            df = pd.read_json(file_path)
            mcppo_returns.append(df.values.tolist())

    mcppo_returns = np.array(mcppo_returns).squeeze()
    ppo_returns = np.array(ppo_returns)

    # Calculate mean and standard deviation
    ppo_mean_reward = np.mean(ppo_returns, axis=0)
    ppo_std_reward = np.std(ppo_returns, axis=0)
    mcppo_mean_reward = np.mean(mcppo_returns, axis=0)
    mcppo_std_reward = np.std(mcppo_returns, axis=0)

    # Plot PPO
    plt.plot(ppo_mean_reward, label="Baseline")
    plt.fill_between(range(len(ppo_mean_reward)),
                     ppo_mean_reward - ppo_std_reward,
                     ppo_mean_reward + ppo_std_reward,
                     alpha=0.5)

    # Plot MC-PPO
    plt.plot(mcppo_mean_reward, label="Ours")
    plt.fill_between(range(len(mcppo_mean_reward)),
                     mcppo_mean_reward - mcppo_std_reward,
                     mcppo_mean_reward + mcppo_std_reward,
                     alpha=0.5)

    plt.xlabel('Episode')
    plt.ylabel('Average Episodic Return')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    dataset = pd.read_csv("processed_dataset.csv")
    dataset = data_split(dataset, '2013-01-01', '2014-01-01')

    with open("configs/ppo_configs.yaml", 'r') as f:
        ppo_configs = yaml.safe_load(f)

    with open("configs/env_configs.yaml", 'r') as f:
        env_configs = yaml.safe_load(f)

    save_path = "runs/stockEnv/"

    env = StockEnv(dataset, **env_configs)

    run_trials(PPO, FeedForwardNN, env, save_path, "ppo_v", **ppo_configs)

    # for run in range(10):
    #     env = StockEnv(dataset, **env_configs)
    #     env = Monitor(env, save_path)
    #     model = BPPO('MlpPolicy', env)
    #     model.learn(total_timesteps=150000, progress_bar=True)
    #     os.rename(os.path.join(save_path, "monitor.csv"), os.path.join(save_path, f"ppo_{run}.csv"))

    plot_learning_curves(save_path)