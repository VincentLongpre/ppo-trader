import gymnasium as gym
import yaml
import json
import numpy as np
import pandas as pd
import os
from agent.PPO import PPO, FeedForwardNN
from utils.run_episode import run_trials
from stable_baselines3 import PPO as BPPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt

def plot_learning_curves(save_path):
    """
    Plot learning curves comparing our and stable baselines's training episodic returns.

    Parameters:
    - save_path (str): Path to the directory containing the saved results.
    """
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

    # Plot baseline PPO
    plt.plot(ppo_mean_reward, label="Baseline")
    plt.fill_between(range(len(ppo_mean_reward)),
                     ppo_mean_reward - ppo_std_reward,
                     ppo_mean_reward + ppo_std_reward,
                     alpha=0.5)

    # Plot our PPO
    plt.plot(mcppo_mean_reward, label="Ours")
    plt.fill_between(range(len(mcppo_mean_reward)),
                     mcppo_mean_reward - mcppo_std_reward,
                     mcppo_mean_reward + mcppo_std_reward,
                     alpha=0.5)

    plt.xlabel('Episode')
    plt.ylabel('Average Episodic Return')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    save_path = "runs1/pendulum/"
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    training = True

    if training:
        with open("configs/ppo_configs.yaml", 'r') as f:
            ppo_configs = yaml.safe_load(f)

        with open("configs/env_configs.yaml", 'r') as f:
            env_configs = yaml.safe_load(f)

        run_trials(PPO, FeedForwardNN, env, save_path, None, "our_ppo", **ppo_configs)

        for run in range(10):
            env = make_vec_env(env_name, monitor_dir=save_path)
            model = BPPO('MlpPolicy', env)
            model.learn(total_timesteps=1000000, progress_bar=True)
            os.rename(os.path.join(save_path, "0.monitor.csv"), os.path.join(save_path, f"ppo_{run}.csv"))

    plot_learning_curves(save_path)

    