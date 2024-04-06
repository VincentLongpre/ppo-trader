import gymnasium as gym
import yaml
import json
import numpy as np
import os
from agent.PPO import MCPPO, FeedForwardNN, run_trials
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

if __name__ == "__main__":
    save_path = "runs/pendulum/"
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)

    with open("configs/ppo_configs.yaml", 'r') as f:
        ppo_configs = yaml.safe_load(f)

    with open("configs/env_configs.yaml", 'r') as f:
        env_configs = yaml.safe_load(f)

    # run_trials(MCPPO, FeedForwardNN, env, save_path, **ppo_configs)

    env = make_vec_env(env_name, monitor_dir=save_path)

    for run in range(10):
        model = PPO('MlpPolicy', env)
        model.learn(total_timesteps=200000, progress_bar=True)
        os.rename(os.path.join(save_path, "0.monitor.csv"), os.path.join(save_path, f"ppo_{run}.csv"))

    