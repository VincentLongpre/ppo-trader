import pandas as pd
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from utils.create_dataset import data_split
from stable_baselines3.common.monitor import Monitor
from env.stockEnv import StockEnv
import torch
import joblib

# Import RecurrentPPO from sb3_contrib
from sb3_contrib import RecurrentPPO

def plot_learning_curves(save_path, label="RecurrentPPO"):
    """
    Plot learning curves for PPO with LSTM.
    """
    returns_list = []

    for file in os.listdir(save_path):
        file_path = os.path.join(save_path, file)
        if file.endswith('.csv'):
            df = pd.read_csv(file_path, skiprows=1)
            returns_list.append(df['r'].values.tolist())

    returns_array = np.array(returns_list)
    mean_reward = np.mean(returns_array, axis=0)
    std_reward = np.std(returns_array, axis=0)

    plt.plot(mean_reward, label=label)
    plt.fill_between(
        range(len(mean_reward)),
        mean_reward - std_reward,
        mean_reward + std_reward,
        alpha=0.5
    )
    plt.xlabel('Episode')
    plt.ylabel('Average Episodic Return')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # ------------------------
    # Load dataset
    # ------------------------
    dataset = pd.read_csv("processed_dataset_train.csv", index_col='date_env')

    # ------------------------
    # Load env configs
    # ------------------------
    with open("configs/env_configs.yaml", 'r') as f:
        env_configs = yaml.safe_load(f)

    # ------------------------
    # Load scaler
    # ------------------------
    scaler = joblib.load("scaler.pkl")
    env_configs["scaler"] = scaler

    run_save_path = "runs/stockEnv/"
    model_save_path = "models/"

    for run in range(1):
        # ------------------------
        # Initialize env
        # ------------------------
        env = StockEnv(dataset, **env_configs)
        env = Monitor(env, run_save_path)

        # ------------------------
        # Initialize RecurrentPPO
        # ------------------------
        model = RecurrentPPO(
            policy="MlpLstmPolicy",  # pass as string
            env=env,
            learning_rate=3e-4,
            clip_range=0.2,
            policy_kwargs=dict(
                lstm_hidden_size=32,
                n_lstm_layers=2,
                activation_fn=torch.nn.LeakyReLU,
            ),
            verbose=1
        )

        # ------------------------
        # Train
        # ------------------------
        model.learn(total_timesteps=150_000, progress_bar=True)

        # ------------------------
        # Save monitor logs and model
        # ------------------------
        os.rename(
            os.path.join(run_save_path, "monitor.csv"),
            os.path.join(run_save_path, f"rppo_{run}.csv")
        )
        os.makedirs(model_save_path + "rppo/", exist_ok=True)
        model.save(model_save_path + f"rppo/{run}.zip")

    # ------------------------
    # Plot learning curves
    # ------------------------
    plot_learning_curves(run_save_path, label="RecurrentPPO")