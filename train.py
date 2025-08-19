import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env.stockEnv import StockEnv


# ------------------------
# Custom Tanh Head
# ------------------------
class TanhHead(nn.Module):
    def forward(self, x):
        return torch.tanh(x)


# ------------------------
# Learning Curve Plotting
# ------------------------
def plot_learning_curves(save_path, label="PPO"):
    returns_list = []
    for file in os.listdir(save_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(save_path, file), skiprows=1)
            if 'r' in df.columns:
                returns_list.append(df['r'].values.tolist())
    if not returns_list:
        print("No monitor logs found.")
        return
    returns_array = np.array(returns_list)
    mean_reward = np.mean(returns_array, axis=0)
    std_reward = np.std(returns_array, axis=0)
    plt.plot(mean_reward, label=label)
    plt.fill_between(range(len(mean_reward)),
                     mean_reward - std_reward,
                     mean_reward + std_reward, alpha=0.5)
    plt.xlabel("Episode")
    plt.ylabel("Average Episodic Return")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    dataset = pd.read_csv("processed_dataset_train.csv", index_col='date_env')
    with open("configs/params_env.yaml", 'r') as f:
        env_configs = yaml.safe_load(f)

    run_save_path = "runs/stockEnv/"
    model_save_path = "models/ppo/"
    os.makedirs(run_save_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    for run in range(1):
        base_env = StockEnv(dataset, **env_configs)
        env = Monitor(base_env, run_save_path)

        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-4,
            clip_range=0.5,
            batch_size=64,
            n_steps=512,
            verbose=1,
            ent_coef=0.01,
            vf_coef=0.65,
            policy_kwargs=dict(
            net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64])
        )
        )

        # add tanh head to actor network
        if hasattr(model.policy.mlp_extractor, "policy_net"):
            model.policy.mlp_extractor.policy_net.add_module("tanh_head", TanhHead())

        model.learn(total_timesteps=100_000, progress_bar=True)

        model_path = os.path.join(model_save_path, f"{run}.zip")
        model.save(model_path)
        print(f"✅ Saved trained model to {model_path}")

        monitor_file = os.path.join(run_save_path, "monitor.csv")
        if os.path.exists(monitor_file):
            monitor_path = os.path.join(run_save_path, f"ppo_{run}.csv")
            os.rename(monitor_file, monitor_path)
            print(f"✅ Saved monitor logs to {monitor_path}")

    plot_learning_curves(run_save_path, label="PPO")