import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
from env.stockEnv import StockEnv

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

    if len(returns_list) == 0:
        print("No monitor logs found for plotting.")
        return

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
    with open("configs/params_env.yaml", 'r') as f:
        env_configs = yaml.safe_load(f)

    run_save_path = "runs/stockEnv/"
    model_save_path = "models/rppo/"

    os.makedirs(run_save_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    for run in range(1):
        # ------------------------
        # Initialize environment
        # ------------------------
        env = StockEnv(dataset, **env_configs)
        env = Monitor(env, run_save_path)

        # ------------------------
        # Initialize RecurrentPPO
        # ------------------------
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=env,
            learning_rate=1e-4,
            clip_range=0.2,
            policy_kwargs=dict(
                lstm_hidden_size=128,
                n_lstm_layers=2,
                activation_fn=torch.nn.Tanh,
            ),
            verbose=1
        )

        # ------------------------
        # Train
        # ------------------------
        model.learn(total_timesteps=5_000, progress_bar=True)

        # ------------------------
        # Save last LSTM state
        # ------------------------
        obs, _ = env.reset()
        lstm_states = None
        done = False
        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
            obs, _, done, _, _ = env.step(action)

        lstm_state_path = os.path.join(model_save_path, f"{run}_last_lstm_state.pt")
        torch.save(lstm_states, lstm_state_path)
        print(f"✅ Saved last LSTM state to {lstm_state_path}")

        # ------------------------
        # Save monitor logs
        # ------------------------
        monitor_path = os.path.join(run_save_path, f"rppo_{run}.csv")
        if os.path.exists(os.path.join(run_save_path, "monitor.csv")):
            os.rename(os.path.join(run_save_path, "monitor.csv"), monitor_path)
            print(f"✅ Saved monitor logs to {monitor_path}")

        # ------------------------
        # Save model
        # ------------------------
        model_path = os.path.join(model_save_path, f"{run}.zip")
        model.save(model_path)
        print(f"✅ Saved trained model to {model_path}")

    # ------------------------
    # Plot learning curves
    # ------------------------
    plot_learning_curves(run_save_path, label="RecurrentPPO")