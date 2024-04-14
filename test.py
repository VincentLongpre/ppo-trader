import pandas as pd
import yaml
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator
import gymnasium as gym
from utils.create_dataset import data_split
from agent.PPO import PPO, FeedForwardNN
from env.stockEnv import StockEnv
from utils.run_episode import episode
from stable_baselines3 import PPO as BPPO
from datetime import datetime
import empyrical


def plot_portfolio_stats(dates, mean_asset_values, volatility):
    upper_volatility_bounds = mean_asset_values + 2 * volatility
    lower_volatility_bounds = mean_asset_values - 2 * volatility

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(dates, mean_asset_values, label='Ours', color='blue', linewidth=2)
    ax.fill_between(dates, lower_volatility_bounds, upper_volatility_bounds, color='lightblue', alpha=0.3)

    ax.grid(True, linestyle='--', alpha=0.5)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Asset Value', fontsize=12)
    ax.set_title('Mean Asset Value and Volatility Bounds', fontsize=14)
    ax.legend(fontsize=10)

    ax.xaxis.set_major_locator(MonthLocator())

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset = pd.read_csv("processed_dataset.csv")
    dataset = data_split(dataset, '2016-01-01', '2020-01-01')
    dates = dataset.date.drop_duplicates().values

    with open("configs/ppo_configs.yaml", 'r') as f:
        ppo_configs = yaml.safe_load(f)

    with open("configs/env_configs.yaml", 'r') as f:
        env_configs = yaml.safe_load(f)

    model_path = "models/"

    env = StockEnv(dataset, **env_configs)

    our_lists_dict = {
        'balances': [],
        'sharpe_ratio': [],
        'ann_return': [],
        'ann_vol': [],
        'max_dd': []
        }
    
    for file in os.listdir(model_path + "our_ppo"):
        with open(model_path + f"our_ppo/{file}", 'rb') as f:
            agent = pkl.load(f)
        
        agent.env = env

        for _ in range(10):
            cur_balcances = episode(agent, 1, testing=True)
            our_lists_dict['balances'].append(cur_balcances)

            daily_returns = np.diff(cur_balcances) / cur_balcances[:-1]

            sharpe_ratio = empyrical.sharpe_ratio(daily_returns)
            our_lists_dict['sharpe_ratio'].append(sharpe_ratio)

            annualized_return = empyrical.annual_return(daily_returns)
            our_lists_dict['ann_return'].append(annualized_return)

            max_drawdown = empyrical.max_drawdown(daily_returns)
            our_lists_dict['max_dd'].append(max_drawdown)

            annualized_volatility = empyrical.annual_volatility(daily_returns)
            our_lists_dict['ann_vol'].append(annualized_volatility)

    our_mean_asset = np.mean(our_lists_dict['balances'], axis=0)
    our_asset_volatility = np.std(our_lists_dict['balances'], axis=0)

    plot_portfolio_stats(dates, our_mean_asset, our_asset_volatility)

        


