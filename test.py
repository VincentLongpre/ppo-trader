import pandas as pd
import yaml
import os
import yfinance as yf
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

def sb3_evaluate_episode(model, env, max_iter=10000):
    asset_history = [env.asset_memory[0]]

    obs, _ = env.reset()
    termination = False

    for _ in range(max_iter):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)

        if done:
            break

        asset_history.append(env.asset_memory[-1])

    return asset_history

def plot_portfolio_stats(dates, mean_asset_values_ours, mean_asset_values_baseline, variance_ours, variance_baseline, dija_price_rel):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(dates, mean_asset_values_ours, label='Ours', color='blue', linewidth=2)
    ax.fill_between(dates, mean_asset_values_ours - 2 * variance_ours, mean_asset_values_ours + 2 * variance_ours, color='lightblue', alpha=0.3)

    ax.plot(dates, mean_asset_values_baseline, label='Baseline', color='red', linewidth=2)
    ax.fill_between(dates, mean_asset_values_baseline - 2 * variance_baseline, mean_asset_values_baseline + 2 * variance_baseline, color='salmon', alpha=0.3)

    ax.plot(dates, djia_price_rel, label='DJIA', color='green', linewidth=2)

    ax.grid(True, linestyle='--', alpha=0.5)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Asset Value', fontsize=12)
    ax.legend(fontsize=10)

    ax.xaxis.set_major_locator(MonthLocator())

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

def print_mean_statistics(sb3_lists_dict, our_lists_dict):
    mean_sharpe_ratio_sb3 = np.mean(sb3_lists_dict['sharpe_ratio'])
    mean_ann_return_sb3 = np.mean(sb3_lists_dict['ann_return'])
    mean_max_dd_sb3 = np.mean(sb3_lists_dict['max_dd'])
    mean_ann_vol_sb3 = np.mean(sb3_lists_dict['ann_vol'])
    mean_cum_returns_sb3 = np.mean(sb3_lists_dict['cum_returns'])

    mean_sharpe_ratio_ours = np.mean(our_lists_dict['sharpe_ratio'])
    mean_ann_return_ours = np.mean(our_lists_dict['ann_return'])
    mean_max_dd_ours = np.mean(our_lists_dict['max_dd'])
    mean_ann_vol_ours = np.mean(our_lists_dict['ann_vol'])
    mean_cum_returns_ours = np.mean(our_lists_dict['cum_returns'])

    data = {
        'Method': ['SB3', 'Ours'],
        'Mean Sharpe Ratio': [mean_sharpe_ratio_sb3, mean_sharpe_ratio_ours],
        'Mean Annual Return': [mean_ann_return_sb3, mean_ann_return_ours],
        'Mean Max Drawdown': [mean_max_dd_sb3, mean_max_dd_ours],
        'Mean Annual Volatility': [mean_ann_vol_sb3, mean_ann_vol_ours],
        'Mean Cumulative Returns': [mean_cum_returns_sb3, mean_cum_returns_ours]
    }
    df = pd.DataFrame(data)

    print("Mean Statistics:")
    print(df)

if __name__ == "__main__":
    dataset = pd.read_csv("processed_dataset.csv")
    dataset = data_split(dataset, '2016-01-01', '2020-01-01')
    dates = dataset.date.drop_duplicates().values

    with open("configs/ppo_configs.yaml", 'r') as f:
        ppo_configs = yaml.safe_load(f)

    with open("configs/env_configs.yaml", 'r') as f:
        env_configs = yaml.safe_load(f)
        # env_configs['env_type'] = 'test'

    model_path = "models/"

    env = StockEnv(dataset, **env_configs)

    # Evaluate our PPO
    our_lists_dict = {
        'balances': [],
        'cum_returns': [],
        'sharpe_ratio': [],
        'ann_return': [],
        'ann_vol': [],
        'max_dd': []
        }
    
    for file in [f for f in os.listdir(model_path + "our_ppo") if not f.startswith('.')]:
        with open(model_path + f"our_ppo/{file}", 'rb') as f:
            agent = pkl.load(f)
        
        agent.env = env

        for _ in range(10):
            cur_balcances = episode(agent, 1, testing=True)
            our_lists_dict['balances'].append(cur_balcances)

            daily_returns = np.diff(cur_balcances) / cur_balcances[:-1]

            cumulative_returns = empyrical.cum_returns_final(daily_returns)
            our_lists_dict['cum_returns'].append(cumulative_returns)

            sharpe_ratio = empyrical.sharpe_ratio(daily_returns)
            our_lists_dict['sharpe_ratio'].append(sharpe_ratio)

            annualized_return = empyrical.annual_return(daily_returns)
            our_lists_dict['ann_return'].append(annualized_return)

            max_drawdown = empyrical.max_drawdown(daily_returns)
            our_lists_dict['max_dd'].append(max_drawdown)

            annualized_volatility = empyrical.annual_volatility(daily_returns)
            our_lists_dict['ann_vol'].append(annualized_volatility)

    n_samples = len(our_lists_dict['balances'])
    our_mean_asset = np.mean(our_lists_dict['balances'], axis=0)
    our_asset_volatility = np.std(our_lists_dict['balances'], axis=0) / np.sqrt(n_samples)

    # Evaluate stable baselines's PPO
    sb3_lists_dict = {
        'balances': [],
        'cum_returns': [],
        'sharpe_ratio': [],
        'ann_return': [],
        'ann_vol': [],
        'max_dd': []
        }
    
    for file in [f for f in os.listdir(model_path + "sb3_ppo") if not f.startswith('.')]:
        file = os.path.splitext(file)[0]
        agent = BPPO.load(model_path + f"sb3_ppo/{file}")

        for _ in range(10):
            cur_balcances = sb3_evaluate_episode(agent, env)
            sb3_lists_dict['balances'].append(cur_balcances)

            daily_returns = np.diff(cur_balcances) / cur_balcances[:-1]

            cumulative_returns = empyrical.cum_returns_final(daily_returns)
            sb3_lists_dict['cum_returns'].append(cumulative_returns)

            sharpe_ratio = empyrical.sharpe_ratio(daily_returns)
            sb3_lists_dict['sharpe_ratio'].append(sharpe_ratio)

            annualized_return = empyrical.annual_return(daily_returns)
            sb3_lists_dict['ann_return'].append(annualized_return)

            max_drawdown = empyrical.max_drawdown(daily_returns)
            sb3_lists_dict['max_dd'].append(max_drawdown)

            annualized_volatility = empyrical.annual_volatility(daily_returns)
            sb3_lists_dict['ann_vol'].append(annualized_volatility)

    n_samples = len(sb3_lists_dict['balances'])
    sb3_mean_asset = np.mean(sb3_lists_dict['balances'], axis=0)
    sb3_asset_volatility = np.std(sb3_lists_dict['balances'], axis=0) / np.sqrt(n_samples)

    djia_data = yf.download("^DJI", start="2016-01-01", end="2020-01-01")

    first_close_price = djia_data['Close'].iloc[0]
    djia_price_rel = 1e6 *  djia_data['Close'] / first_close_price

    plot_portfolio_stats(dates, our_mean_asset, sb3_mean_asset, our_asset_volatility, sb3_asset_volatility, djia_price_rel)

    print_mean_statistics(sb3_lists_dict, our_lists_dict)

        



