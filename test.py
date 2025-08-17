import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator
import yfinance as yf
from env.stockEnv import StockEnv
from sb3_contrib import RecurrentPPO
import empyrical
import yaml
import torch

def evaluate_episode(model, env, lstm_states=None, max_iter=10000):
    """Evaluate one episode of the agent in raw-dollar terms using a given LSTM state."""
    asset_history = [env.asset_memory[0]]  # initial cash
    obs, _ = env.reset()
    done = False

    for _ in range(max_iter):
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        asset_history.append(env.asset_memory[-1])
        if done:
            break
    return asset_history

def plot_portfolio(dates, mean_assets, asset_std, benchmark):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, mean_assets, label='RecurrentPPO', color='red', linewidth=2)
    ax.fill_between(dates,
                    mean_assets - 2 * asset_std,
                    mean_assets + 2 * asset_std,
                    color='salmon', alpha=0.3)
    ax.plot(dates, benchmark, label='VOO', color='green', linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    ax.xaxis.set_major_locator(MonthLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def print_statistics(stats_dict):
    df = pd.DataFrame({
        "Method": ["RecurrentPPO"],
        "Mean Sharpe Ratio": [np.mean(stats_dict['sharpe'])],
        "Mean Annual Return": [np.mean(stats_dict['annual_return'])],
        "Mean Max Drawdown": [np.mean(stats_dict['max_dd'])],
        "Mean Annual Volatility": [np.mean(stats_dict['annual_vol'])],
        "Mean Cumulative Returns": [np.mean(stats_dict['cum_returns'])]
    })
    print("Mean Portfolio Statistics:")
    print(df)

if __name__ == "__main__":
    # -------------------------
    # 1. Load test dataset
    # -------------------------
    dataset = pd.read_csv("processed_dataset_test.csv", index_col='date_env')
    dates = pd.to_datetime(dataset['date'].drop_duplicates().values)

    # -------------------------
    # 2. Load env config
    # -------------------------
    with open("configs/params_env.yaml", 'r') as f:
        env_configs = yaml.safe_load(f)

    # -------------------------
    # 3. Initialize environment
    # -------------------------
    env = StockEnv(dataset, **env_configs)

    stats = {
        'balances': [],
        'cum_returns': [],
        'sharpe': [],
        'annual_return': [],
        'annual_vol': [],
        'max_dd': []
    }

    # -------------------------
    # 4. Load trained RecurrentPPO models
    # -------------------------
    model_dir = "models/rppo/"
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.zip')]

    for f in model_files:
        model_path = os.path.join(model_dir, f)
        model = RecurrentPPO.load(model_path)

        # Load last LSTM state corresponding to this model
        lstm_state_path = os.path.join(model_dir, f"last_lstm_state_{f.split('.')[0]}.pt")
        if os.path.exists(lstm_state_path):
            # Allow numpy unpickling
            with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
                lstm_states = torch.load(lstm_state_path, weights_only=False)
            print(f"✅ Loaded last LSTM state from {lstm_state_path}")
        else:
            lstm_states = None
            print(f"⚠️ No saved LSTM state found for {model_path}, starting from zero.")

        cur_assets = evaluate_episode(model, env, lstm_states=lstm_states)
        stats['balances'].append(cur_assets)

        daily_returns = np.diff(cur_assets) / cur_assets[:-1]
        stats['cum_returns'].append(empyrical.cum_returns_final(daily_returns))
        stats['sharpe'].append(empyrical.sharpe_ratio(daily_returns))
        stats['annual_return'].append(empyrical.annual_return(daily_returns))
        stats['max_dd'].append(empyrical.max_drawdown(daily_returns))
        stats['annual_vol'].append(empyrical.annual_volatility(daily_returns))

    # -------------------------
    # 5. Compute mean portfolio stats
    # -------------------------
    n_samples = len(stats['balances'])
    mean_assets = np.mean(stats['balances'], axis=0)
    asset_std = np.std(stats['balances'], axis=0) / np.sqrt(n_samples)

    # -------------------------
    # 6. VOO benchmark
    # -------------------------
    voo_data = yf.download("VOO", start=dates.min(), end=dates.max(), auto_adjust=True)
    voo_data = voo_data.reindex(dates)
    first_close = voo_data['Close'].iloc[0]
    benchmark = 1e6 * voo_data['Close'] / first_close
    benchmark = benchmark.values

    # -------------------------
    # 7. Plot and print statistics
    # -------------------------
    plot_portfolio(dates, mean_assets[1:], asset_std[1:], benchmark)
    print_statistics(stats)