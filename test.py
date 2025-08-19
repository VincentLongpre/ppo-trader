import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator
import yfinance as yf
from env.stockEnv import StockEnv
from stable_baselines3 import PPO
import empyrical
import yaml


def evaluate_episode(model, env, max_iter=10000):
    """Evaluate one episode of the agent in raw-dollar terms using real prices."""
    asset_history = [env.asset_memory[0]]  # initial cash
    obs, _ = env.reset()
    done = False

    for _ in range(max_iter):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        # compute total asset using un-logged/un-scaled prices
        real_prices = np.expm1(env.data.adjcp.values)
        total_asset = env.state[0] + sum(real_prices * np.array(env.state[(env.nb_stock + 1):(env.nb_stock * 2 + 1)]))
        asset_history.append(total_asset)
        if done:
            break
    return asset_history


def plot_portfolio(dates, mean_assets, asset_std, benchmark_voo, benchmark_dji):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, mean_assets, label='PPO', color='blue', linewidth=2)
    ax.fill_between(dates,
                    mean_assets - 2 * asset_std,
                    mean_assets + 2 * asset_std,
                    color='skyblue', alpha=0.3)
    ax.plot(dates, benchmark_voo, label='VOO', color='green', linewidth=2)
    ax.plot(dates, benchmark_dji, label='Dow Jones', color='red', linewidth=2)
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
        "Method": ["PPO"],
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
    # 4. Load trained PPO models
    # -------------------------
    model_dir = "models/ppo/"
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.zip')]

    for f in model_files:
        model_path = os.path.join(model_dir, f)
        model = PPO.load(model_path)

        cur_assets = evaluate_episode(model, env)
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
    first_voo_close = voo_data['Close'].iloc[0]
    benchmark_voo = 1e6 * voo_data['Close'] / first_voo_close
    benchmark_voo = benchmark_voo.values

    # -------------------------
    # 7. DJI benchmark
    # -------------------------
    dji_data = yf.download("^DJI", start=dates.min(), end=dates.max(), auto_adjust=True)
    dji_data = dji_data.reindex(dates)
    first_dji_close = dji_data['Close'].iloc[0]
    benchmark_dji = 1e6 * dji_data['Close'] / first_dji_close
    benchmark_dji = benchmark_dji.values

    # -------------------------
    # 8. Plot and print statistics
    # -------------------------
    plot_portfolio(dates, mean_assets[1:], asset_std[1:], benchmark_voo, benchmark_dji)
    print_statistics(stats)