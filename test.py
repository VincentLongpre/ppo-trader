import pandas as pd
import yaml
import os
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator
from env.stockEnv import StockEnv
from stable_baselines3 import PPO as BPPO
import empyrical
import joblib  # to load the saved MinMaxScaler


def sb3_evaluate_episode(model, env, max_iter=10000):
    """Evaluate one episode of the agent in real-dollar terms."""
    asset_history = [env.asset_memory[0]]  # initial cash
    obs, _ = env.reset()
    for _ in range(max_iter):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)

        # Append current portfolio value
        asset_history.append(env.asset_memory[-1])
        if done:
            break
    return asset_history


def plot_portfolio_stats(dates, mean_asset_values, variance, benchmark_values):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates, mean_asset_values, label='BPPO', color='red', linewidth=2)
    ax.fill_between(dates,
                    mean_asset_values - 2 * variance,
                    mean_asset_values + 2 * variance,
                    color='salmon', alpha=0.3)
    ax.plot(dates, benchmark_values, label='VOO', color='green', linewidth=2)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value', fontsize=12)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(MonthLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def print_mean_statistics(sb3_lists_dict):
    mean_sharpe_ratio = np.mean(sb3_lists_dict['sharpe_ratio'])
    mean_ann_return = np.mean(sb3_lists_dict['ann_return'])
    mean_max_dd = np.mean(sb3_lists_dict['max_dd'])
    mean_ann_vol = np.mean(sb3_lists_dict['ann_vol'])
    mean_cum_returns = np.mean(sb3_lists_dict['cum_returns'])
    df = pd.DataFrame({
        'Method': ['BPPO'],
        'Mean Sharpe Ratio': [mean_sharpe_ratio],
        'Mean Annual Return': [mean_ann_return],
        'Mean Max Drawdown': [mean_max_dd],
        'Mean Annual Volatility': [mean_ann_vol],
        'Mean Cumulative Returns': [mean_cum_returns]
    })
    print("Mean Statistics:")
    print(df)


if __name__ == "__main__":
    # ------------------------------
    # 1. Load test dataset
    # ------------------------------
    dataset = pd.read_csv("processed_dataset_test.csv", index_col='date_env')
    dataset.index = pd.to_datetime(dataset.index)

    # ------------------------------
    # 2. Load environment config and scaler
    # ------------------------------
    with open("configs/env_configs.yaml", 'r') as f:
        env_configs = yaml.safe_load(f)

    scaler = joblib.load("scaler.pkl")
    env_configs["scaler"] = scaler

    # ------------------------------
    # 3. Initialize environment
    # ------------------------------
    env = StockEnv(dataset, **env_configs)

    sb3_lists_dict = {
        'balances': [],
        'cum_returns': [],
        'sharpe_ratio': [],
        'ann_return': [],
        'ann_vol': [],
        'max_dd': []
    }

    dates = pd.to_datetime(dataset['date'].drop_duplicates().values)

    # ------------------------------
    # 4. Load agents and evaluate
    # ------------------------------
    model_path = "models/sb3_ppo"
    for file in [f for f in os.listdir(model_path) if f.endswith('.zip')]:
        agent = BPPO.load(os.path.join(model_path, file))
        cur_balances = sb3_evaluate_episode(agent, env)
        sb3_lists_dict['balances'].append(cur_balances)

        daily_returns = np.diff(cur_balances) / cur_balances[:-1]
        sb3_lists_dict['cum_returns'].append(empyrical.cum_returns_final(daily_returns))
        sb3_lists_dict['sharpe_ratio'].append(empyrical.sharpe_ratio(daily_returns))
        sb3_lists_dict['ann_return'].append(empyrical.annual_return(daily_returns))
        sb3_lists_dict['max_dd'].append(empyrical.max_drawdown(daily_returns))
        sb3_lists_dict['ann_vol'].append(empyrical.annual_volatility(daily_returns))

    # ------------------------------
    # 5. Compute mean portfolio stats
    # ------------------------------
    n_samples = len(sb3_lists_dict['balances'])
    sb3_mean_asset = np.mean(sb3_lists_dict['balances'], axis=0)
    sb3_asset_volatility = np.std(sb3_lists_dict['balances'], axis=0) / np.sqrt(n_samples)

    # ------------------------------
    # 6. VOO benchmark
    # ------------------------------
    voo_data = yf.download("VOO", start=dates.min(), end=dates.max(), auto_adjust=True)
    voo_data = voo_data.reindex(dates)
    first_close_price = voo_data['Close'].iloc[0]
    voo_price_rel = 1e6 * voo_data['Close'] / first_close_price
    voo_price_rel = voo_price_rel.values

    # ------------------------------
    # 7. Plot and print statistics
    # ------------------------------
    plot_portfolio_stats(dates, sb3_mean_asset[1:], sb3_asset_volatility[1:], voo_price_rel)
    print_mean_statistics(sb3_lists_dict)