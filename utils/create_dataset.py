import os
import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib

# --------------------------
# 1. Get Dow Jones tickers
# --------------------------
def get_dow_jones_tickers():
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    try:
        tables = pd.read_html(url, header=0)
        table = tables[1]
        if "Symbol" in table.columns:
            col = "Symbol"
        elif "Ticker" in table.columns:
            col = "Ticker"
        else:
            raise ValueError("No ticker column found")
        tickers = table[col].astype(str).str.replace('.', '-', regex=False).tolist()
        return tickers
    except Exception as e:
        print(f"⚠️ Could not fetch from Wikipedia ({e}), using hardcoded list.")
        return [
            "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GS",
            "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
            "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT", "XOM"
        ]

# --------------------------
# 2. Download Yahoo Finance data
# --------------------------
def get_ticker_data(start_date, end_date, ticker_list):
    all_data = []
    for ticker in ticker_list:
        stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        if stock_data.empty:
            continue
        stock_data['ticker'] = ticker
        stock_data = stock_data.reset_index()[['Date','ticker','Adj Close','Open','High','Low','Volume']]
        stock_data.columns = ['date','ticker','adjcp','open','high','low','volume']
        stock_data['day'] = stock_data['date'].dt.dayofweek
        all_data.append(stock_data)
    df = pd.concat(all_data, ignore_index=True).sort_values(['date','ticker']).reset_index(drop=True)
    return df

# --------------------------
# 3. Add technical indicators
# --------------------------
def add_features(df):
    for ticker, group in df.groupby('ticker'):
        macd = MACD(group['adjcp']).macd()
        rsi = RSIIndicator(group['adjcp']).rsi()
        cci = CCIIndicator(group['high'], group['low'], group['adjcp']).cci()
        adx = ADXIndicator(group['high'], group['low'], group['adjcp']).adx()
        # Sector-based / rolling features (here we use simple rolling as placeholder)
        volatility = group['adjcp'].pct_change().rolling(20).std()
        ma10 = group['adjcp'].rolling(10).mean()
        ma50 = group['adjcp'].rolling(50).mean()
        df.loc[df['ticker']==ticker, ['macd','rsi','cci','adx','volatility','ma10','ma50']] = pd.DataFrame({
            'macd': macd,
            'rsi': rsi,
            'cci': cci,
            'adx': adx,
            'volatility': volatility,
            'ma10': ma10,
            'ma50': ma50
        }).values
    return df

# --------------------------
# 4. Calculate turbulence
# --------------------------
def calculate_turbulence(df, window=252):
    df_copy = df.copy()
    price_pivot = df.pivot(index='date', columns='ticker', values='adjcp')
    for i in range(window, len(price_pivot)):
        current = price_pivot.iloc[i]
        hist = price_pivot.iloc[i-window:i]
        cov = hist.cov()
        mean_hist = hist.mean()
        returns = current - mean_hist
        try:
            turb = np.dot(returns.values, np.linalg.inv(cov)).dot(returns.values.T)
            turb = max(turb, 0)
        except np.linalg.LinAlgError:
            turb = 0
        df_copy.loc[df_copy['date']==current.name, 'turbulence'] = turb
    df_copy['turbulence'] = df_copy['turbulence'].fillna(0)
    return df_copy

# --------------------------
# 5. Align and fill missing
# --------------------------
def align_fill(df):
    df['date'] = pd.to_datetime(df['date'])
    most_complete = df.groupby('ticker')['date'].count().idxmax()
    complete_dates = df[df['ticker']==most_complete]['date'].sort_values().unique()
    def align(group):
        group = group.set_index('date').reindex(complete_dates)
        group['ticker'] = group['ticker'].ffill().bfill()
        return group.reset_index().rename(columns={'index':'date'})
    df = df.groupby('ticker', group_keys=False).apply(align).reset_index(drop=True)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate().ffill().bfill()
    return df

# --------------------------
# 6. Apply Min-Max scaling
# --------------------------
def apply_scaler(train_df, test_df, indicator_cols):
    scaler = MinMaxScaler()
    train_df[indicator_cols] = scaler.fit_transform(train_df[indicator_cols])
    test_df[indicator_cols] = scaler.transform(test_df[indicator_cols])
    return train_df, test_df, scaler

# --------------------------
# 7. Train/test split
# --------------------------
def data_split(df, start, end):
    df = df[(df['date'] >= start) & (df['date'] < end)].sort_values(['date','ticker'], ignore_index=True)
    df.index = df['date'].factorize()[0]
    return df

# --------------------------
# 8. Main execution
# --------------------------
if __name__ == "__main__":
    start_date = '2009-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    train_start, train_end = '2021-01-01','2024-01-01'
    test_start, test_end = '2024-01-01','2025-07-01'

    # ----------------------
    # Load or create processed dataset
    # ----------------------
    if os.path.exists("processed_dataset.csv"):
        df = pd.read_csv("processed_dataset.csv", parse_dates=['date'])
        print("⚡ Loaded processed_dataset.csv")
    else:
        tickers = get_dow_jones_tickers()
        print(f"📈 Using {len(tickers)} Dow Jones tickers: {tickers}")
        df = get_ticker_data(start_date, end_date, tickers)
        df = add_features(df)
        df = calculate_turbulence(df)
        df = align_fill(df)
        df['adjcp'] = np.log1p(df['adjcp'])  # log-normalize prices
        df.to_csv("processed_dataset.csv", index=False)
        print("✅ Saved processed_dataset.csv")

    # ----------------------
    # Train/test split
    # ----------------------
    train_data = data_split(df, train_start, train_end)
    test_data = data_split(df, test_start, test_end)

    # ----------------------
    # Min-Max scaling for indicators + turbulence
    # ----------------------
    indicator_cols = ['macd','rsi','cci','adx','volatility','ma10','ma50','turbulence']
    train_data_scaled, test_data_scaled, scaler = apply_scaler(train_data, test_data, indicator_cols)
    joblib.dump(scaler, "scaler.pkl")
    print("💾 Saved MinMaxScaler to scaler.pkl")

    # ----------------------
    # Save train/test datasets
    # ----------------------
    train_data_scaled.to_csv('processed_dataset_train.csv', index=True, index_label='date_env')
    test_data_scaled.to_csv('processed_dataset_test.csv', index=True, index_label='date_env')
    print("✅ Saved train/test datasets (log prices + min-max scaled indicators)")