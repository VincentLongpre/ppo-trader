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
# 1. Get S&P 500 tickers
# --------------------------
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, header=0)[0]
    return table['Symbol'].tolist()


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
def add_technical_indicators(df):
    for ticker, group in df.groupby('ticker'):
        macd = MACD(group['adjcp']).macd()
        rsi = RSIIndicator(group['adjcp']).rsi()
        cci = CCIIndicator(group['high'], group['low'], group['adjcp']).cci()
        adx = ADXIndicator(group['high'], group['low'], group['adjcp']).adx()
        df.loc[df['ticker']==ticker, ['macd','rsi','cci','adx']] = pd.DataFrame({
            'macd': macd, 'rsi': rsi, 'cci': cci, 'adx': adx
        }).values
    return df


# --------------------------
# 4. Turbulence calculation
# --------------------------
def calculate_turbulence(df, window=252):
    df_copy = df.copy()
    price_pivot = df.pivot(index='date', columns='ticker', values='adjcp')
    for i in range(window, len(price_pivot)):
        current = price_pivot.iloc[i]
        hist = price_pivot.iloc[i-window:i]
        cov = hist.cov()
        mean_hist = hist.mean()
        returns = (current - mean_hist)
        try:
            turb = np.dot(returns.values, np.linalg.inv(cov)).dot(returns.values.T)
            turb = max(turb, 0)
        except np.linalg.LinAlgError:
            turb = 0
        df_copy.loc[df_copy['date']==current.name,'turbulence'] = turb
    df_copy['turbulence'] = df_copy['turbulence'].fillna(0)
    return df_copy


# --------------------------
# 5. Create dataset
# --------------------------
def create_dataset(start_date, end_date, tickers):
    buffer_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
    df = get_ticker_data(buffer_start, end_date, tickers)
    df = add_technical_indicators(df)

    # Log-transform price columns
    for col in ['adjcp','open','high','low']:
        df[col] = np.log(df[col].replace(0, np.nan)).replace(-np.inf, np.nan)

    df = calculate_turbulence(df)

    # Align tickers to complete calendar
    df['date'] = pd.to_datetime(df['date'])
    most_complete = df.groupby('ticker')['date'].count().idxmax()
    complete_dates = df[df['ticker']==most_complete]['date'].sort_values().unique()

    def align(group):
        group = group.set_index('date').reindex(complete_dates)
        group['ticker'] = group['ticker'].ffill().bfill()
        return group.reset_index().rename(columns={'index':'date'})

    df = df.groupby('ticker', group_keys=False).apply(align).reset_index(drop=True)

    # Fill missing numeric values
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate().ffill().bfill()

    # Remove buffer period
    df = df[df['date'] > start_date]
    return df


# --------------------------
# 6. Train/test split
# --------------------------
def data_split(df, start, end):
    df = df[(df['date'] >= start) & (df['date'] < end)].sort_values(['date','ticker'], ignore_index=True)
    df.index = df['date'].factorize()[0]
    return df


# --------------------------
# 7. Main execution
# --------------------------
if __name__ == "__main__":
    start_date = '2009-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    train_start, train_end = '2021-01-01', '2024-01-01'
    test_start, test_end = '2024-01-01', '2025-07-01'

    # Load or create dataset
    if os.path.exists("processed_dataset.csv"):
        dataset = pd.read_csv("processed_dataset.csv", parse_dates=['date'])
        print("⚡ Loaded processed_dataset.csv")
    else:
        tickers = get_sp500_tickers()
        dataset = create_dataset(start_date, end_date, tickers)
        dataset.to_csv("processed_dataset.csv", index=False)
        print("✅ Saved processed_dataset.csv")

    # Train/test split
    train_data = data_split(dataset, train_start, train_end)
    test_data = data_split(dataset, test_start, test_end)

    # MinMaxScaler: fit on train, transform both
    scaler = MinMaxScaler(feature_range=(-1,1))
    feature_cols = train_data.select_dtypes(include=[np.number]).columns.drop(['day'])
    train_data[feature_cols] = scaler.fit_transform(train_data[feature_cols])
    test_data[feature_cols] = scaler.transform(test_data[feature_cols])

    # Save processed data
    train_data.to_csv('processed_dataset_train.csv', index=True, index_label='date_env')
    test_data.to_csv('processed_dataset_test.csv', index=True, index_label='date_env')
    joblib.dump(scaler, "scaler.pkl")
    print("✅ Saved train/test datasets and scaler")
