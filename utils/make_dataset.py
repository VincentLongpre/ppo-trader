import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.trend import CCIIndicator
from ta.trend import ADXIndicator

def get_ticker_data(start_date, end_date, ticker_list):
    all_data = []
    for ticker_symbol in ticker_list:
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

        stock_data['ticker'] = ticker_symbol
        stock_data = stock_data.reset_index()
        stock_data = stock_data[['Date', 'ticker', 'Adj Close', 'Open', 'High', 'Low', 'Volume']]
        stock_data.columns = ['date', 'ticker', 'adjcp', 'open', 'high', 'low', 'volume']
        all_data.append(stock_data)

    res_df = pd.concat(all_data, ignore_index=True).sort_values(['date', 'ticker']).reset_index(drop=True)
    return res_df

def add_technical_indicators(dataset):
    technical_indicators = {}
    
    for ticker_symbol, data in dataset.groupby('ticker'):
        macd = MACD(data['adjcp']).macd()
        
        rsi = RSIIndicator(data['adjcp']).rsi()
        
        cci = CCIIndicator(data['high'], data['low'], data['adjcp']).cci()
        
        adx = ADXIndicator(data['high'], data['low'], data['adjcp']).adx()
        
        technical_indicators[ticker_symbol] = pd.DataFrame({
            'macd': macd,
            'rsi': rsi,
            'cci': cci,
            'adx': adx
        }, index=data.index)
    
    for ticker_symbol, indicators_data in technical_indicators.items():
        dataset.loc[dataset['ticker'] == ticker_symbol, ['macd', 'rsi', 'cci', 'adx']] = indicators_data.values
    
    return dataset

def calculate_turbulence(df, window=252):
    df_copy = df.copy()
    df_price_pivot = df.pivot(index='date', columns='ticker', values='adjcp')
    
    for i in range(window, len(df_price_pivot)):
        current_price = df_price_pivot.iloc[i]
        hist_prices = df_price_pivot.iloc[i - window:i]
        
        cov_temp = hist_prices.cov()
        mean_returns = hist_prices.mean()
        
        current_returns = (current_price - mean_returns)
        
        temp = np.dot(current_returns.values, np.linalg.inv(cov_temp)).dot(current_returns.values.T)
        turbulence_temp = temp if temp > 0 else 0
        
        df_copy.loc[df['date'] == hist_prices.index[-1], 'turbulence'] = turbulence_temp
    
    return df_copy

if __name__ == "__main__":
    buffer_start_date = '2008-01-01'
    start_date = '2009-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    ticker_list = [
    'AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GS', 'HD', 
    'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 
    'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT', 'XOM'
    ]

    df = get_ticker_data(buffer_start_date, end_date, ticker_list)
    df = add_technical_indicators(df)
    df = calculate_turbulence(df)

    # df.to_csv('test.csv')