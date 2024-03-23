import pandas as pd
import yfinance as yf
from datetime import datetime

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

if __name__ == "__main__":
    start_date = '2009-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    ticker_list = [
    'AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GS', 'HD', 
    'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 
    'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT', 'XOM'
    ]

    df = get_ticker_data(start_date, end_date, ticker_list)
    print(df)