import pandas as pd
import yfinance as yf

class YahooDownloader:

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list


    def fetch_data(self, proxy=None) -> pd.DataFrame:

        df = pd.DataFrame()
        num_failures = 0

        for tic in self.ticker_list:
            temp = yf.download(
                tic, start=self.start_date, end=self.end_date, proxy=None
            )

            temp["tic"] = tic
            if len(temp) > 0:
                # data_df = data_df.append(temp_df)
                df = pd.concat([df, temp], axis=0)
            else:
                num_failures += 1

        if num_failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")
            # reset the index, we want to use numbers as index instead of dates

        df = df.reset_index()

        df.columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjcp",
            "volume",
            "tic",
        ]

        df["close"] = df["adjcp"]
        # drop the adjusted close price column
        df = df.drop(labels="adjcp", axis=1)

        df = df.sort_values(['date', 'tic'], ignore_index=True)
        df.index = df.date.factorize()[0]

        df["day"] = df["date"].dt.dayofweek
        df["date"] = df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        df = df.dropna()
        # df = df.reset_index(drop=True)
        print("Shape of DataFrame: ", df.shape)
        # print("Display DataFrame: ", data.head())

        df = df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return df
    def select_equal_rows_stock(self, df):
        # Select equal row counts
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]

        return df

