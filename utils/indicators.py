import pandas as pd

def get_tp(df):
    return (df['close'] + df['high'] + df['low']).divide(3.0)

def smma(series, window):
    return series.ewm(alpha=1.0 / window, min_periods=0).mean()


# Moving Average Convergence/Divergence
def get_macd(df, long=26, short=12):
    close = df['close']
    ema_short = close.ewm(ignore_na=False, span=short, min_periods=1, adjust=True).mean()
    ema_long = close.ewm(ignore_na=False, span=long, min_periods=1, adjust=True).mean()
    return ema_short - ema_long

# Commodity Channel Index
def get_cci(df, n=14):
    tp = get_tp(df)
    sma = tp.rolling(n, 1).mean()
    mean_deviation = (tp - sma).abs().rolling(n, 1).mean()
    cci = (tp - sma) / (0.015 * mean_deviation)
    return cci

def get_adx(df: pd.DataFrame, n=14):

    high = df['high']
    low = df['low']
    close = df['close']

    # Calculate True Range

    # tr = pd.concat([(high - close.shift()).abs()], axis=1).max(axis=1)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = smma(tr, n)

    # Calculate +DM and -DM

    plus_dm = high.diff()
    plus_dm.loc[0]=0
    minus_dm = -low.diff()
    minus_dm.loc[0]=0
    plus_dm = ((plus_dm > 0) & (plus_dm > minus_dm)) * plus_dm
    minus_dm = ((minus_dm > 0) & (minus_dm > plus_dm)) * minus_dm
    plus_dm = smma(plus_dm, n)
    minus_dm = smma(minus_dm, n)

    # Calculate +DI and -DI

    plus_di = 100 * (plus_dm / atr)
    minus_di = 100 * (minus_dm / atr)

    # Calculate the ADX

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(span=6, min_periods=1).mean()

    return adx


def get_rsi(df, n=14):
    change = df['close'].diff()
    change.iloc[0] = 0
    up = (change + abs(change))/2
    down = (-change + abs(change))/2

    u_ema = smma(up, n)
    d_ema = smma(down, n)

    rs = u_ema / d_ema
    return 100 - 100 / (1.0 + rs)

