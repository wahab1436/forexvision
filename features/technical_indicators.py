import pandas as pd
import ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # SMA
    for window in [5, 10, 20, 50]:
        df[f'sma_{window}'] = ta.trend.sma_indicator(df['Close'], window=window)
    
    # EMA
    for span in [5, 12, 26]:
        df[f'ema_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
    
    # MACD
    macd = ta.trend.MACD(df['Close'], window_fast=12, window_slow=26, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    # RSI
    for period in [7, 14, 21]:
        df[f'rsi_{period}'] = ta.momentum.rsi(df['Close'], window=period)
    
    # ATR
    df['atr_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_pct'] = (df['Close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Williams %R
    df['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
    
    return df
