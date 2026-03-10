import pandas as pd
import numpy as np

def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Log Returns
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Rolling Mean and Std of Returns
    for window in [5, 10, 20, 50]:
        df[f'ret_mean_{window}'] = df['log_return'].rolling(window=window).mean()
        df[f'ret_std_{window}'] = df['log_return'].rolling(window=window).std()
    
    # Realized Volatility
    df['realized_vol_20'] = df['log_return'].rolling(window=20).std() * np.sqrt(252 * 78) # Annualized approx
    
    # Z-Score
    df['z_score'] = (df['log_return'] - df['ret_mean_20']) / df['ret_std_20']
    
    # Autocorrelation
    for lag in [1, 2, 3, 5]:
        df[f'autocorr_{lag}'] = df['log_return'].autocorr(lag=lag)
        
    # Range Position
    df['range_min_20'] = df['Close'].rolling(window=20).min()
    df['range_max_20'] = df['Close'].rolling(window=20).max()
    df['range_pos'] = (df['Close'] - df['range_min_20']) / (df['range_max_20'] - df['range_min_20'])
    
    return df
