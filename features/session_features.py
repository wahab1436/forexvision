import pandas as pd

def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
    # Simple session flags (UTC based approximations)
    df['session_tokyo'] = ((df['hour'] >= 0) & (df['hour'] < 9)).astype(int)
    df['session_london'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
    df['session_ny'] = ((df['hour'] >= 12) & (df['hour'] < 21)).astype(int)
    
    return df
