import numpy as np
import pandas as pd

def calculate_metrics(df):
    trades = df[df['trade_id'] != 0].groupby('trade_id')['pnl'].sum()
    if len(trades) == 0:
        return {}
        
    win_rate = (trades > 0).sum() / len(trades)
    total_pnl = trades.sum()
    profit_factor = trades[trades > 0].sum() / abs(trades[trades < 0].sum()) if trades[trades < 0].sum() != 0 else 0
    
    equity = df['equity_curve']
    returns = equity.pct_change().dropna()
    sharpe = np.sqrt(252 * 78) * returns.mean() / returns.std() if returns.std() != 0 else 0
    
    max_drawdown = (equity.cummax() - equity).max()
    
    return {
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'num_trades': len(trades)
    }
