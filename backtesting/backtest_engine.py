import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtesting engine for strategy evaluation."""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.trades = []
    
    def run(self, data, strategy):
        """Execute backtest on given data."""
        for index, row in data.iterrows():
            signal = strategy.generate_signal(row)
            if signal != 0:
                self._execute_trade(signal, row)
        return self._calculate_results()
    
    def _execute_trade(self, signal, row):
        """Execute a single trade."""
        trade = {
            'date': row.name,
            'signal': signal,
            'price': row['Close']
        }
        self.trades.append(trade)
    
    def _calculate_results(self):
        """Calculate performance metrics."""
        return {
            'total_trades': len(self.trades),
            'initial_capital': self.initial_capital
        }                df.at[df.index[i], 'trade_id'] = trade_id
                
            elif position != 0:
                df.at[df.index[i], 'trade_id'] = trade_id
                # Check SL/TP
                # Apply spread to exit price calculation for realism
                exit_price = current_price - (position * spread) 
                
                if (position == 1 and exit_price <= sl) or (position == -1 and exit_price >= sl):
                    pnl = (exit_price - entry_price) * position
                    df.at[df.index[i], 'pnl'] = pnl
                    equity += pnl * 100000
                    position = 0
                elif (position == 1 and exit_price >= tp) or (position == -1 and exit_price <= tp):
                    pnl = (exit_price - entry_price) * position
                    df.at[df.index[i], 'pnl'] = pnl
                    equity += pnl * 100000
                    position = 0
            
            equity_curve.append(equity)
            
        df['equity_curve'] = equity_curve
        return df
