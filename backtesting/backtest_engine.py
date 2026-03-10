import pandas as pd
import numpy as np
from loguru import logger

class BacktestEngine:
    def __init__(self, config):
        self.config = config
        self.risk_per_trade = config['risk']['risk_per_trade']
        self.sl_mult = config['risk']['atr_sl_multiplier']
        self.tp_mult = config['risk']['atr_tp_multiplier']
        self.buy_thresh = config['signals']['buy_threshold']
        self.sell_thresh = config['signals']['sell_threshold']
        
    def run(self, df, predictions):
        df = df.copy()
        df['pred_return'] = predictions
        df['signal'] = 0
        df.loc[df['pred_return'] > self.buy_thresh, 'signal'] = 1
        df.loc[df['pred_return'] < self.sell_thresh, 'signal'] = -1
        
        df['pnl'] = 0.0
        df['trade_id'] = 0
        trade_id = 0
        position = 0
        entry_price = 0
        sl = 0
        tp = 0
        atr = 0
        
        equity = 100000
        equity_curve = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            current_price = row['Close']
            current_atr = row['atr_14'] if 'atr_14' in row else 0.001
            
            if position == 0 and row['signal'] != 0:
                trade_id += 1
                position = row['signal']
                entry_price = current_price
                atr = current_atr
                sl = entry_price - (position * self.sl_mult * atr)
                tp = entry_price + (position * self.tp_mult * atr)
                df.at[df.index[i], 'trade_id'] = trade_id
                
            elif position != 0:
                df.at[df.index[i], 'trade_id'] = trade_id
                if (position == 1 and current_price <= sl) or (position == -1 and current_price >= sl):
                    pnl = (current_price - entry_price) * position
                    df.at[df.index[i], 'pnl'] = pnl
                    equity += pnl * 100000 # Approximate pip value scaling
                    position = 0
                elif (position == 1 and current_price >= tp) or (position == -1 and current_price <= tp):
                    pnl = (current_price - entry_price) * position
                    df.at[df.index[i], 'pnl'] = pnl
                    equity += pnl * 100000
                    position = 0
            
            equity_curve.append(equity)
            
        df['equity_curve'] = equity_curve
        return df
