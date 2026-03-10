import pandas as pd
import os
import csv
from loguru import logger
from datetime import datetime, date

class TradeExecutor:
    def __init__(self, config):
        self.config = config
        self.buy_thresh = config['signals']['buy_threshold']
        self.sell_thresh = config['signals']['sell_threshold']
        self.max_positions = config['risk']['max_open_positions']
        self.daily_loss_limit = config['risk']['daily_loss_limit']
        self.current_positions = 0
        self.daily_pnl = 0.0
        self.last_trade_date = date.today()
        self.log_file = "data/trade_log.csv"
        os.makedirs("data", exist_ok=True)

    def check_daily_reset(self):
        today = date.today()
        if today > self.last_trade_date:
            self.daily_pnl = 0.0
            self.last_trade_date = today
            self.current_positions = 0

    def generate_signal(self, predicted_return):
        self.check_daily_reset()
        
        if self.daily_pnl <= self.daily_loss_limit:
            logger.warning("Daily loss limit reached. Trading halted.")
            return 'HALT'
            
        if self.current_positions >= self.max_positions:
            logger.warning("Max open positions reached.")
            return 'HOLD'

        if predicted_return > self.buy_thresh:
            return 'BUY'
        elif predicted_return < self.sell_thresh:
            return 'SELL'
        else:
            return 'HOLD'
            
    def execute_trade(self, pair, signal, price, sl, tp, pnl=0.0):
        if signal in ['BUY', 'SELL']:
            self.current_positions += 1
        # Logic to decrement positions would require trade closure tracking 
        # simplified here for signal generation phase
        
        self.daily_pnl += pnl
        self.log_trade(pair, signal, price, sl, tp)

    def log_trade(self, pair, signal, price, sl, tp):
        file_exists = os.path.isfile(self.log_file)
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'pair', 'signal', 'entry', 'sl', 'tp'])
            writer.writerow([pd.Timestamp.now(), pair, signal, price, sl, tp])
        logger.info(f"Trade logged: {pair} {signal} at {price}")
