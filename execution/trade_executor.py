import pandas as pd
from loguru import logger

class TradeExecutor:
    def __init__(self, config):
        self.config = config
        self.buy_thresh = config['signals']['buy_threshold']
        self.sell_thresh = config['signals']['sell_threshold']
        
    def generate_signal(self, predicted_return):
        if predicted_return > self.buy_thresh:
            return 'BUY'
        elif predicted_return < self.sell_thresh:
            return 'SELL'
        else:
            return 'HOLD'
            
    def log_trade(self, pair, signal, price, sl, tp, filename="data/trade_log.csv"):
        import os
        import csv
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'pair', 'signal', 'entry', 'sl', 'tp'])
            writer.writerow([pd.Timestamp.now(), pair, signal, price, sl, tp])
        logger.info(f"Trade logged: {pair} {signal} at {price}")
