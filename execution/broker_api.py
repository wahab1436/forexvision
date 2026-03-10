from utils.data_utils import DataManager
from loguru import logger

class BrokerAPI:
    def __init__(self, config):
        self.dm = DataManager()
        self.config = config
        
    def get_latest_candle(self, ticker):
        data = self.dm.get_data(ticker, mode='realtime')
        if data is not None:
            return data.iloc[-1]
        return None
