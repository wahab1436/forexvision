import os
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from loguru import logger
import yaml
import time

class DataManager:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.cache_dir = self.config['data']['cache_dir']
        self.av_key = self.config['data']['alpha_vantage_key']
        self.ts = TimeSeries(key=self.av_key, output_format='pandas')
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_yfinance(self, ticker, period='60d', interval='5m'):
        logger.info(f"Fetching {ticker} from Yahoo Finance")
        try:
            data = yf.download(ticker, period=period, interval=interval)
            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
            data.columns = data.columns.droplevel(1) if isinstance(data.columns, pd.MultiIndex) else data.columns
            return data
        except Exception as e:
            logger.error(f"Yahoo Finance fetch failed: {e}")
            return None

    def fetch_alpha_vantage(self, ticker):
        logger.info(f"Fetching {ticker} from Alpha Vantage")
        try:
            data, meta = self.ts.intraday(symbol=ticker, interval='5min', outputsize='compact')
            if data.empty:
                return None
            data.index = pd.to_datetime(data.index)
            return data
        except Exception as e:
            logger.error(f"Alpha Vantage fetch failed: {e}")
            return None

    def get_data(self, ticker, mode='historical'):
        cache_file = os.path.join(self.cache_dir, f"{ticker.replace('=X', '')}.parquet")
        
        if mode == 'historical':
            if os.path.exists(cache_file):
                logger.info(f"Loading cached data for {ticker}")
                return pd.read_parquet(cache_file)
            else:
                data = self.fetch_yfinance(ticker)
                if data is not None:
                    data.to_parquet(cache_file)
                return data
        
        elif mode == 'realtime':
            data = self.fetch_alpha_vantage(ticker)
            if data is None:
                logger.warning("Alpha Vantage failed, falling back to Yahoo Finance")
                time.sleep(12)
                data = self.fetch_yfinance(ticker, period='1d', interval='5m')
            return data

    def clean_data(self, df):
        if df is None:
            return None
        df = df.copy()
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        df = df.ffill()
        return df
