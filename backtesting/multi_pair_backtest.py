import pandas as pd
import os
from loguru import logger
from backtesting.backtest_engine import BacktestEngine
from backtesting.metrics import calculate_metrics
from utils.data_utils import DataManager
from features.technical_indicators import add_technical_indicators
from features.statistical_features import add_statistical_features
from features.session_features import add_session_features
import yaml

class MultiPairBacktest:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.dm = DataManager(config_path)
        self.results = []

    def prepare_data(self, df):
        df = self.dm.clean_data(df)
        df = add_technical_indicators(df)
        df = add_statistical_features(df)
        df = add_session_features(df)
        return df

    def run_all_pairs(self, models, predictions_dict):
        """
        models: dict of pair_name -> model_instance
        predictions_dict: dict of pair_name -> predictions_array
        """
        portfolio_pnl = 0
        pair_metrics = {}

        for pair in self.config['pairs']:
            logger.info(f"Backtesting {pair}")
            df = self.dm.get_data(pair, mode='historical')
            if df is None:
                continue
            
            df = self.prepare_data(df)
            
            # Align predictions with dataframe (assuming predictions match test set length)
            # For this utility, we assume predictions_dict[pair] aligns with the tail of df
            preds = predictions_dict.get(pair, None)
            if preds is None:
                logger.warning(f"No predictions found for {pair}")
                continue

            engine = BacktestEngine(self.config)
            # Slice df to match prediction length for simulation
            test_df = df.iloc[-len(preds):].copy()
            results_df = engine.run(test_df, preds)
            metrics = calculate_metrics(results_df)
            
            metrics['pair'] = pair
            pair_metrics[pair] = metrics
            self.results.append(metrics)
            portfolio_pnl += metrics.get('total_pnl', 0)

        # Save Summary
        summary_df = pd.DataFrame(self.results)
        os.makedirs("backtesting/results", exist_ok=True)
        summary_df.to_csv("backtesting/results/multi_pair_summary.csv", index=False)
        logger.info(f"Multi-pair backtest complete. Total PnL: {portfolio_pnl}")
        return summary_df
