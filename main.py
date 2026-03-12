import os
import sys
import argparse
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from utils.data_utils import get_data, prepare_features
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.ensemble_model import EnsembleModel
from models.hyperparameter_tuning import HyperparameterTuner
from backtesting.backtest_engine import BacktestEngine
from backtesting.metrics import calculate_metrics
from backtesting.multi_pair_backtest import MultiPairBacktest
from execution.trade_executor import TradeExecutor
from dashboard.app import run_dashboard
from config.config_loader import ConfigLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/forexvision.log')
    ]
)
logger = logging.getLogger(__name__)


class ForexVisionSystem:
    """
    Main system class for ForexVision trading platform.
    
    Handles both backend processing (backtesting, training) and 
    frontend dashboard launching.
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize the ForexVision system."""
        self.config = ConfigLoader(config_path).load()
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.results = {}
        
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models/saved', exist_ok=True)
        
        self.logger.info("ForexVision System initialized")
    
    def run_backtest(self, pairs: List[str]) -> Dict[str, Any]:
        """
        Run backtesting on specified currency pairs.
        
        Args:
            pairs: List of currency pairs to backtest
            
        Returns:
            Dictionary containing backtest results for each pair
        """
        self.logger.info("Starting Backtesting Mode")
        results = {}
        
        for pair in pairs:
            try:
                self.logger.info(f"Processing {pair} for Backtesting")
                
                # Load data
                df = get_data(pair, self.config['data']['start_date'])
                if df is None or df.empty:
                    self.logger.warning(f"No data for {pair}, skipping")
                    continue
                
                # Prepare features
                df = prepare_features(df, self.config['features'])
                df = df.dropna()
                
                if len(df) < 100:
                    self.logger.warning(f"Insufficient data for {pair}, skipping")
                    continue
                
                # Prepare train/val/test splits
                split = int(len(df) * 0.7)
                val_split = int(len(df) * 0.85)
                
                feature_cols = [c for c in df.columns if c not in ['Close', 'Date', 'trade_id']]
                X = df[feature_cols].values
                y = df['Close'].shift(-1).values  # Predict next day close
                
                # Remove NaN from target
                mask = ~np.isnan(y)
                X = X[mask]
                y = y[mask]
                
                # Adjust splits after masking
                split = int(len(X) * 0.7)
                val_split = int(len(X) * 0.85)
                
                X_train, y_train = X[:split], y[:split]
                X_val, y_val = X[split:val_split], y[split:val_split]
                X_test, y_test = X[val_split:], y[val_split:]
                
                self.logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
                
                # Train models
                ensemble = self.train_models(X_train, y_train, X_val, y_val)
                
                # Generate predictions
                predictions = ensemble.predict(X_test)
                
                # Run backtest
                test_df = df.iloc[val_split:].copy()
                test_df['prediction'] = predictions
                test_df['signal'] = np.where(predictions > test_df['Close'], 1, 
                                           np.where(predictions < test_df['Close'], -1, 0))
                
                backtest = BacktestEngine(initial_capital=self.config['backtest']['initial_capital'])
                backtest_results = backtest.run(test_df, self.config['backtest'])
                
                # Calculate metrics
                metrics = calculate_metrics(backtest_results, test_df)
                
                results[pair] = {
                    'metrics': metrics,
                    'predictions': predictions.tolist(),
                    'equity_curve': backtest_results.get('equity_curve', []),
                    'trades': backtest_results.get('trades', [])
                }
                
                self.logger.info(f"Completed backtest for {pair}. Return: {metrics.get('total_return', 0):.2f}%")
                
            except Exception as e:
                self.logger.error(f"Error processing {pair}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                results[pair] = {'error': str(e)}
        
        self.results = results
        return results
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """
        Train ensemble of models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Trained EnsembleModel
        """
        models = []
        
        # Train XGBoost
        try:
            self.logger.info("Training XGBoost Model")
            xgb = XGBoostModel()
            xgb.train(X_train, y_train, X_val, y_val)
            models.append(('xgboost', xgb))
        except Exception as e:
            self.logger.warning(f"XGBoost training failed: {e}")
        
        # Train LightGBM
        try:
            self.logger.info("Training LightGBM Model")
            lgb = LightGBMModel()
            lgb.train(X_train, y_train, X_val, y_val)
            models.append(('lightgbm', lgb))
        except Exception as e:
            self.logger.warning(f"LightGBM training failed: {e}")
        
        if not models:
            raise ValueError("No models could be trained")
        
        # Create ensemble
        ensemble = EnsembleModel(models)
        self.logger.info(f"Ensemble created with {len(models)} models")
        
        return ensemble
    
    def run_paper_trading(self, pairs: List[str]) -> None:
        """
        Run paper trading mode.
        
        Args:
            pairs: List of currency pairs to trade
        """
        self.logger.info("Starting Paper Trading Mode")
        
        executor = TradeExecutor(mode='paper', config=self.config)
        
        for pair in pairs:
            try:
                df = get_data(pair, self.config['data']['start_date'])
                if df is None or df.empty:
                    continue
                
                df = prepare_features(df, self.config['features'])
                
                # Generate signals using trained models
                if self.models:
                    features = df[self.config['features']['columns']].values
                    signals = self.models[0].predict(features)
                    df['signal'] = np.where(signals > df['Close'], 1,
                                          np.where(signals < df['Close'], -1, 0))
                    
                    # Execute trades
                    for idx, row in df.iterrows():
                        if row['signal'] != 0:
                            executor.execute_trade(pair, row['signal'], row['Close'], idx)
                
                self.logger.info(f"Paper trading completed for {pair}")
                
            except Exception as e:
                self.logger.error(f"Error in paper trading for {pair}: {e}")
    
    def launch_dashboard(self, results: Optional[Dict] = None) -> None:
        """
        Launch the Streamlit dashboard.
        
        Args:
            results: Optional backtest results to display
        """
        self.logger.info("Launching Dashboard")
        run_dashboard(results=results, config=self.config)
    
    def run(self, mode: str, pairs: List[str], launch_dash: bool = False) -> None:
        """
        Main entry point for running the system.
        
        Args:
            mode: Operating mode ('backtest', 'paper_trade', 'train', 'dashboard')
            pairs: List of currency pairs
            launch_dash: Whether to launch dashboard after processing
        """
        self.logger.info(f"Running in {mode} mode with pairs: {pairs}")
        
        results = None
        
        if mode == 'backtest':
            results = self.run_backtest(pairs)
        elif mode == 'paper_trade':
            self.run_paper_trading(pairs)
        elif mode == 'train':
            self.logger.info("Training mode - models will be saved")
            # Training logic here
        elif mode == 'dashboard':
            self.launch_dashboard()
            return
        else:
            self.logger.error(f"Unknown mode: {mode}")
            return
        
        if launch_dash or mode == 'dashboard':
            self.launch_dashboard(results)


def main():
    """Main entry point for CLI and Streamlit."""
    parser = argparse.ArgumentParser(description='ForexVision Trading System')
    parser.add_argument('--mode', type=str, default='backtest',
                       choices=['backtest', 'paper_trade', 'train', 'dashboard'],
                       help='Operating mode')
    parser.add_argument('--pairs', type=str, nargs='+', default=None,
                       help='Currency pairs to process')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dashboard', action='store_true',
                       help='Launch dashboard after processing')
    
    args = parser.parse_args()
    
    # Initialize system
    system = ForexVisionSystem(config_path=args.config)
    
    # Get pairs from config if not specified
    pairs = args.pairs
    if not pairs and 'pairs' in system.config:
        pairs = system.config['pairs']
    
    if not pairs:
        pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
    
    # Run system
    system.run(mode=args.mode, pairs=pairs, launch_dash=args.dashboard)


if __name__ == "__main__":
    main()
