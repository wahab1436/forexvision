import xgboost as xgb
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, Union

logger = logging.getLogger(__name__)


class XGBoostModel:
    """XGBoost regression model for financial forecasting."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize XGBoost model with configurable parameters."""
        default_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        if params:
            default_params.update(params)
        
        self.params = default_params
        self.model = None
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the XGBoost model with early stopping."""
        # Convert pandas to numpy
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.values.ravel()
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if isinstance(y_val, (pd.Series, pd.DataFrame)):
            y_val = y_val.values.ravel()
        
        # Remove NaN values
        mask_train = ~np.isnan(y_train)
        mask_val = ~np.isnan(y_val)
        
        X_train = X_train[mask_train]
        y_train = y_train[mask_train]
        X_val = X_val[mask_val]
        y_val = y_val[mask_val]
        
        logger.info(f"Training XGBoost with {len(X_train)} samples")
        
        # Initialize and train model
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbose=False
        )
        
        logger.info(f"Training complete. Best iteration: {self.model.best_iteration}")
    
    def predict(self, X):
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.nan_to_num(X, nan=0.0)
        return self.model.predict(X)
    
    def save(self, filepath):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk."""
        self.model = xgb.XGBRegressor()
        self.model.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
