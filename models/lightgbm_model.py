import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, Union

logger = logging.getLogger(__name__)


class LightGBMModel:
    """
    LightGBM regression model for financial forecasting.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        if params:
            default_params.update(params)
        
        self.params = default_params
        self.model: Optional[lgb.LGBMRegressor] = None
    
    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray]
    ) -> None:
        # Convert to numpy if pandas
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.values.ravel()
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if isinstance(y_val, (pd.Series, pd.DataFrame)):
            y_val = y_val.values.ravel()
        
        # Handle NaN values
        mask_train = ~np.isnan(y_train)
        mask_val = ~np.isnan(y_val)
        
        X_train = X_train[mask_train]
        y_train = y_train[mask_train]
        X_val = X_val[mask_val]
        y_val = y_val[mask_val]
        
        logger.info(f"Training LightGBM with {X_train.shape[0]} samples")
        
        self.model = lgb.LGBMRegressor(**self.params)
        
        # Train with early stopping - MUST include eval_metric
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',  # Required for early stopping
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        logger.info(f"LightGBM training complete. Best iteration: {self.model.best_iteration_}")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.nan_to_num(X, nan=0.0)
        return self.model.predict(X)
    
    def save(self, filepath: str) -> None:
        if self.model is None:
            raise ValueError("No model to save")
        self.model.booster_.save_model(filepath)
        logger.info(f"LightGBM model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        self.model = lgb.LGBMRegressor()
        self.model.fit(
            np.array([[0]]), np.array([0]),
            init_model=filepath,
            keep_training_booster=True
        )
        logger.info(f"LightGBM model loaded from {filepath}")
