import xgboost as xgb
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class XGBoostModel:
    """XGBoost regression model for financial forecasting."""
    
    def __init__(self, params=None):
        """Initialize XGBoost model."""
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
        
        # Initialize model
        self.model = xgb.XGBRegressor(**self.params)
        
        # Create DMatrix for callback-based early stopping (modern XGBoost)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Convert sklearn params to native XGBoost params
        native_params = {k: v for k, v in self.params.items() 
                        if k not in ['n_estimators', 'n_jobs', 'random_state', 'verbosity']}
        native_params['objective'] = 'reg:squarederror'
        
        # Train with callback-based early stopping
        self.model = xgb.train(
            params=native_params,
            dtrain=dtrain,
            num_boost_round=self.params['n_estimators'],
            evals=[(dval, 'validation')],
            callbacks=[
                xgb.callback.EarlyStopping(
                    rounds=50,
                    metric_name='rmse',
                    data_name='validation',
                    save_best=True,
                    verbose=False
                )
            ]
        )
        
        logger.info(f"Training complete. Best iteration: {self.model.best_iteration}")
    
    def predict(self, X):
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.nan_to_num(X, nan=0.0)
        
        # Convert to DMatrix for prediction
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def save(self, filepath):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk."""
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
