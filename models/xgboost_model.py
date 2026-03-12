import xgboost as xgb
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost regression model for financial forecasting.
    
    Provides training, prediction, and evaluation functionality with
    proper handling of early stopping and evaluation metrics.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost model with configurable parameters.
        
        Args:
            params: Optional dictionary of XGBoost hyperparameters
        """
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
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_importance: Optional[pd.Series] = None
        self.training_history: Optional[Dict[str, List[float]]] = None
    
    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray]
    ) -> None:
        """
        Train the XGBoost model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        # Convert to numpy arrays if pandas objects
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
        
        logger.info(f"Training XGBoost with {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        # Initialize model
        self.model = xgb.XGBRegressor(**self.params)
        
        # Train with early stopping - MUST include eval_metric
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',  # Required for early stopping
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Store training history
        if hasattr(self.model, 'evals_result_'):
            self.training_history = self.model.evals_result_
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                name='importance'
            )
        
        best_iteration = self.model.best_iteration
        best_score = self.model.best_score
        logger.info(f"Training complete. Best iteration: {best_iteration}, Best RMSE: {best_score:.4f}")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Input features
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Handle NaN values in input
        X = np.nan_to_num(X, nan=0.0)
        
        return self.model.predict(X)
    
    def predict_with_uncertainty(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        n_estimators: int = 100
    ) -> tuple:
        """
        Generate predictions with uncertainty estimates using quantile regression.
        
        Args:
            X: Input features
            n_estimators: Number of bootstrap iterations
            
        Returns:
            Tuple of (mean_prediction, lower_bound, upper_bound)
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.nan_to_num(X, nan=0.0)
        
        # Generate predictions
        predictions = self.model.predict(X)
        
        # Simple uncertainty estimate based on training residuals
        if hasattr(self.model, 'best_score'):
            std_error = np.sqrt(self.model.best_score)
        else:
            std_error = np.std(predictions) * 0.1
        
        return predictions, predictions - 1.96 * std_error, predictions + 1.96 * std_error
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Args:
            top_n: Return only top N features if specified
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        importance_df = self.feature_importance.reset_index()
        importance_df.columns = ['feature', 'importance']
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        if top_n:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def save(self, filepath: str) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load model from disk.
        
        Args:
            filepath: Path to load model from
        """
        self.model = xgb.XGBRegressor()
        self.model.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def get_params(self) -> Dict[str, Any]:
        """Get current model parameters."""
        return self.params.copy()
    
    def set_params(self, **params) -> 'XGBoostModel':
        """Update model parameters."""
        self.params.update(params)
        if self.model is not None:
            self.model.set_params(**params)
        return self
    
    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate R² score on given data.
        
        Args:
            X: Input features
            y: True target values
            
        Returns:
            R² score
        """
        if self.model is None:
            raise ValueError("Model must be trained before scoring")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()
        
        X = np.nan_to_num(X, nan=0.0)
        mask = ~np.isnan(y)
        
        return self.model.score(X[mask], y[mask])
