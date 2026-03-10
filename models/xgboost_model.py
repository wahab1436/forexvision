import xgboost as xgb
from sklearn.model_selection import ParameterGrid
from loguru import logger

class XGBoostModel:
    def __init__(self, params=None):
        self.params = params or {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'n_jobs': -1
        }
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        logger.info("XGBoost training complete")
        return self.model

    def predict(self, X):
        return self.model.predict(X)
