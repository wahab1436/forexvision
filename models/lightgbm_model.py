import lightgbm as lgb
from loguru import logger

class LightGBMModel:
    def __init__(self, params=None):
        self.params = params or {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 63,
            'min_child_samples': 20,
            'n_jobs': -1
        }
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        logger.info("LightGBM training complete")
        return self.model

    def predict(self, X):
        return self.model.predict(X)
