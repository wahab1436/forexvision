from sklearn.ensemble import RandomForestRegressor
from loguru import logger

class RandomForestModel:
    def __init__(self, params=None):
        self.params = params or {
            'n_estimators': 300,
            'max_depth': 10,
            'min_samples_leaf': 10,
            'n_jobs': -1
        }
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_train, y_train)
        logger.info("Random Forest training complete")
        return self.model

    def predict(self, X):
        return self.model.predict(X)
