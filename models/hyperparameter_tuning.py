import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from loguru import logger

class HyperparameterTuner:
    def __init__(self, model_type="xgboost"):
        self.model_type = model_type

    def objective_xgb(self, trial, X_train, y_train, X_val, y_val):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'objective': 'reg:squarederror',
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        mse = ((preds - y_val) ** 2).mean()
        return mse

    def objective_lgb(self, trial, X_train, y_train, X_val, y_val):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'n_jobs': -1
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        mse = ((preds - y_val) ** 2).mean()
        return mse

    def tune(self, X_train, y_train, X_val, y_val, n_trials=50):
        logger.info(f"Starting hyperparameter tuning for {self.model_type}")
        if self.model_type == "xgboost":
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: self.objective_xgb(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
        elif self.model_type == "lightgbm":
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: self.objective_lgb(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
        else:
            raise ValueError("Unsupported model type")

        logger.info(f"Best params: {study.best_params}")
        return study.best_params
