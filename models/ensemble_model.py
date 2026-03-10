import numpy as np
from loguru import logger

class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1/3, 1/3, 1/3]

    def predict(self, X):
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        return weighted_pred
