import shap
import pandas as pd
from loguru import logger

class ShapAnalysis:
    def __init__(self, model, X_background):
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        self.background = X_background

    def get_summary_values(self, X_data):
        try:
            shap_values = self.explainer.shap_values(X_data)
            return shap_values
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            return None

    def plot_summary(self, X_data, save_path="utils/plots/shap_summary.png"):
        import matplotlib.pyplot as plt
        shap_values = self.get_summary_values(X_data)
        if shap_values is not None:
            plt.figure()
            shap.summary_plot(shap_values, X_data, show=False)
            plt.savefig(save_path)
            plt.close()
            logger.info(f"SHAP summary saved to {save_path}")
