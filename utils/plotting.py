import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class PlottingUtils:
    def __init__(self, save_dir="utils/plots"):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        sns.set(style="whitegrid")

    def plot_feature_importance(self, feature_names, importances, title="Feature Importance"):
        plt.figure(figsize=(10, 8))
        indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
        top_10_indices = indices[:10]
        
        plt.barh(range(len(top_10_indices)), [importances[i] for i in top_10_indices])
        plt.yticks(range(len(top_10_indices)), [feature_names[i] for i in top_10_indices])
        plt.xlabel("Importance")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "feature_importance.png"))
        plt.close()

    def plot_equity_curve(self, equity_data, title="Equity Curve"):
        plt.figure(figsize=(12, 6))
        plt.plot(equity_data, label="Equity")
        plt.title(title)
        plt.xlabel("Trade Index")
        plt.ylabel("Balance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "equity_curve.png"))
        plt.close()

    def plot_correlation_heatmap(self, df, title="Asset Correlation"):
        plt.figure(figsize=(10, 8))
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "correlation_heatmap.png"))
        plt.close()
