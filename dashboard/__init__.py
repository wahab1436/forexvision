"""
Dashboard Module

Contains Streamlit application logic and Plotly charting utilities
for real-time monitoring of the trading system.
"""

from .plotly_charts import create_candlestick_chart, create_equity_curve

__all__ = [
    "create_candlestick_chart",
    "create_equity_curve"
]
