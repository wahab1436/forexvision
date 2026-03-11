"""
Backtesting Module

Contains engines for simulating trading strategies, calculating metrics,
and performing multi-pair analysis.
"""

from .backtest_engine import BacktestEngine
from .metrics import calculate_metrics
from .multi_pair_backtest import MultiPairBacktest

__all__ = [
    "BacktestEngine",
    "calculate_metrics",
    "MultiPairBacktest"
]
