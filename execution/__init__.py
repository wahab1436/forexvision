"""
Execution Module

Contains logic for trade execution, signal generation, and broker API
interactions for paper trading and live deployment.
"""

from .trade_executor import TradeExecutor
from .broker_api import BrokerAPI

__all__ = [
    "TradeExecutor",
    "BrokerAPI"
]
