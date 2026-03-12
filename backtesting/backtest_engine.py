import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtesting engine for strategy evaluation.
    
    Provides functionality for executing trading strategies on historical data,
    tracking positions, and calculating performance metrics.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize the backtest engine.
        
        Args:
            initial_capital: Starting capital for backtesting
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.current_position: Optional[Dict[str, Any]] = None
    
    def run(self, data: pd.DataFrame, strategy) -> Dict[str, Any]:
        """
        Execute backtest on given data.
        
        Args:
            data: DataFrame with OHLCV data and signals
            strategy: Strategy object with generate_signal method
            
        Returns:
            Dictionary containing backtest results and metrics
        """
        logger.info(f"Starting backtest with {len(data)} bars")
        
        self.capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.current_position = None
        
        trade_id = 0
        
        for i in range(len(data)):
            row = data.iloc[i]
            signal = strategy.generate_signal(row)
            
            # Execute trades based on signal
            if signal != 0 and self.current_position is None:
                trade_id += 1
                self._open_position(signal, row, trade_id)
            elif signal == 0 and self.current_position is not None:
                self._close_position(row)
            
            # Update equity curve
            current_equity = self._calculate_current_equity(row)
            self.equity_curve.append(current_equity)
            
            # Record trade ID in dataframe
            df_at_index = data.index[i]
            data.at[df_at_index, 'trade_id'] = trade_id
        
        # Close any open positions at the end
        if self.current_position is not None and len(data) > 0:
            self._close_position(data.iloc[-1])
        
        results = self._calculate_results()
        logger.info(f"Backtest completed. Total trades: {len(self.trades)}")
        
        return results
    
    def _open_position(self, signal: int, row: pd.Series, trade_id: int) -> None:
        """
        Open a new trading position.
        
        Args:
            signal: Trading signal (1 for long, -1 for short)
            row: Current price data
            trade_id: Unique trade identifier
        """
        position = {
            'trade_id': trade_id,
            'entry_date': row.name,
            'entry_price': row['Close'],
            'direction': 'long' if signal > 0 else 'short',
            'size': self._calculate_position_size(row['Close']),
            'status': 'open'
        }
        
        self.current_position = position
        self.positions.append(position)
        
        logger.debug(f"Opened {position['direction']} position at {position['entry_price']}")
    
    def _close_position(self, row: pd.Series) -> None:
        """
        Close the current trading position.
        
        Args:
            row: Current price data
        """
        if self.current_position is None:
            return
        
        exit_price = row['Close']
        entry_price = self.current_position['entry_price']
        direction = self.current_position['direction']
        size = self.current_position['size']
        
        # Calculate P&L
        if direction == 'long':
            pnl = (exit_price - entry_price) * size
        else:
            pnl = (entry_price - exit_price) * size
        
        # Update capital
        self.capital += pnl
        
        # Record trade
        trade = {
            'trade_id': self.current_position['trade_id'],
            'entry_date': self.current_position['entry_date'],
            'exit_date': row.name,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'size': size,
            'pnl': pnl,
            'return_pct': pnl / (entry_price * size) * 100
        }
        
        self.trades.append(trade)
        self.current_position = None
        
        logger.debug(f"Closed position. P&L: {pnl:.2f}")
    
    def _calculate_position_size(self, price: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            price: Current asset price
            
        Returns:
            Position size in units
        """
        # Risk 2% of capital per trade
        risk_amount = self.capital * 0.02
        position_size = risk_amount / price
        return position_size
    
    def _calculate_current_equity(self, row: pd.Series) -> float:
        """
        Calculate current portfolio equity.
        
        Args:
            row: Current price data
            
        Returns:
            Current equity value
        """
        equity = self.capital
        
        if self.current_position is not None:
            current_price = row['Close']
            entry_price = self.current_position['entry_price']
            direction = self.current_position['direction']
            size = self.current_position['size']
            
            if direction == 'long':
                unrealized_pnl = (current_price - entry_price) * size
            else:
                unrealized_pnl = (entry_price - current_price) * size
            
            equity += unrealized_pnl
        
        return equity
    
    def _calculate_results(self) -> Dict[str, Any]:
        """
        Calculate comprehensive backtest results.
        
        Returns:
            Dictionary containing all performance metrics
        """
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'initial_capital': self.initial_capital,
                'final_capital': self.capital
            }
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Sharpe ratio (simplified)
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }
    
    def _calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown from equity curve.
        
        Returns:
            Maximum drawdown as percentage
        """
        if len(self.equity_curve) < 2:
            return 0.0
        
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self) -> float:
        """
        Calculate Sharpe ratio from trade returns.
        
        Returns:
            Annualized Sharpe ratio
        """
        if len(self.trades) < 2:
            return 0.0
        
        returns = [t['return_pct'] for t in self.trades]
        returns_series = pd.Series(returns)
        
        if returns_series.std() == 0:
            return 0.0
        
        # Annualize (assuming daily trades)
        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252)
        
        return sharpe
    
    def get_equity_curve(self) -> pd.Series:
        """
        Get equity curve as pandas Series.
        
        Returns:
            Equity curve series
        """
        return pd.Series(self.equity_curve)
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        Get all trades as DataFrame.
        
        Returns:
            DataFrame with all trade details
        """
        return pd.DataFrame(self.trades)
    
    def reset(self) -> None:
        """Reset the engine to initial state."""
        self.capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.current_position = None
        logger.info("Backtest engine reset")
