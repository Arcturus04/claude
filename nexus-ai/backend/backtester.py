"""
Vectorized Backtesting Engine
From Hilpisch (2021) Part IV - Chapters 10-12

Implements vectorized backtesting for evaluating trading strategies,
event-based risk management (stop-loss / take-profit), and
performance analytics.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorizedBacktester:
    """
    Vectorized backtesting engine that evaluates trading strategies
    using matrix operations for speed (Chapter 10).

    Key concepts from the book:
    - Vectorized strategy evaluation (no loops)
    - Transaction costs
    - Performance metrics: Sharpe, Sortino, max drawdown
    - Comparison against buy-and-hold benchmark
    """

    def __init__(self, initial_capital: float = 10000.0,
                 transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.tc = transaction_cost  # 0.1% per trade (10 bps)

    def run(self, prices: np.ndarray, signals: np.ndarray,
            dates: Optional[List[str]] = None) -> Dict:
        """
        Run vectorized backtest.

        Args:
            prices: Array of closing prices
            signals: Array of position signals (1=long, 0=flat, -1=short)
            dates: Optional date labels

        Returns:
            Dict with performance metrics and equity curve
        """
        if len(prices) != len(signals):
            raise ValueError("Prices and signals must be same length")

        n = len(prices)
        if dates is None:
            dates = [f"t{i}" for i in range(n)]

        # Log returns (Chapter 4 - foundation of financial analysis)
        log_returns = np.diff(np.log(prices))
        log_returns = np.insert(log_returns, 0, 0)

        # Strategy returns = signal * market return
        # Shift signals by 1 to avoid look-ahead bias
        shifted_signals = np.roll(signals, 1)
        shifted_signals[0] = 0

        strategy_returns = shifted_signals * log_returns

        # Transaction costs on signal changes (position flips)
        trades = np.diff(shifted_signals)
        trades = np.insert(trades, 0, 0)
        tc_costs = np.abs(trades) * self.tc

        strategy_returns_net = strategy_returns - tc_costs

        # Cumulative returns
        cum_market = np.exp(np.cumsum(log_returns)) * self.initial_capital
        cum_strategy = np.exp(np.cumsum(strategy_returns_net)) * self.initial_capital

        # Performance metrics
        n_trades = int(np.sum(np.abs(trades) > 0))
        total_return_market = (cum_market[-1] / self.initial_capital - 1) * 100
        total_return_strategy = (cum_strategy[-1] / self.initial_capital - 1) * 100

        # Annualized metrics (assuming 252 trading days)
        daily_strat_returns = strategy_returns_net[1:]
        if len(daily_strat_returns) > 0 and np.std(daily_strat_returns) > 0:
            sharpe = (np.mean(daily_strat_returns) / np.std(daily_strat_returns)) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino ratio (only penalizes downside volatility)
        downside = daily_strat_returns[daily_strat_returns < 0]
        if len(downside) > 0 and np.std(downside) > 0:
            sortino = (np.mean(daily_strat_returns) / np.std(downside)) * np.sqrt(252)
        else:
            sortino = 0.0

        # Maximum drawdown
        running_max = np.maximum.accumulate(cum_strategy)
        drawdown = (cum_strategy - running_max) / running_max * 100
        max_drawdown = np.min(drawdown)

        # Win rate
        winning_days = np.sum(daily_strat_returns > 0)
        total_days = len(daily_strat_returns)
        win_rate = (winning_days / total_days * 100) if total_days > 0 else 0

        # Build equity curve data
        equity_curve = []
        for i in range(n):
            equity_curve.append({
                "date": dates[i],
                "market": round(float(cum_market[i]), 2),
                "strategy": round(float(cum_strategy[i]), 2),
                "signal": int(shifted_signals[i]),
                "drawdown": round(float(drawdown[i]), 2),
            })

        return {
            "initial_capital": self.initial_capital,
            "final_value_market": round(float(cum_market[-1]), 2),
            "final_value_strategy": round(float(cum_strategy[-1]), 2),
            "total_return_market": round(total_return_market, 2),
            "total_return_strategy": round(total_return_strategy, 2),
            "outperformance": round(total_return_strategy - total_return_market, 2),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(float(max_drawdown), 2),
            "win_rate": round(win_rate, 1),
            "total_trades": n_trades,
            "transaction_costs_total": round(float(np.sum(tc_costs) * self.initial_capital), 2),
            "equity_curve": equity_curve,
        }


class EventBacktester:
    """
    Event-based backtesting with risk management (Chapter 11).

    Simulates trade-by-trade execution with:
    - Stop-loss orders
    - Take-profit orders
    - Position sizing
    - Trade log with entry/exit prices
    """

    def __init__(self, initial_capital: float = 10000.0,
                 stop_loss: float = 0.0175,   # 1.75% from book
                 take_profit: float = 0.0185,  # 1.85% from book
                 position_size: float = 0.1):  # 10% of capital per trade
        self.initial_capital = initial_capital
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size

    def run(self, prices: np.ndarray, signals: np.ndarray,
            dates: Optional[List[str]] = None) -> Dict:
        """
        Run event-based backtest with risk management.
        """
        n = len(prices)
        if dates is None:
            dates = [f"t{i}" for i in range(n)]

        capital = self.initial_capital
        position = 0       # 0=flat, 1=long, -1=short
        entry_price = 0.0
        trades = []
        equity = [capital]

        for i in range(1, n):
            signal = signals[i - 1]  # Use previous signal (no look-ahead)
            price = prices[i]

            # Check stop-loss / take-profit if in position
            if position != 0:
                pnl_pct = (price - entry_price) / entry_price * position

                if pnl_pct <= -self.stop_loss:
                    # Stop-loss triggered
                    trade_pnl = -self.stop_loss * capital * self.position_size
                    capital += trade_pnl
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": dates[i],
                        "direction": "LONG" if position == 1 else "SHORT",
                        "entry_price": round(entry_price, 2),
                        "exit_price": round(price, 2),
                        "pnl": round(trade_pnl, 2),
                        "pnl_pct": round(-self.stop_loss * 100, 2),
                        "exit_reason": "STOP_LOSS",
                    })
                    position = 0
                    entry_price = 0.0

                elif pnl_pct >= self.take_profit:
                    # Take-profit triggered
                    trade_pnl = self.take_profit * capital * self.position_size
                    capital += trade_pnl
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": dates[i],
                        "direction": "LONG" if position == 1 else "SHORT",
                        "entry_price": round(entry_price, 2),
                        "exit_price": round(price, 2),
                        "pnl": round(trade_pnl, 2),
                        "pnl_pct": round(self.take_profit * 100, 2),
                        "exit_reason": "TAKE_PROFIT",
                    })
                    position = 0
                    entry_price = 0.0

            # Open new position based on signal
            if position == 0:
                if signal == 1:
                    position = 1
                    entry_price = price
                    entry_date = dates[i]
                elif signal == -1:
                    position = -1
                    entry_price = price
                    entry_date = dates[i]

            # Close position on signal reversal
            elif (position == 1 and signal == -1) or (position == -1 and signal == 1):
                pnl_pct = (price - entry_price) / entry_price * position
                trade_pnl = pnl_pct * capital * self.position_size
                capital += trade_pnl
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": dates[i],
                    "direction": "LONG" if position == 1 else "SHORT",
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(price, 2),
                    "pnl": round(trade_pnl, 2),
                    "pnl_pct": round(pnl_pct * 100, 2),
                    "exit_reason": "SIGNAL_REVERSAL",
                })
                # Open opposite position
                position = 1 if signal == 1 else -1
                entry_price = price
                entry_date = dates[i]

            equity.append(capital)

        # Close any open position at the end
        if position != 0:
            price = prices[-1]
            pnl_pct = (price - entry_price) / entry_price * position
            trade_pnl = pnl_pct * capital * self.position_size
            capital += trade_pnl
            trades.append({
                "entry_date": entry_date,
                "exit_date": dates[-1],
                "direction": "LONG" if position == 1 else "SHORT",
                "entry_price": round(entry_price, 2),
                "exit_price": round(price, 2),
                "pnl": round(trade_pnl, 2),
                "pnl_pct": round(pnl_pct * 100, 2),
                "exit_reason": "END_OF_DATA",
            })
            equity.append(capital)

        # Compute trade statistics
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] <= 0]

        avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        return {
            "initial_capital": self.initial_capital,
            "final_capital": round(capital, 2),
            "total_return": round((capital / self.initial_capital - 1) * 100, 2),
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": round(len(winning_trades) / len(trades) * 100, 1) if trades else 0,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "stop_loss_hits": len([t for t in trades if t["exit_reason"] == "STOP_LOSS"]),
            "take_profit_hits": len([t for t in trades if t["exit_reason"] == "TAKE_PROFIT"]),
            "risk_params": {
                "stop_loss": f"{self.stop_loss * 100}%",
                "take_profit": f"{self.take_profit * 100}%",
                "position_size": f"{self.position_size * 100}%",
            },
            "trades": trades[-50:],  # Last 50 trades
            "equity_curve": [round(e, 2) for e in equity],
        }


def generate_model_signals(model, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Generate trading signals from a trained model.
    Used to bridge models.py predictions to backtester inputs.
    """
    predictions = model.predict(features, verbose=0).flatten()
    signals = np.where(predictions > threshold + 0.05, 1,
              np.where(predictions < threshold - 0.05, -1, 0))
    return signals
