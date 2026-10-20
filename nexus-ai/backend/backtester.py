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

        # Sortino ratio: mean return over downside deviation, the root mean
        # square of the below-zero returns taken over ALL sessions. (The
        # standard deviation of only the losing sessions measures how much the
        # losses vary among themselves and understates downside risk.)
        if len(daily_strat_returns) > 0:
            downside_dev = np.sqrt(np.mean(np.minimum(daily_strat_returns, 0.0) ** 2))
        else:
            downside_dev = 0.0
        if downside_dev > 0:
            sortino = (np.mean(daily_strat_returns) / downside_dev) * np.sqrt(252)
        else:
            sortino = 0.0

        # Maximum drawdown
        running_max = np.maximum.accumulate(cum_strategy)
        drawdown = (cum_strategy - running_max) / running_max * 100
        max_drawdown = np.min(drawdown)

        # Win rate over the sessions where a position was actually held; flat
        # sessions earn nothing and say nothing about the signal.
        in_market = shifted_signals[1:] != 0
        days_in_market = int(np.sum(in_market))
        winning_days = int(np.sum(daily_strat_returns[in_market] > 0))
        win_rate = (winning_days / days_in_market * 100) if days_in_market > 0 else 0

        # Dollar costs, charged on the equity actually traded at each change.
        equity_before = np.concatenate([[self.initial_capital], cum_strategy[:-1]])
        tc_dollars = float(np.sum(tc_costs * equity_before))

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
            "win_rate_definition": "share of in-market sessions with a positive net return",
            "days_in_market": days_in_market,
            "total_trades": n_trades,
            "transaction_costs_total": round(tc_dollars, 2),
            "equity_curve": equity_curve,
        }


class EventBacktester:
    """
    Event-based backtesting with risk management (Chapter 11).

    Simulates trade-by-trade execution with:
    - Stop-loss orders
    - Take-profit orders
    - Position sizing
    - Transaction costs on every entry and exit
    - Trade log with entry/exit prices

    Fill model: only daily closes are available, so stops and targets are
    checked at each close and filled AT THAT CLOSE. A position that gaps
    through its stop is booked at the actual gapped price, not at the stop
    level; booking it at the stop would understate losses exactly when they
    are largest. Real intraday fills could land better or worse.
    """

    def __init__(self, initial_capital: float = 10000.0,
                 stop_loss: float = 0.0175,   # 1.75% from book
                 take_profit: float = 0.0185,  # 1.85% from book
                 position_size: float = 0.1,   # 10% of capital per trade
                 transaction_cost: float = 0.001):  # 10 bps per side
        self.initial_capital = initial_capital
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.tc = transaction_cost

    def run(self, prices: np.ndarray, signals: np.ndarray,
            dates: Optional[List[str]] = None) -> Dict:
        """
        Run event-based backtest with risk management.
        """
        n = len(prices)
        if len(signals) != n:
            raise ValueError("Prices and signals must be same length")
        if dates is None:
            dates = [f"t{i}" for i in range(n)]

        capital = self.initial_capital
        position = 0        # 0=flat, 1=long, -1=short
        entry_price = 0.0
        entry_date = None
        notional = 0.0      # capital committed at entry
        entry_cost = 0.0    # transaction cost paid at entry
        trades = []
        costs_total = 0.0

        def close_position(i: int, reason: str) -> None:
            nonlocal capital, position, entry_price, entry_date, notional, entry_cost, costs_total
            price = prices[i]
            pnl_pct = (price - entry_price) / entry_price * position
            # Value of the exit trade (shares x exit price), long or short.
            exit_value = notional * price / entry_price
            exit_cost = exit_value * self.tc
            trade_pnl = notional * pnl_pct - entry_cost - exit_cost
            capital += notional * pnl_pct - exit_cost  # entry cost already paid
            costs_total += exit_cost
            trades.append({
                "entry_date": entry_date,
                "exit_date": dates[i],
                "direction": "LONG" if position == 1 else "SHORT",
                "entry_price": round(float(entry_price), 2),
                "exit_price": round(float(price), 2),
                "pnl": round(float(trade_pnl), 2),
                "pnl_pct": round(float(trade_pnl / notional * 100), 2),
                "exit_reason": reason,
            })
            position, entry_price, entry_date, notional, entry_cost = 0, 0.0, None, 0.0, 0.0

        def open_position(i: int, direction: int) -> None:
            nonlocal capital, position, entry_price, entry_date, notional, entry_cost, costs_total
            position = direction
            entry_price = prices[i]
            entry_date = dates[i]
            notional = capital * self.position_size
            entry_cost = notional * self.tc
            capital -= entry_cost
            costs_total += entry_cost

        def mark_to_market(i: int) -> float:
            if position == 0:
                return capital
            return capital + notional * (prices[i] - entry_price) / entry_price * position

        equity = [capital]
        for i in range(1, n):
            signal = signals[i - 1]  # Use previous signal (no look-ahead)

            # Check stop-loss / take-profit at this close, filled at this close
            if position != 0:
                pnl_pct = (prices[i] - entry_price) / entry_price * position
                if pnl_pct <= -self.stop_loss:
                    close_position(i, "STOP_LOSS")
                elif pnl_pct >= self.take_profit:
                    close_position(i, "TAKE_PROFIT")

            # Open new position based on signal
            if position == 0:
                if signal in (1, -1):
                    open_position(i, int(signal))

            # Close and reverse on an opposite signal
            elif (position == 1 and signal == -1) or (position == -1 and signal == 1):
                close_position(i, "SIGNAL_REVERSAL")
                open_position(i, int(signal))

            equity.append(mark_to_market(i))

        # Close any open position at the end
        if position != 0:
            close_position(n - 1, "END_OF_DATA")
            equity[-1] = capital

        # Compute trade statistics
        pnls = np.array([t["pnl"] for t in trades], dtype=float)
        wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
        avg_win = float(wins.mean()) if wins.size else 0.0
        avg_loss = float(losses.mean()) if losses.size else 0.0
        gross_loss = float(-losses.sum())
        # Profit factor = gross profit / gross loss. (Average win / average loss
        # is a different metric, the payoff ratio, reported separately.) Both
        # are undefined without a loss, reported as null rather than infinity,
        # which is not valid JSON.
        profit_factor = float(wins.sum()) / gross_loss if gross_loss > 0 else None
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else None

        equity_arr = np.array(equity, dtype=float)
        running_max = np.maximum.accumulate(equity_arr)
        max_drawdown = float(np.min((equity_arr - running_max) / running_max * 100))

        return {
            "initial_capital": self.initial_capital,
            "final_capital": round(capital, 2),
            "total_return": round((capital / self.initial_capital - 1) * 100, 2),
            "total_trades": len(trades),
            "winning_trades": int(wins.size),
            "losing_trades": int(losses.size),
            "win_rate": round(wins.size / len(trades) * 100, 1) if trades else 0,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
            "payoff_ratio": round(payoff_ratio, 2) if payoff_ratio is not None else None,
            "max_drawdown": round(max_drawdown, 2),
            "transaction_costs_total": round(costs_total, 2),
            "stop_loss_hits": len([t for t in trades if t["exit_reason"] == "STOP_LOSS"]),
            "take_profit_hits": len([t for t in trades if t["exit_reason"] == "TAKE_PROFIT"]),
            "risk_params": {
                "stop_loss": f"{self.stop_loss * 100}%",
                "take_profit": f"{self.take_profit * 100}%",
                "position_size": f"{self.position_size * 100}%",
                "transaction_cost": f"{self.tc * 100}% per side",
                "fills": "stops and targets checked and filled at the daily close",
            },
            "trades": trades[-50:],  # Last 50 trades
            # Same shape as the vectorized curve: strategy equity alongside a
            # buy-and-hold benchmark over the same sessions.
            "equity_curve": [
                {
                    "date": dates[i],
                    "strategy": round(float(e), 2),
                    "market": round(float(self.initial_capital * prices[i] / prices[0]), 2),
                }
                for i, e in enumerate(equity)
            ],
            "total_return_market": round((prices[-1] / prices[0] - 1) * 100, 2),
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
