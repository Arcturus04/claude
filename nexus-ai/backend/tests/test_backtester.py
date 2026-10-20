"""Backtest engines: no lookahead, honest fills, honest metrics."""

import json

import numpy as np

from backtester import EventBacktester, VectorizedBacktester


# ─── Vectorized ──────────────────────────────────────────────────────────────

def test_signal_is_applied_one_session_later():
    prices = np.array([100.0, 110.0, 99.0, 108.9, 108.9])
    signals = np.array([0, 1, 0, 0, 0])  # long call made at the close of day 1
    r = VectorizedBacktester(transaction_cost=0.0).run(prices, signals)
    strategy = [p["strategy"] for p in r["equity_curve"]]
    # Day 1's own +10% is NOT earned; day 2's -10% is.
    assert strategy[1] == 10000.0
    assert np.isclose(strategy[2], 10000 * 99 / 110, atol=0.01)
    assert np.isclose(strategy[-1], strategy[2])


def test_same_day_signal_cannot_capture_same_day_return():
    """A signal that knows today's move must not be able to trade it.

    Without the one-session delay it would earn every move (perfect
    foresight). With it, it is just yesterday's direction applied to today.
    """
    rng = np.random.default_rng(1)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    log_ret = np.diff(np.log(prices), prepend=np.log(prices[0]))
    cheat = np.sign(log_ret).astype(int)
    r = VectorizedBacktester(transaction_cost=0.0).run(prices, cheat)

    perfect = 10000 * np.exp(np.sum(np.abs(log_ret)))
    one_day_momentum = 10000 * np.exp(np.sum(cheat[:-1] * log_ret[1:]))
    assert np.isclose(r["final_value_strategy"], one_day_momentum, rtol=1e-6)
    assert r["final_value_strategy"] < perfect / 10


def test_sortino_uses_downside_deviation():
    prices = np.array([100.0, 101, 99, 100, 102, 98, 99, 103])
    signals = np.ones(len(prices), dtype=int)
    r = VectorizedBacktester(transaction_cost=0.0).run(prices, signals)
    strat = np.diff(np.log(prices))  # long from the first close onward
    dd = np.sqrt(np.mean(np.minimum(strat, 0) ** 2))
    assert np.isclose(r["sortino_ratio"], np.mean(strat) / dd * np.sqrt(252), atol=1e-3)


def test_transaction_costs_are_charged_on_traded_equity():
    prices = np.full(10, 100.0)
    signals = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    r = VectorizedBacktester(initial_capital=10000, transaction_cost=0.001).run(prices, signals)
    # One entry and one exit, each ~0.1% of ~$10,000.
    assert 19.9 < r["transaction_costs_total"] < 20.1


# ─── Event-based ─────────────────────────────────────────────────────────────

def test_gap_through_stop_is_booked_at_the_actual_price():
    prices = np.array([100.0, 100.0, 90.0, 90.0])
    signals = np.array([1, 0, 0, 0])
    r = EventBacktester(stop_loss=0.0175, transaction_cost=0.0).run(prices, signals)
    trade = r["trades"][0]
    assert trade["exit_reason"] == "STOP_LOSS"
    assert trade["exit_price"] == 90.0
    assert np.isclose(trade["pnl_pct"], -10.0)  # not the -1.75% stop level
    assert np.isclose(r["final_capital"], 10000 - 0.1 * 10000 * 0.10)


def test_round_trip_pays_costs_on_both_legs():
    prices = np.full(6, 100.0)
    signals = np.array([1, 0, -1, 0, 0, 0])  # enter long, then reverse
    r = EventBacktester(transaction_cost=0.001, position_size=0.1).run(prices, signals)
    first = r["trades"][0]
    assert first["exit_reason"] == "SIGNAL_REVERSAL"
    assert np.isclose(first["pnl"], -2.0, atol=0.01)  # $1,000 notional x 0.1% x 2
    assert r["transaction_costs_total"] > 0


def test_profit_factor_is_gross_profit_over_gross_loss():
    # Long +10%, reverse to short -5%, reverse to long -5% (closed at the end).
    # Stops and targets are set out of reach so only reversals close trades.
    prices = np.array([100.0, 100, 110, 115.5, 109.725])
    signals = np.array([1, -1, 1, 0, 0])
    r = EventBacktester(stop_loss=1.0, take_profit=1.0, transaction_cost=0.0).run(prices, signals)
    pnls = [t["pnl"] for t in r["trades"]]
    assert np.allclose(pnls, [100.0, -50.5, -50.25], atol=0.01)
    assert np.isclose(r["profit_factor"], 100.0 / 100.75, atol=0.01)       # gross / gross
    assert np.isclose(r["payoff_ratio"], 100.0 / 50.375, atol=0.01)        # avg / avg


def test_no_losing_trades_gives_null_not_infinity():
    prices = np.array([100.0, 100, 103, 103])
    signals = np.array([1, 0, 0, 0])
    r = EventBacktester(transaction_cost=0.0).run(prices, signals)
    assert r["losing_trades"] == 0
    assert r["profit_factor"] is None and r["payoff_ratio"] is None
    json.dumps(r, allow_nan=False)  # what the API's JSON encoder requires


def test_drawdown_is_marked_to_market():
    """An open position that dips then recovers must still show a drawdown."""
    prices = np.array([100.0, 100, 99, 98.5, 99.5, 101.9])
    signals = np.array([1, 0, 0, 0, 0, 0])
    r = EventBacktester(stop_loss=0.05, take_profit=0.0185, transaction_cost=0.0).run(prices, signals)
    assert r["trades"][0]["pnl"] > 0
    assert r["max_drawdown"] < 0
    assert len(r["equity_curve"]) == len(prices)
