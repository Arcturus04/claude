"""
Shared fixtures. Tests never touch the network: market data comes from a
deterministic synthetic price series, and credentials are cleared so the
Reddit, X and Claude clients stay unconfigured.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
for var in ("ANTHROPIC_API_KEY", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET",
            "TWITTER_BEARER_TOKEN"):
    os.environ.pop(var, None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# yfinance-style period strings mapped to trading sessions.
PERIOD_SESSIONS = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "5y": 1260}


def make_ohlcv(n: int = 1300, seed: int = 0, drift: float = 0.0003,
               vol: float = 0.01, end: str = "2026-10-16") -> pd.DataFrame:
    """A geometric random walk dressed as an OHLCV frame on business days."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    spread = np.abs(rng.normal(0, 0.005, n))
    return pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.002, n)),
        "High": close * (1 + spread),
        "Low": close * (1 - spread),
        "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n).astype(float),
    }, index=pd.bdate_range(end=end, periods=n))


class FakeTicker:
    """Stands in for yfinance.Ticker. Every period is a tail of one series,
    so training and backtest fetches agree on dates, as with real data."""

    known = {"SPY": make_ohlcv(seed=0), "AAPL": make_ohlcv(seed=1)}

    def __init__(self, symbol: str):
        self.symbol = symbol.upper()

    def history(self, period: str = "1y", **_):
        df = self.known.get(self.symbol)
        if df is None:
            return pd.DataFrame()
        return df.tail(PERIOD_SESSIONS.get(period, len(df))).copy()


@pytest.fixture
def prices():
    return make_ohlcv()


@pytest.fixture
def client(monkeypatch):
    """FastAPI test client with yfinance replaced by FakeTicker."""
    from fastapi.testclient import TestClient
    import server

    monkeypatch.setattr(server.yf, "Ticker", FakeTicker)
    with TestClient(server.app) as c:
        yield c
