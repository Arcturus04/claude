"""
End-to-end through the HTTP API: train, predict, backtest.

Training is shortened to a single epoch/episode; these tests check what the
API reports and on which data, not how well the models trade.
"""

import json

import pytest
from fastapi.testclient import TestClient

from conftest import FakeTicker

TINY = {"symbol": "SPY", "period": "2y", "epochs_dnn": 1, "epochs_rnn": 1, "episodes_dql": 1}


@pytest.fixture(scope="module")
def trained():
    """Train once for the whole module. Tests needing an untrained server
    live in test_api_untrained.py, since starting another app would reset
    the shared server state mid-module."""
    import server

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(server.yf, "Ticker", FakeTicker)
        with TestClient(server.app) as client:
            r = client.post("/api/train", json=TINY)  # background task runs before return
            assert r.status_code == 200
            health = client.get("/api/health").json()
            assert health["training"]["status"] == "complete", health["training"]
            yield client, health


def test_health_reports_out_of_sample_metrics_with_baselines(trained):
    _, health = trained
    for name in ("dnn", "rnn"):
        m = health["models"][name]
        assert m["status"] == "active"
        assert m["accuracy"] is not None and m["baseline_accuracy"] is not None
        assert m["input_features"] == len(health["training"]["features"])
    dql = health["models"]["dql"]
    assert dql["status"] == "active"
    assert "out-of-sample" in dql["accuracy_definition"]
    t = health["training"]
    assert t["symbol"] == "SPY"
    assert t["test_period"][0] > t["train_end_date"]
    assert "sentiment" not in t["features"]
    assert "not a learned meta-model" in health["ensemble"]["type"]


def test_predict_forecasts_the_next_session(trained):
    client, _ = trained
    r = client.post("/api/predict", json={"symbol": "SPY"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicts"].startswith("direction of the next session")
    assert body["symbol_matches_training"] is True
    assert body["models"]["rnn"]["direction"] in ("LONG", "SHORT")
    assert "no_skill_warning" in body["ensemble"]


def test_predict_flags_a_symbol_the_models_were_not_trained_on(trained):
    client, _ = trained
    body = client.post("/api/predict", json={"symbol": "AAPL"}).json()
    assert body["symbol_matches_training"] is False
    assert body["trained_symbol"] == "SPY"


@pytest.mark.parametrize("model", ["dnn", "rnn", "ensemble"])
@pytest.mark.parametrize("mode", ["vectorized", "event"])
def test_backtest_only_trades_after_the_training_cutoff(trained, model, mode):
    client, health = trained
    r = client.post("/api/backtest", json={"symbol": "SPY", "period": "2y",
                                           "model": model, "mode": mode})
    assert r.status_code == 200, r.text
    body = r.json()
    cutoff = health["training"]["train_end_date"]
    oos = body["out_of_sample"]
    assert oos["start"] > cutoff and oos["train_end_date"] == cutoff
    dates = [p["date"] for p in body["equity_curve"]]
    assert dates[0] == oos["start"] and dates[-1] == oos["end"]
    assert all(d > cutoff for d in dates)
    json.dumps(body, allow_nan=False)


def test_backtest_refuses_too_little_unseen_data(trained):
    client, _ = trained
    r = client.post("/api/backtest", json={"symbol": "SPY", "period": "3mo"})
    assert r.status_code == 400
    assert "training cutoff" in r.json()["detail"]


def test_backtest_period_too_short_for_indicators_is_400_not_500(trained):
    client, _ = trained
    r = client.post("/api/backtest", json={"symbol": "SPY", "period": "1mo"})
    assert r.status_code == 400
    assert "too short" in r.json()["detail"]


def test_backtest_validates_model_and_mode(trained):
    client, _ = trained
    assert client.post("/api/backtest", json={"model": "gpt"}).status_code == 400
    assert client.post("/api/backtest", json={"mode": "live"}).status_code == 400
