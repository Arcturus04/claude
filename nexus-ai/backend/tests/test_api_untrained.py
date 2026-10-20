"""API behaviour before any training has happened."""


def test_fresh_server_reports_untrained(client):
    health = client.get("/api/health").json()
    assert health["training"]["status"] == "idle"
    assert all(m["status"] == "untrained" for m in health["models"].values())


def test_predict_requires_training(client):
    assert client.post("/api/predict", json={"symbol": "SPY"}).status_code == 400


def test_backtest_requires_training(client):
    assert client.post("/api/backtest", json={}).status_code == 400


def test_unknown_symbol_is_404_not_500(client):
    assert client.get("/api/market/ZZZZ").status_code == 404
