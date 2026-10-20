"""Model plumbing: window alignment, greedy inference, honest scoring, skill weighting."""

import numpy as np

from models import DQLAgent, EnsemblePredictor, RNNPredictor


def test_rnn_window_takes_the_label_of_its_last_row():
    rnn = RNNPredictor(sequence_length=3, n_features=1)
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.array([0, 1, 0, 1, 1, 0])
    Xs, ys = rnn._create_sequences(X, y)
    assert len(Xs) == 4
    for window, label in zip(Xs, ys):
        last_row = int(window[-1, 0])
        assert label == y[last_row]


def test_rnn_probability_series_aligns_with_rows():
    rnn = RNNPredictor(sequence_length=4, n_features=2)
    X = np.random.default_rng(0).normal(size=(10, 2))
    probs = rnn.predict_proba_series(X)
    assert np.isnan(probs[:3]).all() and not np.isnan(probs[3:]).any()
    # Row 9's value must be the window ending on row 9.
    direct = rnn.model.predict(X[6:10].reshape(1, 4, 2), verbose=0)[0][0]
    assert np.isclose(probs[9], direct, atol=1e-5)


def test_dql_inference_never_explores():
    agent = DQLAgent(state_dim=4)
    agent.epsilon = 1.0  # would always act randomly while training
    state = np.random.default_rng(1).normal(size=4)
    greedy = int(np.argmax(agent.q_values(state)[0]))
    assert all(agent.act(state, explore=False) == greedy for _ in range(25))
    assert agent.get_signal(state) == agent.get_signal(state)


def test_dql_evaluate_scores_calls_and_excludes_holds(monkeypatch):
    agent = DQLAgent(state_dim=1)
    actions = np.array([1, 2, 0, 1, 2])  # BUY, SELL, HOLD, BUY, SELL
    monkeypatch.setattr(agent, "greedy_actions", lambda states: actions[: len(states)])
    prices = np.array([100.0, 101, 100, 99, 100, 99])  # up, down, down, up, down
    result = agent.evaluate(np.zeros((6, 1)), prices)
    # BUY up: hit; SELL down: hit; HOLD skipped; BUY up: hit; SELL down: hit.
    assert result["calls"] == 4
    assert result["hit_rate"] == 100.0
    assert np.isclose(result["coverage"], 80.0)
    assert result["baseline"] == 50.0


def test_dql_replay_takes_one_gradient_step_per_batch(monkeypatch):
    agent = DQLAgent(state_dim=3)
    rng = np.random.default_rng(2)
    for _ in range(64):
        s = rng.normal(size=3)
        agent.remember(s, int(rng.integers(3)), float(rng.normal()), rng.normal(size=3), False)
    calls = []
    monkeypatch.setattr(agent.model, "train_on_batch", lambda x, y: calls.append(x.shape))
    agent.replay()
    assert calls == [(agent.batch_size, 3)]


def test_ensemble_weights_follow_skill_over_baseline():
    ens = EnsemblePredictor()
    ens.update_weights({"dnn": (56.0, 52.0), "rnn": (51.0, 52.0), "dql": (None, None)})
    assert ens.weights["dnn"] == 1.0
    assert ens.weights["rnn"] == 0.0 and ens.weights["dql"] == 0.0
    assert ens.no_skill is False


def test_ensemble_warns_when_no_model_has_skill():
    ens = EnsemblePredictor()
    ens.update_weights({"dnn": (50.0, 53.0), "rnn": (49.0, 53.0), "dql": (52.0, 54.0)})
    assert ens.no_skill is True
    assert all(np.isclose(ens.weights[k], 1 / 3) for k in ("dnn", "rnn", "dql"))
    out = ens.predict((1, 0.9), (1, 0.9), {"action": "BUY", "confidence": 0.9})
    assert out["no_skill_warning"] is True and "warning" in out


def test_ensemble_weights_sum_to_one_with_igon():
    ens = EnsemblePredictor()
    ens.update_weights({"dnn": (60.0, 50.0), "rnn": (55.0, 50.0), "dql": (None, None)})
    out = ens.predict((1, 1.0), (1, 1.0), {"action": "BUY", "confidence": 1.0},
                      igon_signal={"score": 100})
    # Every input at full strength in the same direction gives exactly 1.0.
    assert np.isclose(out["combined_score"], 1.0)
