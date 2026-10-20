"""Model plumbing: window alignment, greedy inference, honest scoring, skill weighting."""

import numpy as np

from models import DQLAgent, EnsemblePredictor, RNNPredictor, paired_skill_test


def test_paired_skill_test_values():
    # Model right on all 10 disagreements: 0.5 ** 10.
    assert np.isclose(paired_skill_test([True] * 10, [False] * 10), 0.5 ** 10)
    # Identical results: nothing distinguishes them.
    assert paired_skill_test([True, False], [True, False]) == 1.0
    # Model worse than baseline: no evidence of skill.
    assert paired_skill_test([False] * 5, [True] * 5) == 1.0
    assert paired_skill_test([], []) is None


def test_random_guessing_is_rarely_called_significant():
    """False-positive rate on a no-skill model should be about the 5% level."""
    rng = np.random.default_rng(0)
    trials, n, flagged = 500, 90, 0
    for _ in range(trials):
        y = rng.random(n) < 0.52
        guesses = rng.random(n) < 0.5
        p = paired_skill_test(guesses == y, y)  # baseline: always "up"
        flagged += p < 0.05
    assert flagged / trials < 0.08


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
    result = agent.evaluate(np.zeros((6, 1)), prices, majority_up=True)
    # BUY up: hit; SELL down: hit; HOLD skipped; BUY up: hit; SELL down: hit.
    assert result["calls"] == 4
    assert result["hit_rate"] == 100.0
    assert np.isclose(result["coverage"], 80.0)
    # Always calling "up" on the four called sessions: 2 of 4.
    assert result["baseline"] == 50.0


def test_dql_baseline_follows_the_training_majority(monkeypatch):
    agent = DQLAgent(state_dim=1)
    monkeypatch.setattr(agent, "greedy_actions", lambda states: np.ones(len(states), dtype=int))
    prices = np.array([100.0, 99, 98, 97, 98])  # down, down, down, up
    assert agent.evaluate(np.zeros((5, 1)), prices, majority_up=True)["baseline"] == 25.0
    assert agent.evaluate(np.zeros((5, 1)), prices, majority_up=False)["baseline"] == 75.0


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


def test_ensemble_weights_follow_significant_skill():
    ens = EnsemblePredictor()
    ens.update_weights({"dnn": (56.0, 52.0, 0.01), "rnn": (51.0, 52.0, 0.6),
                        "dql": (None, None, None)})
    assert ens.weights["dnn"] == 1.0
    assert ens.weights["rnn"] == 0.0 and ens.weights["dql"] == 0.0
    assert ens.no_skill is False


def test_ensemble_ignores_an_edge_that_is_not_significant():
    # 56% vs 50% looks like an edge but p = 0.3: indistinguishable from luck.
    ens = EnsemblePredictor()
    ens.update_weights({"dnn": (56.0, 50.0, 0.3), "rnn": (49.0, 50.0, 0.7),
                        "dql": (None, None, None)})
    assert ens.no_skill is True
    assert all(np.isclose(ens.weights[k], 1 / 3) for k in ("dnn", "rnn", "dql"))


def test_ensemble_warns_when_no_model_has_skill():
    ens = EnsemblePredictor()
    ens.update_weights({"dnn": (50.0, 53.0, 0.8), "rnn": (49.0, 53.0, 0.9),
                        "dql": (52.0, 54.0, 0.7)})
    assert ens.no_skill is True
    out = ens.predict((1, 0.9), (1, 0.9), {"action": "BUY", "confidence": 0.9})
    assert out["no_skill_warning"] is True and "warning" in out


def test_ensemble_weights_sum_to_one_with_igon():
    ens = EnsemblePredictor()
    ens.update_weights({"dnn": (60.0, 50.0, 0.01), "rnn": (55.0, 50.0, 0.02),
                        "dql": (None, None, None)})
    out = ens.predict((1, 1.0), (1, 1.0), {"action": "BUY", "confidence": 1.0},
                      igon_signal={"score": 100})
    # Every input at full strength in the same direction gives exactly 1.0.
    assert np.isclose(out["combined_score"], 1.0)
