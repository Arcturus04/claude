"""
The label must be tomorrow's direction, and no feature may see tomorrow.

These tests exist because the original pipeline labelled each row with its
own return while computing indicators from that same close, so the models
were graded on answers already present in their inputs.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from conftest import make_ohlcv
from models import FeatureEngine


def test_label_is_next_session_direction(prices):
    f = FeatureEngine().compute_features(prices)
    labelled = f.dropna(subset=["direction"])
    close = prices["Close"]
    for date in labelled.index[:50]:
        pos = close.index.get_loc(date)
        went_up = close.iloc[pos + 1] > close.iloc[pos]
        assert labelled.loc[date, "direction"] == float(went_up)


def test_features_do_not_depend_on_future_prices(prices):
    """Rewrite every price after a cutoff; features up to it must not move."""
    fe = FeatureEngine()
    cols = fe.get_feature_columns()
    before = fe.compute_features(prices)

    cutoff = prices.index[700]
    altered = prices.copy()
    rng = np.random.default_rng(99)
    future = altered.index > cutoff
    shock = rng.uniform(0.5, 1.5, future.sum())
    for col in ("Open", "High", "Low", "Close"):
        altered.loc[future, col] *= shock
    altered.loc[future, "Volume"] *= rng.uniform(0.1, 10, future.sum())
    after = fe.compute_features(altered)

    common = before.index[before.index <= cutoff]
    pd.testing.assert_frame_equal(before.loc[common, cols], after.loc[common, cols])


def test_last_row_is_kept_for_live_prediction(prices):
    f = FeatureEngine().compute_features(prices)
    assert f.index[-1] == prices.index[-1]
    assert np.isnan(f["direction"].iloc[-1])
    assert not f[FeatureEngine().get_feature_columns()].iloc[-1].isna().any()


def test_prepare_data_is_chronological_and_records_cutoff(prices):
    fe = FeatureEngine()
    f = fe.compute_features(prices)
    X_tr, X_te, y_tr, y_te = fe.prepare_data(f, train_ratio=0.8)
    labelled = f.dropna(subset=["direction"])
    assert len(X_tr) + len(X_te) == len(labelled)
    assert fe.train_end_date == labelled.index[len(X_tr) - 1]
    assert set(np.unique(np.concatenate([y_tr, y_te]))) <= {0, 1}


def test_sentiment_is_not_a_model_input():
    # A single current sentiment reading copied onto every row has zero
    # variance in training and only adds noise at prediction time.
    assert "sentiment" not in FeatureEngine().get_feature_columns()


def test_random_walk_is_not_predictable():
    """No pattern exists in a random walk, so honest accuracy is ~50%.

    With the original same-day label this scored about 93%.
    """
    accs = []
    for seed in range(5):
        fe = FeatureEngine()
        f = fe.compute_features(make_ohlcv(n=1500, seed=seed, drift=0.0))
        X_tr, X_te, y_tr, y_te = fe.prepare_data(f)
        clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
        accs.append(clf.score(X_te, y_te) * 100)
    assert 44 < np.mean(accs) < 56, accs


def test_majority_baseline_uses_the_training_majority():
    # Training mostly up, so the naive forecast is "up" on every test day.
    assert FeatureEngine.majority_baseline([1, 1, 1, 0], y_train=[1, 1, 0]) == 75.0
    assert FeatureEngine.majority_baseline([], y_train=[1]) == 0.0


def test_majority_baseline_has_no_hindsight():
    # Training says "up"; a test period that went all down must score 0%,
    # not the 100% that picking the test period's own majority would give.
    assert FeatureEngine.majority_baseline([0, 0, 0, 0], y_train=[1, 1, 0]) == 0.0
    assert FeatureEngine.majority_baseline([0, 0, 0, 0], y_train=[0, 0, 1]) == 100.0
