"""
╔══════════════════════════════════════════════════════════════╗
║  NEXUS•AI — Machine Learning Trading Models                 ║
║  DNN, RNN, DQL architectures from Hilpisch (2021)           ║
║  Enhanced with sentiment features                           ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
import random
import json
import os

# TensorFlow imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import binomtest

logger = logging.getLogger(__name__)

SIGNIFICANCE_LEVEL = 0.05


def paired_skill_test(model_correct, baseline_correct) -> Optional[float]:
    """
    One-sided exact McNemar test: does the model beat the baseline on the
    same sessions by more than chance?

    Only sessions where exactly one of the two is right carry information.
    If the model had no edge, it would win each of those like a coin flip.
    Returns the p-value, or None when there is nothing to compare.

    Daily direction is hard to predict, and a few points of accuracy over a
    few dozen test sessions is well within noise; on random-walk prices a
    model "beats" the baseline about half the time by luck alone.
    """
    model_correct = np.asarray(model_correct, dtype=bool)
    baseline_correct = np.asarray(baseline_correct, dtype=bool)
    if model_correct.size == 0:
        return None
    wins = int(np.sum(model_correct & ~baseline_correct))
    losses = int(np.sum(~model_correct & baseline_correct))
    if wins + losses == 0:
        return 1.0
    return float(binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue)


def skill_fields(accuracy: Optional[float], baseline: Optional[float],
                 p_value: Optional[float]) -> Dict:
    """Accuracy reported together with what it has to beat and whether it did."""
    def r(v, nd=2):
        return round(v, nd) if v is not None else None
    return {
        "accuracy": r(accuracy),
        "baseline_accuracy": r(baseline),
        "p_value": r(p_value, 4),
        "significant": p_value is not None and p_value < SIGNIFICANCE_LEVEL,
    }


class FeatureEngine:
    """
    Transforms raw OHLCV price data into ML-ready features.

    Every feature on row t is computed from data up to and including the close
    of day t, and the label on row t is the direction of the *next* day's
    return. The models therefore learn to predict t -> t+1 from what was
    knowable at the close of t.

    Social sentiment is deliberately not a model input. The scraper produces a
    single current reading, not a history aligned with the price series, so as
    a feature it was one constant copied onto every row: the models could learn
    nothing from it during training, and any different value at prediction time
    only perturbed the output through an untrained weight. Sentiment is used by
    the Claude analysis layer instead.
    """

    def __init__(self, lags: int = 5):
        self.lags = lags
        self.scaler = StandardScaler()
        self.fitted = False
        self.train_end_date = None

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from OHLCV data:
        - Log returns (today's and the previous lags - 1 days)
        - Moving averages (SMA, EMA) relative to the close
        - Volatility (rolling std)
        - RSI, MACD signals
        - Volume momentum
        - ATR

        Rows lost to indicator warm-up are dropped. The final row is kept with
        an empty label: its features are the input for a live prediction of the
        next session, whose outcome is not yet known.
        """
        data = df.copy()

        # Log returns
        data["log_return"] = np.log(data["Close"] / data["Close"].shift(1))

        # Recent returns as features. return_lag_0 is today's return, which is
        # known at today's close; return_lag_k is the return k days earlier.
        for lag in range(self.lags):
            data[f"return_lag_{lag}"] = data["log_return"].shift(lag)

        # Moving averages
        data["sma_20"] = data["Close"].rolling(20).mean() / data["Close"] - 1
        data["sma_50"] = data["Close"].rolling(50).mean() / data["Close"] - 1
        data["ema_12"] = data["Close"].ewm(span=12).mean() / data["Close"] - 1

        # Volatility
        data["volatility_20"] = data["log_return"].rolling(20).std()

        # RSI (14-period)
        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data["rsi"] = 100 - (100 / (1 + rs))
        data["rsi_normalized"] = (data["rsi"] - 50) / 50  # Scale to [-1, 1]

        # MACD
        ema_12 = data["Close"].ewm(span=12).mean()
        ema_26 = data["Close"].ewm(span=26).mean()
        data["macd"] = (ema_12 - ema_26) / data["Close"]

        # Volume momentum
        if "Volume" in data.columns:
            data["volume_sma"] = data["Volume"].rolling(20).mean()
            data["volume_ratio"] = data["Volume"] / data["volume_sma"] - 1
        else:
            data["volume_ratio"] = 0

        # ATR (Average True Range) — from the book's risk chapter
        if all(col in data.columns for col in ["High", "Low"]):
            high_low = data["High"] - data["Low"]
            high_close = abs(data["High"] - data["Close"].shift(1))
            low_close = abs(data["Low"] - data["Close"].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data["atr"] = true_range.rolling(14).mean() / data["Close"]
        else:
            data["atr"] = data["volatility_20"]

        # Label: did the NEXT session close higher? Labelling a row with its own
        # return leaked the answer, because the moving averages, RSI, MACD,
        # volatility and ATR on that row are all computed from the same close
        # that defines it. A linear model scored ~93% out of sample on pure
        # random-walk prices with that label; with this one it scores ~50%.
        data["next_log_return"] = data["log_return"].shift(-1)
        data["direction"] = np.where(data["next_log_return"] > 0, 1.0, 0.0)
        data.loc[data["next_log_return"].isna(), "direction"] = np.nan

        # Drop indicator warm-up rows only; the last row keeps its NaN label.
        data = data.dropna(subset=self.get_feature_columns() + ["log_return"])

        return data

    def get_feature_columns(self) -> List[str]:
        """Return the feature column names used for training."""
        cols = [f"return_lag_{i}" for i in range(self.lags)]
        cols += [
            "sma_20", "sma_50", "ema_12",
            "volatility_20", "rsi_normalized", "macd",
            "volume_ratio", "atr",
        ]
        return cols

    def prepare_data(self, df: pd.DataFrame, train_ratio: float = 0.8):
        """
        Chronological train/test split and scaling.

        Only rows with a known label are used. The scaler is fit on the
        training rows only, and the date of the last training row is recorded
        so that backtests can be restricted to dates the models never saw.
        """
        labelled = df.dropna(subset=["direction"])
        feature_cols = self.get_feature_columns()
        X = labelled[feature_cols].values
        y = labelled["direction"].values.astype(int)

        split = int(len(X) * train_ratio)
        if split < 1 or split >= len(X):
            raise ValueError(
                f"Not enough data to split: {len(X)} labelled rows after indicator warm-up"
            )
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        self.train_end_date = labelled.index[split - 1]

        # Fit scaler on training data only
        self.scaler.fit(X_train)
        self.fitted = True

        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def majority_baseline(y_test, y_train) -> float:
        """
        Test accuracy of always predicting the training set's most common
        direction (scikit-learn's DummyClassifier "most_frequent").

        A directional model is only skilful if it beats this. In a rising
        market "always up" can score well above 50%, so a raw accuracy figure
        quoted without its baseline says little. The majority is taken from
        the training data because that is all a forecaster could have known;
        taking it from the test data would use hindsight, and on small test
        sets it inflates the baseline (to ~55% on pure noise).
        """
        y_test, y_train = np.asarray(y_test), np.asarray(y_train)
        if len(y_test) == 0:
            return 0.0
        majority_up = len(y_train) == 0 or float(np.mean(y_train)) >= 0.5
        hit = y_test == (1 if majority_up else 0)
        return float(np.mean(hit) * 100)


class DNNPredictor:
    """
    Dense Neural Network for market direction prediction.
    Architecture from Hilpisch Chapter 7.
    """

    def __init__(self, input_dim: int = 13, hidden_units: int = 64,
                 dropout_rate: float = 0.3, learning_rate: float = 0.001):
        self.input_dim = input_dim
        self.model = self._build_model(input_dim, hidden_units, dropout_rate, learning_rate)
        self.history = None
        self.accuracy = 0.0
        self.baseline_accuracy = None
        self.p_value = None
        self.trained = False
        self.name = "DNN Predictor"

    def _build_model(self, input_dim, hu, dropout, lr):
        model = Sequential([
            Dense(hu, input_dim=input_dim, activation="relu"),
            Dropout(dropout),
            Dense(hu, activation="relu"),
            Dropout(dropout),
            Dense(hu // 2, activation="relu"),
            Dropout(dropout / 2),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=lr),
            metrics=["accuracy"],
        )
        return model

    def train(self, X_train, y_train, X_test, y_test,
              epochs: int = 50, batch_size: int = 32):
        """Train the DNN model."""
        logger.info(f"Training DNN: {epochs} epochs, batch_size={batch_size}")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        y_pred = (self.model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
        self.accuracy = accuracy_score(y_test, y_pred) * 100
        self.baseline_accuracy = FeatureEngine.majority_baseline(y_test, y_train)
        majority = 1 if np.mean(y_train) >= 0.5 else 0
        self.p_value = paired_skill_test(y_pred == y_test, y_test == majority)
        self.trained = True
        logger.info(
            f"DNN out-of-sample accuracy: {self.accuracy:.2f}% "
            f"(majority-class baseline {self.baseline_accuracy:.2f}%, p={self.p_value:.3f})"
        )
        return self.accuracy

    def predict(self, X) -> Tuple[int, float]:
        """Predict next-session direction and confidence."""
        prob = self.model.predict(X.reshape(1, -1), verbose=0)[0][0]
        direction = 1 if prob > 0.5 else 0
        confidence = prob if direction == 1 else 1 - prob
        return direction, float(confidence)

    def get_status(self) -> Dict:
        return {
            "name": self.name,
            "type": "Dense Neural Network",
            **skill_fields(self.accuracy if self.trained else None,
                           self.baseline_accuracy, self.p_value),
            "status": "active" if self.trained else "untrained",
            "input_features": self.input_dim,
            "architecture": "Dense(64) → Dense(64) → Dense(32) → Dense(1)",
        }


class RNNPredictor:
    """
    Recurrent Neural Network (LSTM) for sequence-based prediction.
    Architecture from Hilpisch Chapter 8.
    """

    def __init__(self, sequence_length: int = 20, n_features: int = 13,
                 hidden_units: int = 64, learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model(n_features, hidden_units, learning_rate)
        self.accuracy = 0.0
        self.baseline_accuracy = None
        self.p_value = None
        self.trained = False
        self.name = "RNN Sequencer"

    def _build_model(self, n_features, hu, lr):
        model = Sequential([
            LSTM(hu, input_shape=(self.sequence_length, n_features),
                 return_sequences=True),
            Dropout(0.2),
            LSTM(hu // 2),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=lr),
            metrics=["accuracy"],
        )
        return model

    def _create_sequences(self, X, y):
        """
        Create sliding-window sequences for LSTM input.

        Each window ends on row i + L - 1 and takes that row's label, which is
        the direction of the session after the window: the same next-session
        target the DNN predicts from a single row.
        """
        L = self.sequence_length
        Xs, ys = [], []
        for i in range(len(X) - L + 1):
            Xs.append(X[i:i + L])
            ys.append(y[i + L - 1])
        return np.array(Xs), np.array(ys)

    def train(self, X_train, y_train, X_test, y_test,
              epochs: int = 30, batch_size: int = 32):
        """Train the LSTM model."""
        logger.info(f"Training RNN/LSTM: {epochs} epochs")

        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)

        # Give each test window the preceding training rows as context so that
        # windows end on every test row, and the LSTM is scored on exactly the
        # same sessions as the DNN. Context rows contribute features only (past
        # data, known at prediction time); every label scored is a test label.
        ctx = self.sequence_length - 1
        X_test_ctx = np.vstack([X_train[-ctx:], X_test]) if ctx else X_test
        y_test_ctx = np.concatenate([y_train[-ctx:], y_test]) if ctx else y_test
        X_test_seq, y_test_seq = self._create_sequences(X_test_ctx, y_test_ctx)

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            logger.warning("Not enough data for RNN sequences")
            return 0.0

        self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_test_seq, y_test_seq),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        y_pred = (self.model.predict(X_test_seq, verbose=0) > 0.5).astype(int).flatten()
        self.accuracy = accuracy_score(y_test_seq, y_pred) * 100
        self.baseline_accuracy = FeatureEngine.majority_baseline(y_test_seq, y_train_seq)
        majority = 1 if np.mean(y_train_seq) >= 0.5 else 0
        self.p_value = paired_skill_test(y_pred == y_test_seq, y_test_seq == majority)
        self.trained = True
        logger.info(
            f"RNN out-of-sample accuracy: {self.accuracy:.2f}% "
            f"(majority-class baseline {self.baseline_accuracy:.2f}%, p={self.p_value:.3f})"
        )
        return self.accuracy

    def predict(self, X_sequence) -> Tuple[int, float]:
        """Predict next-session direction from the most recent window."""
        if len(X_sequence) < self.sequence_length:
            raise ValueError(
                f"LSTM needs {self.sequence_length} rows of features, got {len(X_sequence)}"
            )
        seq = X_sequence[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        prob = self.model.predict(seq, verbose=0)[0][0]
        direction = 1 if prob > 0.5 else 0
        confidence = prob if direction == 1 else 1 - prob
        return direction, float(confidence)

    def predict_proba_series(self, X) -> np.ndarray:
        """
        Up-probability for every row that ends a full window, NaN otherwise.

        Row r's value comes from the window ending on row r, so it is a
        forecast for the session after r: the alignment backtests expect.
        """
        L = self.sequence_length
        out = np.full(len(X), np.nan)
        if len(X) < L:
            return out
        windows = np.stack([X[i - L + 1:i + 1] for i in range(L - 1, len(X))])
        out[L - 1:] = self.model.predict(windows, verbose=0).flatten()
        return out

    def get_status(self) -> Dict:
        return {
            "name": self.name,
            "type": "LSTM Recurrent Neural Net",
            **skill_fields(self.accuracy if self.trained else None,
                           self.baseline_accuracy, self.p_value),
            "status": "active" if self.trained else "untrained",
            "input_features": self.n_features,
            "architecture": f"LSTM(64, seq={self.sequence_length}) → LSTM(32) → Dense(1)",
        }


class DQLAgent:
    """
    Deep Q-Learning Agent for trading.
    Architecture from Hilpisch Chapter 9.

    The agent learns an action-value function Q(s,a) that maps
    states to action values. Actions: 0=HOLD, 1=BUY, 2=SELL.
    Uses experience replay and epsilon-greedy exploration.
    """

    def __init__(self, state_dim: int = 13, gamma: float = 0.95,
                 hidden_units: int = 64, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.gamma = gamma          # Discount factor for delayed rewards
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.01     # Minimum exploration
        self.epsilon_decay = 0.995  # Decay per replay step
        self.batch_size = 32
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.model = self._build_model(state_dim, hidden_units, learning_rate)
        self.accuracy = None        # Out-of-sample directional hit rate
        self.baseline_accuracy = None
        self.p_value = None
        self.coverage = None        # Share of test sessions with a BUY/SELL call
        self.total_reward = 0.0     # Best training-episode reward (in-sample)
        self.trained = False
        self.name = "DQL Agent"

    def _build_model(self, state_dim, hu, lr):
        """Q-network: state → Q-values for each action."""
        model = Sequential([
            Dense(hu, input_dim=state_dim, activation="relu"),
            Dense(hu, activation="relu"),
            Dense(3, activation="linear"),  # 3 actions: HOLD, BUY, SELL
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=lr))
        return model

    def q_values(self, states: np.ndarray) -> np.ndarray:
        """Q-values for a batch of states, shape (n, 3).

        Calls the network directly: Keras' predict() carries tens of
        milliseconds of fixed overhead per call, which dominated training time
        when it was invoked per sample.
        """
        states = np.asarray(states, dtype=np.float32).reshape(-1, self.state_dim)
        return self.model(states, training=False).numpy()

    def act(self, state, explore: bool = True) -> int:
        """Choose an action: epsilon-greedy while training, greedy otherwise.

        Prediction and backtesting must pass explore=False. With exploration on,
        a trained agent still picks a random action with probability epsilon,
        so the same state could yield different signals on different calls.
        """
        if explore and random.random() <= self.epsilon:
            return random.randint(0, 2)  # Explore
        return int(np.argmax(self.q_values(state)[0]))  # Exploit

    def greedy_actions(self, states: np.ndarray) -> np.ndarray:
        """Greedy action for every state in a batch (0=HOLD, 1=BUY, 2=SELL)."""
        if len(states) == 0:
            return np.zeros(0, dtype=int)
        return np.argmax(self.q_values(states), axis=1)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train on a batch of experiences from the replay buffer.

        One batched forward pass for the current and next states and one
        gradient step for the whole batch, instead of two predict() calls and
        a fit() per sample. Same Q-learning update:
            Q(s, a) <- r + gamma * max_a' Q(s', a')
        """
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=bool)

        targets = self.q_values(states)
        next_max = self.q_values(next_states).max(axis=1)
        updated = rewards + np.where(dones, 0.0, self.gamma * next_max)
        targets[np.arange(self.batch_size), actions] = updated
        self.model.train_on_batch(states, targets)

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def evaluate(self, features: np.ndarray, prices: np.ndarray,
                 majority_up: bool = True) -> Dict:
        """
        Out-of-sample score on data the agent did not train on.

        The agent acts greedily on every session but the last. A BUY is a hit
        if the next close is higher, a SELL if it is lower; HOLDs make no
        directional claim and are excluded, which is why coverage is reported
        alongside the hit rate.

        majority_up is the training set's majority direction. The baseline is
        the hit rate of always calling that direction on the same sessions.
        """
        empty = {"hit_rate": None, "baseline": None, "p_value": None,
                 "coverage": 0.0, "calls": 0}
        if len(features) < 2:
            return empty
        actions = self.greedy_actions(features[:-1])
        next_move = np.sign(np.diff(prices))
        called = actions != 0
        if not called.any():
            return empty
        predicted = np.where(actions == 1, 1.0, -1.0)
        hits = (predicted == next_move) & called
        # Baseline: always calling the training majority on the same sessions.
        naive = 1.0 if majority_up else -1.0
        naive_hits = next_move[called] == naive
        return {
            "hit_rate": float(hits.sum() / called.sum() * 100),
            "baseline": float(np.mean(naive_hits) * 100),
            "p_value": paired_skill_test(hits[called], naive_hits),
            "coverage": float(called.mean() * 100),
            "calls": int(called.sum()),
        }

    def train_on_data(self, features: np.ndarray, prices: np.ndarray,
                      episodes: int = 50):
        """
        Train the DQL agent on historical data.
        Simulates trading and learns from rewards.

        Pass training rows only. Scoring is done separately by evaluate() on
        held-out rows; the share of profitable training episodes this method
        used to report was an in-sample number, not an accuracy.
        """
        logger.info(f"Training DQL Agent: {episodes} episodes")

        best_reward = -np.inf
        rewards_history = []

        for episode in range(episodes):
            total_reward = 0
            position = 0  # -1, 0, 1

            for t in range(1, len(features) - 1):
                state = features[t]
                action = self.act(state)

                # Calculate reward based on action and price movement
                price_change = (prices[t + 1] - prices[t]) / prices[t]

                if action == 1:     # BUY
                    reward = price_change * 100
                    position = 1
                elif action == 2:   # SELL
                    reward = -price_change * 100
                    position = -1
                else:               # HOLD
                    reward = abs(price_change) * -10  # Small penalty for inaction
                    if position == 1:
                        reward = price_change * 50
                    elif position == -1:
                        reward = -price_change * 50

                next_state = features[t + 1] if t + 1 < len(features) else state
                done = (t == len(features) - 2)

                self.remember(state, action, reward, next_state, done)
                self.replay()

                total_reward += reward

            rewards_history.append(total_reward)
            if total_reward > best_reward:
                best_reward = total_reward

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                logger.info(
                    f"DQL Episode {episode + 1}/{episodes} | "
                    f"Reward: {total_reward:.2f} | "
                    f"Best: {best_reward:.2f} | "
                    f"ε: {self.epsilon:.4f}"
                )

        self.total_reward = best_reward if rewards_history else 0.0
        self.trained = True

        logger.info(f"DQL training complete. Best training reward: {self.total_reward:.2f}")
        return self.total_reward

    def get_signal(self, state) -> Dict:
        """Get the greedy trading signal for one state."""
        q_values = self.q_values(state)[0]
        action = int(np.argmax(q_values))
        confidence = float(np.max(q_values) / (np.sum(np.abs(q_values)) + 1e-8))

        signals = {0: "HOLD", 1: "BUY", 2: "SELL"}
        return {
            "action": signals[action],
            "confidence": round(min(max(confidence, 0.0), 1.0), 4),
            "q_values": {signals[i]: round(float(q_values[i]), 4) for i in range(3)},
        }

    def get_status(self) -> Dict:
        return {
            "name": self.name,
            "type": "Deep Q-Learning",
            **skill_fields(self.accuracy, self.baseline_accuracy, self.p_value),
            "accuracy_definition": "out-of-sample hit rate on BUY/SELL calls",
            "coverage": round(self.coverage, 2) if self.coverage is not None else None,
            "status": "active" if self.trained else "untrained",
            "input_features": self.state_dim,
            "epsilon": round(self.epsilon, 4),
            "gamma": self.gamma,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "memory_size": len(self.memory),
            "total_reward": round(self.total_reward, 2),
            "architecture": "Dense(64) → Dense(64) → Dense(3) [Q-values]",
        }


class EnsemblePredictor:
    """
    Weighted average of the DNN, RNN and DQL signals.

    This is a fixed weighting rule, not a learned stacking model. A model
    earns weight only if it beat its majority-class baseline out of sample
    with statistical significance (one-sided McNemar test, p < 0.05); its
    weight is then its accuracy in excess of that baseline. If no model
    qualifies, the weights fall back to equal and every prediction carries a
    warning saying the signal is not evidence of skill.
    """

    IGON_WEIGHT = 0.15

    def __init__(self):
        self.name = "Ensemble α"
        self.weights = {"dnn": 1 / 3, "rnn": 1 / 3, "dql": 1 / 3, "igon": self.IGON_WEIGHT}
        self.skill = {}
        self.no_skill = True

    def predict(self, dnn_signal: Tuple, rnn_signal: Tuple,
                dql_signal: Dict, igon_signal: Optional[Dict] = None) -> Dict:
        """
        Combine model outputs into a unified signal.
        Includes Igon value screen when fundamental data is available.
        """
        # Convert all to direction scores [-1, 1]
        dnn_dir = (dnn_signal[0] * 2 - 1) * dnn_signal[1]   # direction * confidence
        rnn_dir = (rnn_signal[0] * 2 - 1) * rnn_signal[1]

        dql_action = dql_signal["action"]
        dql_dir = (1 if dql_action == "BUY" else -1 if dql_action == "SELL" else 0)
        dql_dir *= dql_signal["confidence"]

        # Igon value signal: map score (0-100) to [-1, 1]
        igon_dir = 0.0
        if igon_signal and "score" in igon_signal:
            igon_dir = (igon_signal["score"] - 50) / 50  # 0-100 -> [-1, 1]
            # Boost signal if it passes all Igon criteria
            if igon_signal.get("passes_igon_screen"):
                igon_dir = max(igon_dir, 0.5)

        # Weighted combination. Model weights sum to 1; with an Igon signal
        # they are scaled down so the Igon share keeps the total at 1.
        models_combined = (
            self.weights["dnn"] * dnn_dir +
            self.weights["rnn"] * rnn_dir +
            self.weights["dql"] * dql_dir
        )
        if igon_signal:
            w_igon = self.weights["igon"]
            combined = (1 - w_igon) * models_combined + w_igon * igon_dir
        else:
            combined = models_combined

        # Determine signal
        if combined > 0.15:
            signal = "LONG"
        elif combined < -0.15:
            signal = "SHORT"
        else:
            signal = "NEUTRAL"

        confidence = min(abs(combined), 1.0)

        components = {
            "dnn": round(dnn_dir, 4),
            "rnn": round(rnn_dir, 4),
            "dql": round(dql_dir, 4),
        }
        if igon_signal:
            components["igon"] = round(igon_dir, 4)

        result = {
            "signal": signal,
            "confidence": round(confidence, 4),
            "combined_score": round(combined, 4),
            "components": components,
            "no_skill_warning": self.no_skill,
        }
        if self.no_skill:
            result["warning"] = (
                "No model beat its majority-class baseline out of sample by more than "
                "chance, so this signal is not evidence of predictive skill."
            )
        return result

    def update_weights(self, scores: Dict[str, Tuple[Optional[float], Optional[float],
                                                       Optional[float]]]):
        """
        Weight models by statistically significant out-of-sample skill.

        scores maps "dnn" / "rnn" / "dql" to (accuracy, baseline, p_value),
        accuracies in percent. None means the model made no scoreable calls.
        A model gets weight equal to its excess over baseline only if that
        excess is significant; otherwise it gets none.
        """
        self.skill = {}
        for k in ("dnn", "rnn", "dql"):
            acc, base, p = scores.get(k, (None, None, None))
            significant = p is not None and p < SIGNIFICANCE_LEVEL
            if acc is not None and base is not None and significant:
                self.skill[k] = max(acc - base, 0.0)
            else:
                self.skill[k] = 0.0
        total = sum(self.skill.values())
        self.no_skill = total <= 0
        for k in self.skill:
            self.weights[k] = (self.skill[k] / total) if total > 0 else 1 / 3

    def get_status(self) -> Dict:
        return {
            "name": self.name,
            "type": "Skill-weighted average (not a learned meta-model)",
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "skill_over_baseline": {k: round(v, 2) for k, v in self.skill.items()},
            "no_skill_warning": self.no_skill,
            "status": "active",
        }
