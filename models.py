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
from typing import Dict, List, Tuple, Optional
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

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Transforms raw price data + sentiment into ML-ready features.
    Computes technical indicators and merges social sentiment scores.
    """

    def __init__(self, lags: int = 5):
        self.lags = lags
        self.scaler = StandardScaler()
        self.fitted = False

    def compute_features(self, df: pd.DataFrame, sentiment: float = 0.0) -> pd.DataFrame:
        """
        Generate features from OHLCV data:
        - Log returns (multiple lags)
        - Moving averages (SMA, EMA)
        - Volatility (rolling std)
        - RSI, MACD signals
        - Volume momentum
        - Social sentiment score
        """
        data = df.copy()

        # Log returns
        data["log_return"] = np.log(data["Close"] / data["Close"].shift(1))

        # Lagged returns as features
        for lag in range(1, self.lags + 1):
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

        # Social sentiment feature
        data["sentiment"] = sentiment

        # Direction label (what we're predicting)
        data["direction"] = np.where(data["log_return"] > 0, 1, 0)

        # Drop NaN rows
        data.dropna(inplace=True)

        return data

    def get_feature_columns(self) -> List[str]:
        """Return the feature column names used for training."""
        cols = [f"return_lag_{i}" for i in range(1, self.lags + 1)]
        cols += [
            "sma_20", "sma_50", "ema_12",
            "volatility_20", "rsi_normalized", "macd",
            "volume_ratio", "atr", "sentiment",
        ]
        return cols

    def prepare_data(self, df: pd.DataFrame, train_ratio: float = 0.8):
        """Split and scale data for training."""
        feature_cols = self.get_feature_columns()
        X = df[feature_cols].values
        y = df["direction"].values

        split = int(len(X) * train_ratio)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Fit scaler on training data only
        self.scaler.fit(X_train)
        self.fitted = True

        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test


class DNNPredictor:
    """
    Dense Neural Network for market direction prediction.
    Architecture from Hilpisch Chapter 7.
    """

    def __init__(self, input_dim: int = 14, hidden_units: int = 64,
                 dropout_rate: float = 0.3, learning_rate: float = 0.001):
        self.model = self._build_model(input_dim, hidden_units, dropout_rate, learning_rate)
        self.history = None
        self.accuracy = 0.0
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
        logger.info(f"DNN accuracy: {self.accuracy:.2f}%")
        return self.accuracy

    def predict(self, X) -> Tuple[int, float]:
        """Predict direction and confidence."""
        prob = self.model.predict(X.reshape(1, -1), verbose=0)[0][0]
        direction = 1 if prob > 0.5 else 0
        confidence = prob if direction == 1 else 1 - prob
        return direction, float(confidence)

    def get_status(self) -> Dict:
        return {
            "name": self.name,
            "type": "Dense Neural Network",
            "accuracy": round(self.accuracy, 2),
            "status": "active" if self.accuracy > 0 else "untrained",
            "architecture": "Dense(64) → Dense(64) → Dense(32) → Dense(1)",
        }


class RNNPredictor:
    """
    Recurrent Neural Network (LSTM) for sequence-based prediction.
    Architecture from Hilpisch Chapter 8.
    """

    def __init__(self, sequence_length: int = 20, n_features: int = 14,
                 hidden_units: int = 64, learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model(n_features, hidden_units, learning_rate)
        self.accuracy = 0.0
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
        """Create sliding window sequences for LSTM input."""
        Xs, ys = [], []
        for i in range(len(X) - self.sequence_length):
            Xs.append(X[i:i + self.sequence_length])
            ys.append(y[i + self.sequence_length])
        return np.array(Xs), np.array(ys)

    def train(self, X_train, y_train, X_test, y_test,
              epochs: int = 30, batch_size: int = 32):
        """Train the LSTM model."""
        logger.info(f"Training RNN/LSTM: {epochs} epochs")

        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)
        X_test_seq, y_test_seq = self._create_sequences(X_test, y_test)

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
        logger.info(f"RNN accuracy: {self.accuracy:.2f}%")
        return self.accuracy

    def predict(self, X_sequence) -> Tuple[int, float]:
        """Predict from a sequence of features."""
        if len(X_sequence) < self.sequence_length:
            return 0, 0.5
        seq = X_sequence[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        prob = self.model.predict(seq, verbose=0)[0][0]
        direction = 1 if prob > 0.5 else 0
        confidence = prob if direction == 1 else 1 - prob
        return direction, float(confidence)

    def get_status(self) -> Dict:
        return {
            "name": self.name,
            "type": "LSTM Recurrent Neural Net",
            "accuracy": round(self.accuracy, 2),
            "status": "active" if self.accuracy > 0 else "untrained",
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

    def __init__(self, state_dim: int = 14, gamma: float = 0.95,
                 hidden_units: int = 64, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.gamma = gamma          # Discount factor for delayed rewards
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.01     # Minimum exploration
        self.epsilon_decay = 0.995  # Decay per episode
        self.batch_size = 32
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.model = self._build_model(state_dim, hidden_units, learning_rate)
        self.accuracy = 0.0
        self.total_reward = 0.0
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

    def act(self, state) -> int:
        """Choose action using epsilon-greedy policy."""
        if random.random() <= self.epsilon:
            return random.randint(0, 2)  # Explore
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        return int(np.argmax(q_values))  # Exploit

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train on a batch of experiences from replay buffer."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                # Q-learning update: Q(s,a) = r + γ * max Q(s', a')
                target = reward + self.gamma * np.max(
                    self.model.predict(next_state.reshape(1, -1), verbose=0)[0]
                )

            target_vec = self.model.predict(state.reshape(1, -1), verbose=0)[0]
            target_vec[action] = target
            self.model.fit(state.reshape(1, -1), target_vec.reshape(1, -1),
                           epochs=1, verbose=0)

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_on_data(self, features: np.ndarray, prices: np.ndarray,
                      episodes: int = 50):
        """
        Train the DQL agent on historical data.
        Simulates trading and learns from rewards.
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

        self.total_reward = best_reward
        # Approximate accuracy from reward sign
        positive = len([r for r in rewards_history if r > 0])
        self.accuracy = (positive / len(rewards_history)) * 100 if rewards_history else 0

        logger.info(f"DQL training complete. Best reward: {best_reward:.2f}")
        return self.accuracy

    def get_signal(self, state) -> Dict:
        """Get trading signal from trained agent."""
        action = self.act(state)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        confidence = float(np.max(q_values) / (np.sum(np.abs(q_values)) + 1e-8))

        signals = {0: "HOLD", 1: "BUY", 2: "SELL"}
        return {
            "action": signals[action],
            "confidence": round(min(confidence, 1.0), 4),
            "q_values": {signals[i]: round(float(q_values[i]), 4) for i in range(3)},
        }

    def get_status(self) -> Dict:
        return {
            "name": self.name,
            "type": "Deep Q-Learning",
            "accuracy": round(self.accuracy, 2),
            "status": "active" if self.accuracy > 0 else "training",
            "epsilon": round(self.epsilon, 4),
            "memory_size": len(self.memory),
            "total_reward": round(self.total_reward, 2),
            "architecture": "Dense(64) → Dense(64) → Dense(3) [Q-values]",
        }


class EnsemblePredictor:
    """
    Meta-learner that combines signals from DNN, RNN, and DQL.
    Weighted by individual model accuracy.
    """

    def __init__(self):
        self.name = "Ensemble α"
        self.weights = {"dnn": 0.35, "rnn": 0.35, "dql": 0.30}

    def predict(self, dnn_signal: Tuple, rnn_signal: Tuple,
                dql_signal: Dict) -> Dict:
        """
        Combine model outputs into a unified signal.
        """
        # Convert all to direction scores [-1, 1]
        dnn_dir = (dnn_signal[0] * 2 - 1) * dnn_signal[1]   # direction * confidence
        rnn_dir = (rnn_signal[0] * 2 - 1) * rnn_signal[1]

        dql_action = dql_signal["action"]
        dql_dir = (1 if dql_action == "BUY" else -1 if dql_action == "SELL" else 0)
        dql_dir *= dql_signal["confidence"]

        # Weighted combination
        combined = (
            self.weights["dnn"] * dnn_dir +
            self.weights["rnn"] * rnn_dir +
            self.weights["dql"] * dql_dir
        )

        # Determine signal
        if combined > 0.15:
            signal = "LONG"
        elif combined < -0.15:
            signal = "SHORT"
        else:
            signal = "NEUTRAL"

        confidence = min(abs(combined), 1.0)

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "combined_score": round(combined, 4),
            "components": {
                "dnn": round(dnn_dir, 4),
                "rnn": round(rnn_dir, 4),
                "dql": round(dql_dir, 4),
            },
        }

    def update_weights(self, dnn_acc: float, rnn_acc: float, dql_acc: float):
        """Adjust weights based on model performance."""
        total = dnn_acc + rnn_acc + dql_acc
        if total > 0:
            self.weights["dnn"] = dnn_acc / total
            self.weights["rnn"] = rnn_acc / total
            self.weights["dql"] = dql_acc / total

    def get_status(self) -> Dict:
        return {
            "name": self.name,
            "type": "Meta-Learner Ensemble",
            "weights": self.weights,
            "status": "active",
        }
