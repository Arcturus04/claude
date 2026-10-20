"""
╔══════════════════════════════════════════════════════════════╗
║  NEXUS•AI — FastAPI Backend Server                          ║
║  Orchestrates ML models, sentiment scraping, and Claude AI  ║
╚══════════════════════════════════════════════════════════════╝

  Run with:  uvicorn server:app --reload --port 8000
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

import yfinance as yf
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Local modules
from models import FeatureEngine, DNNPredictor, RNNPredictor, DQLAgent, EnsemblePredictor
from sentiment_scraper import SentimentScraper
from claude_analyst import ClaudeAnalyst
from backtester import VectorizedBacktester, EventBacktester, generate_model_signals
from igon_screener import (
    score_stock, calculate_position_size, calculate_exit_schedule,
    portfolio_cleanup_check, screen_from_fundamentals, generate_igon_signals,
    IGON_METHOD_SUMMARY,
)

# ─── Configuration ────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger("nexus-ai")

# ─── Global State ─────────────────────────────────────────────
state = {
    "feature_engine": None,
    "dnn": None,
    "rnn": None,
    "dql": None,
    "ensemble": None,
    "scraper": None,
    "claude": None,
    "market_data": None,
    "features_df": None,
    "sentiment_cache": None,
    "last_sentiment_update": None,
    "trained": False,
    # Training metadata, so every metric and backtest can say what it was
    # measured on.
    "training_status": "idle",      # idle | running | complete | failed
    "training_error": None,
    "trained_symbol": None,
    "train_end_date": None,         # last date in the training split
    "test_period": None,            # (first, last) date of the held-out split
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup."""
    logger.info("═" * 60)
    logger.info("  NEXUS•AI — Artificial Financial Intelligence Platform")
    logger.info("═" * 60)

    # Fresh models start untrained: reset everything derived from training or
    # scraping, so a restarted app never reports a previous run's results.
    state.update({
        "market_data": None, "features_df": None,
        "sentiment_cache": None, "last_sentiment_update": None,
        "trained": False, "training_status": "idle", "training_error": None,
        "trained_symbol": None, "train_end_date": None, "test_period": None,
    })

    # Initialize components. Model input sizes come from the feature engine so
    # the two cannot drift apart.
    state["feature_engine"] = FeatureEngine(lags=5)
    n_features = len(state["feature_engine"].get_feature_columns())
    state["dnn"] = DNNPredictor(input_dim=n_features)
    state["rnn"] = RNNPredictor(n_features=n_features)
    state["dql"] = DQLAgent(state_dim=n_features)
    state["ensemble"] = EnsemblePredictor()
    state["scraper"] = SentimentScraper()
    state["claude"] = ClaudeAnalyst()

    logger.info("All components initialized. Ready to serve.")
    yield
    logger.info("NEXUS•AI shutting down.")


# ─── FastAPI App ──────────────────────────────────────────────
app = FastAPI(
    title="NEXUS•AI",
    description="Artificial Financial Intelligence Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ─────────────────────────────────
class TrainRequest(BaseModel):
    symbol: str = "SPY"
    period: str = "2y"        # yfinance period: 1y, 2y, 5y, max
    epochs_dnn: int = 50
    epochs_rnn: int = 30
    episodes_dql: int = 50


class ChatRequest(BaseModel):
    message: str


class PredictRequest(BaseModel):
    symbol: str = "SPY"


# ═══════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "name": "NEXUS•AI",
        "status": "operational",
        "trained": state["trained"],
        "timestamp": datetime.now().isoformat(),
    }


def _sentiment_has_posts() -> bool:
    """True only when the last scrape actually collected posts."""
    cache = state["sentiment_cache"]
    return bool(cache) and cache.get("total_posts", 0) > 0


def _iso_date(value) -> Optional[str]:
    return value.strftime("%Y-%m-%d") if value is not None else None


@app.get("/api/health")
async def health():
    test_period = state["test_period"]
    return {
        "status": "healthy",
        "models": {
            "dnn": state["dnn"].get_status() if state["dnn"] else None,
            "rnn": state["rnn"].get_status() if state["rnn"] else None,
            "dql": state["dql"].get_status() if state["dql"] else None,
        },
        "ensemble": state["ensemble"].get_status() if state["ensemble"] else None,
        "training": {
            "status": state["training_status"],
            "error": state["training_error"],
            "symbol": state["trained_symbol"],
            "train_end_date": _iso_date(state["train_end_date"]),
            "test_period": (
                [_iso_date(test_period[0]), _iso_date(test_period[1])] if test_period else None
            ),
            "features": (
                state["feature_engine"].get_feature_columns() if state["feature_engine"] else []
            ),
        },
        "sentiment_available": _sentiment_has_posts(),
        "claude_available": state["claude"].client is not None if state["claude"] else False,
    }


# ─── MARKET DATA ─────────────────────────────────────────────

@app.get("/api/market/{symbol}")
def get_market_data(symbol: str = "SPY", period: str = "6mo"):
    """Fetch market data from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)

        if df.empty:
            raise HTTPException(404, f"No data found for {symbol}")

        data = []
        for idx, row in df.iterrows():
            data.append({
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
                "close": round(row["Close"], 2),
                "volume": int(row["Volume"]),
            })

        return {
            "symbol": symbol,
            "period": period,
            "count": len(data),
            "data": data,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ─── MODEL TRAINING ──────────────────────────────────────────

@app.post("/api/train")
async def train_models(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Train all ML models on historical data.
    This runs in the background — poll /api/health to check status.
    """
    background_tasks.add_task(_train_models_task, req)
    return {
        "status": "training_started",
        "symbol": req.symbol,
        "message": "Training initiated. Poll /api/health to check progress.",
    }


def _train_models_task(req: TrainRequest):
    """
    Background task to train all models.

    A plain function, not a coroutine: FastAPI runs synchronous background
    tasks in a worker thread. As a coroutine, minutes of TensorFlow training
    ran on the event loop and froze every other request, including the
    /api/health polls the UI uses to watch training progress.

    Every model trains on the chronological training split only and is scored
    on the held-out split that follows it.
    """
    state["training_status"] = "running"
    state["training_error"] = None
    state["trained"] = False
    try:
        logger.info(f"Fetching {req.symbol} data for {req.period}...")

        # Fetch market data
        ticker = yf.Ticker(req.symbol)
        df = ticker.history(period=req.period)

        if df.empty:
            raise ValueError(f"No market data for {req.symbol} over {req.period}")

        state["market_data"] = df

        # Compute features
        fe = state["feature_engine"]
        features_df = fe.compute_features(df)
        state["features_df"] = features_df

        feature_cols = fe.get_feature_columns()
        X_train, X_test, y_train, y_test = fe.prepare_data(features_df)

        labelled = features_df.dropna(subset=["direction"])
        split = len(X_train)
        test_dates = labelled.index[split:]

        logger.info(f"Features computed: {len(features_df)} samples, {len(feature_cols)} features")
        logger.info(
            f"Train/Test split: {len(X_train)}/{len(X_test)} "
            f"(training data ends {fe.train_end_date.date()})"
        )

        # ─── Train DNN ───
        dnn = state["dnn"]
        dnn.train(X_train, y_train, X_test, y_test, epochs=req.epochs_dnn)

        # ─── Train RNN ───
        rnn = state["rnn"]
        rnn.train(X_train, y_train, X_test, y_test, epochs=req.epochs_rnn)

        # ─── Train DQL on the training split, score it on the test split ───
        dql = state["dql"]
        prices = labelled["Close"].values
        dql.train_on_data(X_train, prices[:split], episodes=req.episodes_dql)
        dql_eval = dql.evaluate(X_test, prices[split:], majority_up=float(np.mean(y_train)) >= 0.5)
        dql.accuracy = dql_eval["hit_rate"]
        dql.baseline_accuracy = dql_eval["baseline"]
        dql.p_value = dql_eval["p_value"]
        dql.coverage = dql_eval["coverage"]

        # ─── Weight the ensemble by significant out-of-sample skill ───
        state["ensemble"].update_weights({
            "dnn": (dnn.accuracy, dnn.baseline_accuracy, dnn.p_value),
            "rnn": (rnn.accuracy, rnn.baseline_accuracy, rnn.p_value),
            "dql": (dql.accuracy, dql.baseline_accuracy, dql.p_value),
        })

        state["trained_symbol"] = req.symbol
        state["train_end_date"] = fe.train_end_date
        state["test_period"] = (test_dates[0], test_dates[-1]) if len(test_dates) else None
        state["trained"] = True
        state["training_status"] = "complete"

        def pct(value):
            return f"{value:.2f}%" if value is not None else "n/a"

        logger.info("═" * 60)
        logger.info("  Training complete - out-of-sample results")
        logger.info(f"  DNN accuracy:     {pct(dnn.accuracy)} (baseline {pct(dnn.baseline_accuracy)})")
        logger.info(f"  RNN accuracy:     {pct(rnn.accuracy)} (baseline {pct(rnn.baseline_accuracy)})")
        logger.info(
            f"  DQL hit rate:     {pct(dql.accuracy)} (baseline {pct(dql.baseline_accuracy)}, "
            f"calls on {pct(dql.coverage)} of sessions)"
        )
        logger.info(f"  Ensemble weights: {state['ensemble'].weights}")
        if state["ensemble"].no_skill:
            logger.info("  No model beat its baseline; ensemble signals carry a warning.")
        logger.info("═" * 60)

    except Exception as e:
        state["training_status"] = "failed"
        state["training_error"] = str(e)
        logger.error(f"Training failed: {e}", exc_info=True)


# ─── PREDICTIONS ──────────────────────────────────────────────

@app.post("/api/predict")
def predict(req: PredictRequest):
    """Get next-session predictions from all trained models."""
    if not state["trained"]:
        raise HTTPException(400, "Models not trained yet. Call POST /api/train first.")

    try:
        fe = state["feature_engine"]

        # One year of history: the 50-day average consumes the first 49 rows
        # and the LSTM needs a 20-row window after that. Three months left
        # about 14 rows, so the LSTM never had a full window.
        ticker = yf.Ticker(req.symbol)
        df = ticker.history(period="1y")
        if df.empty:
            raise HTTPException(404, f"No data for {req.symbol}")

        features_df = fe.compute_features(df)
        needed = state["rnn"].sequence_length
        if len(features_df) < needed:
            raise HTTPException(
                400,
                f"Only {len(features_df)} sessions of usable history for {req.symbol}; "
                f"the LSTM needs {needed} after indicator warm-up.",
            )
        feature_cols = fe.get_feature_columns()
        X = fe.scaler.transform(features_df[feature_cols].values)

        # Get individual model predictions
        latest = X[-1]
        dnn_dir, dnn_conf = state["dnn"].predict(latest)
        rnn_dir, rnn_conf = state["rnn"].predict(X)
        dql_signal = state["dql"].get_signal(latest)

        # Ensemble prediction
        ensemble_result = state["ensemble"].predict(
            (dnn_dir, dnn_conf),
            (rnn_dir, rnn_conf),
            dql_signal,
        )

        return {
            "symbol": req.symbol,
            "timestamp": datetime.now().isoformat(),
            "current_price": round(float(df["Close"].iloc[-1]), 2),
            "models": {
                "dnn": {
                    "direction": "LONG" if dnn_dir == 1 else "SHORT",
                    "confidence": round(dnn_conf, 4),
                    "accuracy": state["dnn"].accuracy,
                },
                "rnn": {
                    "direction": "LONG" if rnn_dir == 1 else "SHORT",
                    "confidence": round(rnn_conf, 4),
                    "accuracy": state["rnn"].accuracy,
                },
                "dql": {
                    **dql_signal,
                    "accuracy": state["dql"].accuracy,
                },
            },
            "ensemble": ensemble_result,
            "as_of": features_df.index[-1].strftime("%Y-%m-%d"),
            "predicts": "direction of the next session after as_of",
            "trained_symbol": state["trained_symbol"],
            "symbol_matches_training": req.symbol.upper() == (state["trained_symbol"] or "").upper(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ─── SENTIMENT ────────────────────────────────────────────────

@app.get("/api/sentiment")
async def get_sentiment():
    """Get cached sentiment data."""
    if state["sentiment_cache"]:
        return state["sentiment_cache"]
    return {"status": "no_data", "message": "Run POST /api/sentiment/scrape first"}


@app.post("/api/sentiment/scrape")
async def scrape_sentiment(background_tasks: BackgroundTasks):
    """Trigger a new sentiment scrape from all sources."""
    background_tasks.add_task(_scrape_sentiment_task)
    return {"status": "scraping_started", "message": "Sentiment scrape initiated."}


def _scrape_sentiment_task():
    """
    Background sentiment scraping, in a worker thread: the scrape makes many
    network calls and pauses between 4chan requests, which as a coroutine
    blocked every other request until it finished.
    """
    try:
        result = state["scraper"].scrape_all(limit_per_source=50)
        state["sentiment_cache"] = result
        state["last_sentiment_update"] = datetime.now().isoformat()
        logger.info(f"Sentiment scrape complete: {result['total_posts']} posts analyzed")
    except Exception as e:
        logger.error(f"Sentiment scrape failed: {e}", exc_info=True)


# ─── CLAUDE ANALYSIS ─────────────────────────────────────────

def _claude_response(result, text_key: str) -> Dict:
    """
    Map an analyst result to HTTP.

    A configured client whose call failed is an upstream error (502), not a
    successful answer. Without a key the documented offline mode is returned
    with llm_used=false so callers can tell it from Claude's output.
    """
    if result.error and state["claude"].configured and result.stop_reason != "refusal":
        raise HTTPException(502, result.error)
    return {
        text_key: result.text or result.error or "",
        "llm_used": result.llm_used,
        "model": result.model,
        "error": result.error,
        "stop_reason": result.stop_reason,
        "usage": result.usage,
    }


@app.post("/api/analyze")
def analyze():
    """
    Get Claude's market analysis.

    A plain function so FastAPI runs it in a worker thread: the Claude call
    can take tens of seconds and must not block other requests.
    """
    if not state["claude"]:
        raise HTTPException(500, "Claude analyst not initialized")

    market_data = {}
    if state["market_data"] is not None:
        last_5 = state["market_data"].tail(5)
        market_data = {
            "latest_close": round(float(last_5["Close"].iloc[-1]), 2),
            "5d_return": round(
                float((last_5["Close"].iloc[-1] / last_5["Close"].iloc[0] - 1) * 100), 2
            ),
        }

    model_signals = {}
    if state["trained"]:
        model_signals = {
            "dnn": state["dnn"].get_status(),
            "rnn": state["rnn"].get_status(),
            "dql": state["dql"].get_status(),
            "ensemble": state["ensemble"].get_status(),
            "trained_symbol": state["trained_symbol"],
            "test_period": [_iso_date(d) for d in state["test_period"]] if state["test_period"] else None,
        }

    sentiment_data = state["sentiment_cache"] or {}

    risk_metrics = {
        "note": "Risk metrics from model training data",
    }
    if state["features_df"] is not None:
        returns = state["features_df"]["log_return"]
        risk_metrics.update({
            "volatility_20d": round(float(returns.tail(20).std() * np.sqrt(252) * 100), 2),
            "skewness": round(float(returns.skew()), 4),
            "kurtosis": round(float(returns.kurtosis()), 4),
            "var_95": round(float(np.percentile(returns, 5) * 100), 2),
            "max_drawdown": round(float(
                ((state["features_df"]["Close"] /
                  state["features_df"]["Close"].cummax()) - 1).min() * 100
            ), 2),
        })

    result = state["claude"].analyze(
        market_data, model_signals, sentiment_data, risk_metrics
    )

    return {
        "timestamp": datetime.now().isoformat(),
        **_claude_response(result, "analysis"),
        "data_used": {
            "market_data": bool(market_data),
            "model_signals": bool(model_signals),
            "sentiment": _sentiment_has_posts(),
            "risk_metrics": state["features_df"] is not None,
        },
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Chat with Claude about market conditions (runs in a worker thread)."""
    if not req.message.strip():
        raise HTTPException(400, "message must not be empty")
    context = {
        "trained": state["trained"],
        "trained_symbol": state["trained_symbol"],
        "models": {
            "dnn": state["dnn"].get_status() if state["dnn"] else None,
            "rnn": state["rnn"].get_status() if state["rnn"] else None,
            "dql": state["dql"].get_status() if state["dql"] else None,
        },
        "ensemble": state["ensemble"].get_status() if state["ensemble"] else None,
        "sentiment": (
            {
                "overall": state["sentiment_cache"].get("overall_sentiment"),
                "label": state["sentiment_cache"].get("sentiment_label"),
                "posts": state["sentiment_cache"].get("total_posts"),
            }
            if _sentiment_has_posts() else "no posts collected"
        ),
    }

    result = state["claude"].chat(req.message, context)
    return _claude_response(result, "response")


# ─── RISK METRICS ─────────────────────────────────────────────

@app.get("/api/risk")
async def get_risk_metrics():
    """Calculate current risk metrics."""
    if state["features_df"] is None:
        raise HTTPException(400, "No data available. Train models first.")

    df = state["features_df"]
    returns = df["log_return"]

    from scipy import stats

    return {
        "var_95": round(float(np.percentile(returns, 5) * 100), 4),
        "cvar_95": round(float(returns[returns <= np.percentile(returns, 5)].mean() * 100), 4),
        "volatility_annual": round(float(returns.std() * np.sqrt(252) * 100), 2),
        "skewness": round(float(returns.skew()), 4),
        "kurtosis": round(float(returns.kurtosis()), 4),
        "max_drawdown": round(float(((df["Close"] / df["Close"].cummax()) - 1).min() * 100), 2),
        "atr_14d": round(float(df["atr"].iloc[-1] * 100), 4) if "atr" in df.columns else None,
        "normality_tests": {
            "skew_test_pvalue": round(float(stats.skewtest(returns)[1]), 6),
            "kurtosis_test_pvalue": round(float(stats.kurtosistest(returns)[1]), 6),
            "normal_test_pvalue": round(float(stats.normaltest(returns)[1]), 6),
        },
    }


# ─── BACKTESTING (Part IV - Chapters 10-12) ──────────────────

class BacktestRequest(BaseModel):
    symbol: str = "SPY"
    period: str = "2y"
    model: str = "dnn"          # dnn, rnn, or ensemble
    stop_loss: float = 0.0175   # 1.75%
    take_profit: float = 0.0185 # 1.85%
    initial_capital: float = 10000.0
    mode: str = "vectorized"    # vectorized or event


MIN_BACKTEST_BARS = 20


@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    """
    Backtest a trained model's signals on data it was not trained on.

    Signals are generated for the whole fetched period (the LSTM needs earlier
    rows as context), but only sessions after the training cutoff are traded.
    Backtesting the training period instead would report the model's fit to
    data it has already seen, which says nothing about how it would trade.
    Supports vectorized (fast) and event-based (with risk management) modes.
    """
    if not state["trained"]:
        raise HTTPException(400, "Models not trained. Call POST /api/train first.")
    if req.model not in ("dnn", "rnn", "ensemble"):
        raise HTTPException(400, "model must be one of: dnn, rnn, ensemble")
    if req.mode not in ("vectorized", "event"):
        raise HTTPException(400, "mode must be one of: vectorized, event")

    try:
        fe = state["feature_engine"]

        # Fetch data
        ticker = yf.Ticker(req.symbol)
        df = ticker.history(period=req.period)
        if df.empty:
            raise HTTPException(404, f"No data for {req.symbol}")

        features_df = fe.compute_features(df)
        if features_df.empty:
            raise HTTPException(
                400,
                f"Period {req.period} is too short to compute the indicators "
                f"(the 50-day average alone needs 50 sessions).",
            )
        feature_cols = fe.get_feature_columns()
        X = fe.scaler.transform(features_df[feature_cols].values)

        # Signal on row r is a forecast for the session after r; the
        # backtester shifts it forward one bar before applying it.
        if req.model == "dnn":
            signals = generate_model_signals(state["dnn"].model, X)
        elif req.model == "rnn":
            probs = state["rnn"].predict_proba_series(X)
            signals = np.where(probs > 0.55, 1, np.where(probs < 0.45, -1, 0))
        else:
            # Ensemble - combine DNN + DQL signals, one greedy action per row
            dnn_signals = generate_model_signals(state["dnn"].model, X)
            actions = state["dql"].greedy_actions(X)
            dql_signals = np.where(actions == 1, 1, np.where(actions == 2, -1, 0))
            signals = np.sign(dnn_signals * 0.6 + dql_signals * 0.4)

        # Keep only sessions after the training cutoff.
        oos = features_df.index > state["train_end_date"]
        n_oos = int(oos.sum())
        if n_oos < MIN_BACKTEST_BARS:
            raise HTTPException(
                400,
                f"Only {n_oos} sessions after the training cutoff "
                f"({_iso_date(state['train_end_date'])}) in this period; need at least "
                f"{MIN_BACKTEST_BARS}. Use a longer period, or retrain on an earlier one.",
            )
        prices = features_df["Close"].values[oos]
        dates = [d.strftime("%Y-%m-%d") for d in features_df.index[oos]]
        signals = np.asarray(signals)[oos]

        # Run backtest
        if req.mode == "event":
            bt = EventBacktester(
                initial_capital=req.initial_capital,
                stop_loss=req.stop_loss,
                take_profit=req.take_profit,
            )
        else:
            bt = VectorizedBacktester(initial_capital=req.initial_capital)

        result = bt.run(prices, signals.astype(int), dates)
        result["model"] = req.model
        result["symbol"] = req.symbol
        result["mode"] = req.mode
        result["out_of_sample"] = {
            "trained_symbol": state["trained_symbol"],
            "train_end_date": _iso_date(state["train_end_date"]),
            "start": dates[0],
            "end": dates[-1],
            "sessions": n_oos,
            "note": (
                "Only sessions after the training cutoff are traded. "
                "Past performance does not predict future results."
            ),
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ─── IGON VALUE SCREENER ─────────────────────────────────────

class IgonScreenRequest(BaseModel):
    symbol: str = "TTE.PA"  # Default to Total Energies (Paris)


class IgonScoreRequest(BaseModel):
    pe: float = 8.0
    roe: float = 15.0
    revenue_growth: float = 5.0
    wcr_pct: Optional[float] = None


class IgonPositionRequest(BaseModel):
    portfolio_value: float = 100000.0
    stock_price: float = 50.0
    avg_daily_volume: float = 500000.0
    current_lines: int = 20


class IgonExitRequest(BaseModel):
    pe: float = 16.0
    position_pct: float = 3.0


@app.get("/api/igon/method")
async def igon_method():
    """Get a summary of William Igon's investment methodology."""
    return IGON_METHOD_SUMMARY


@app.post("/api/igon/screen")
async def igon_screen(req: IgonScreenRequest):
    """
    Screen a stock using Igon's value criteria.
    Fetches fundamentals from Yahoo Finance and runs the full analysis.
    """
    try:
        ticker = yf.Ticker(req.symbol)
        info = ticker.info

        if not info or not info.get("shortName"):
            raise HTTPException(404, f"No fundamental data found for {req.symbol}")

        result = screen_from_fundamentals(info)
        result["symbol"] = req.symbol
        result["timestamp"] = datetime.now().isoformat()

        # Add current price for context
        hist = ticker.history(period="5d")
        if not hist.empty:
            result["current_price"] = round(float(hist["Close"].iloc[-1]), 2)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Screening failed: {str(e)}")


@app.post("/api/igon/score")
async def igon_score(req: IgonScoreRequest):
    """Score a stock using manual PE, ROE, and revenue growth inputs."""
    return score_stock(req.pe, req.roe, req.revenue_growth, req.wcr_pct)


@app.post("/api/igon/position-size")
async def igon_position_size(req: IgonPositionRequest):
    """Calculate position sizing using Igon's rules."""
    return calculate_position_size(
        req.portfolio_value, req.stock_price,
        req.avg_daily_volume, req.current_lines,
    )


@app.post("/api/igon/exit-schedule")
async def igon_exit_schedule(req: IgonExitRequest):
    """Calculate profit-taking schedule based on PE expansion."""
    return calculate_exit_schedule(req.pe, req.position_pct)


@app.post("/api/igon/batch-screen")
async def igon_batch_screen(symbols: List[str] = ["TTE.PA", "SAF.PA", "PUB.PA", "VIE.PA"]):
    """Screen multiple stocks at once using Igon's criteria."""
    results = []
    for sym in symbols[:20]:  # Cap at 20 to avoid rate limits
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info
            if info and info.get("shortName"):
                result = screen_from_fundamentals(info)
                result["symbol"] = sym
                results.append(result)
        except Exception as e:
            logger.warning(f"Failed to screen {sym}: {e}")
            results.append({"symbol": sym, "error": str(e)})

    # Sort by score descending
    screened = [r for r in results if "score" in r]
    screened.sort(key=lambda r: r["score"], reverse=True)
    errors = [r for r in results if "error" in r]

    return {
        "timestamp": datetime.now().isoformat(),
        "total_screened": len(screened),
        "passing_igon_screen": len([r for r in screened if r.get("passes_igon_screen")]),
        "results": screened + errors,
    }


# ═══════════════════════════════════════════════════════════════
#  RUN SERVER
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    # Local only by default. The API has no authentication, and /api/chat and
    # /api/analyze spend Anthropic credits, so it should not be reachable from
    # other machines unless HOST is set deliberately.
    host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run("server:app", host=host, port=port, reload=True)
