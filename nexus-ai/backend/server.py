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
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup."""
    logger.info("═" * 60)
    logger.info("  NEXUS•AI — Artificial Financial Intelligence Platform")
    logger.info("═" * 60)

    # Initialize components
    state["feature_engine"] = FeatureEngine(lags=5)
    state["dnn"] = DNNPredictor(input_dim=14)
    state["rnn"] = RNNPredictor(n_features=14)
    state["dql"] = DQLAgent(state_dim=14)
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


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "models": {
            "dnn": state["dnn"].get_status() if state["dnn"] else None,
            "rnn": state["rnn"].get_status() if state["rnn"] else None,
            "dql": state["dql"].get_status() if state["dql"] else None,
        },
        "sentiment_available": state["sentiment_cache"] is not None,
        "claude_available": state["claude"].client is not None if state["claude"] else False,
    }


# ─── MARKET DATA ─────────────────────────────────────────────

@app.get("/api/market/{symbol}")
async def get_market_data(symbol: str = "SPY", period: str = "6mo"):
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


async def _train_models_task(req: TrainRequest):
    """Background task to train all models."""
    try:
        logger.info(f"Fetching {req.symbol} data for {req.period}...")

        # Fetch market data
        ticker = yf.Ticker(req.symbol)
        df = ticker.history(period=req.period)

        if df.empty:
            logger.error(f"No data for {req.symbol}")
            return

        state["market_data"] = df

        # Get current sentiment for feature injection
        sentiment_score = 0.0
        if state["sentiment_cache"]:
            sentiment_score = state["sentiment_cache"].get("overall_sentiment", 0.0)

        # Compute features
        fe = state["feature_engine"]
        features_df = fe.compute_features(df, sentiment=sentiment_score)
        state["features_df"] = features_df

        feature_cols = fe.get_feature_columns()
        X_train, X_test, y_train, y_test = fe.prepare_data(features_df)

        logger.info(f"Features computed: {len(features_df)} samples, {len(feature_cols)} features")
        logger.info(f"Train/Test split: {len(X_train)}/{len(X_test)}")

        # ─── Train DNN ───
        dnn_acc = state["dnn"].train(X_train, y_train, X_test, y_test,
                                      epochs=req.epochs_dnn)

        # ─── Train RNN ───
        rnn_acc = state["rnn"].train(X_train, y_train, X_test, y_test,
                                      epochs=req.epochs_rnn)

        # ─── Train DQL ───
        prices = features_df["Close"].values
        dql_acc = state["dql"].train_on_data(
            fe.scaler.transform(features_df[feature_cols].values),
            prices,
            episodes=req.episodes_dql,
        )

        # ─── Update ensemble weights ───
        state["ensemble"].update_weights(dnn_acc, rnn_acc, dql_acc)
        state["trained"] = True

        logger.info("═" * 60)
        logger.info(f"  Training Complete!")
        logger.info(f"  DNN Accuracy:  {dnn_acc:.2f}%")
        logger.info(f"  RNN Accuracy:  {rnn_acc:.2f}%")
        logger.info(f"  DQL Win Rate:  {dql_acc:.2f}%")
        logger.info(f"  Ensemble Weights: {state['ensemble'].weights}")
        logger.info("═" * 60)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)


# ─── PREDICTIONS ──────────────────────────────────────────────

@app.post("/api/predict")
async def predict(req: PredictRequest):
    """Get predictions from all trained models."""
    if not state["trained"]:
        raise HTTPException(400, "Models not trained yet. Call POST /api/train first.")

    try:
        fe = state["feature_engine"]

        # Get latest data
        ticker = yf.Ticker(req.symbol)
        df = ticker.history(period="3mo")

        sentiment_score = 0.0
        if state["sentiment_cache"]:
            sentiment_score = state["sentiment_cache"].get("overall_sentiment", 0.0)

        features_df = fe.compute_features(df, sentiment=sentiment_score)
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
            "sentiment_injected": sentiment_score,
        }

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


async def _scrape_sentiment_task():
    """Background sentiment scraping."""
    try:
        result = state["scraper"].scrape_all(limit_per_source=50)
        state["sentiment_cache"] = result
        state["last_sentiment_update"] = datetime.now().isoformat()
        logger.info(f"Sentiment scrape complete: {result['total_posts']} posts analyzed")
    except Exception as e:
        logger.error(f"Sentiment scrape failed: {e}", exc_info=True)


# ─── CLAUDE ANALYSIS ─────────────────────────────────────────

@app.post("/api/analyze")
async def analyze():
    """Get Claude's comprehensive market analysis."""
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
        # Re-use latest prediction logic
        model_signals = {
            "dnn": state["dnn"].get_status(),
            "rnn": state["rnn"].get_status(),
            "dql": state["dql"].get_status(),
            "ensemble": {"weights": state["ensemble"].weights},
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

    analysis = state["claude"].analyze(
        market_data, model_signals, sentiment_data, risk_metrics
    )

    return {
        "timestamp": datetime.now().isoformat(),
        "analysis": analysis,
        "data_used": {
            "market_data": bool(market_data),
            "model_signals": bool(model_signals),
            "sentiment": bool(sentiment_data),
            "risk_metrics": bool(risk_metrics),
        },
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Chat with Claude about market conditions."""
    context = {
        "trained": state["trained"],
        "models": {
            "dnn": state["dnn"].get_status() if state["dnn"] else None,
            "rnn": state["rnn"].get_status() if state["rnn"] else None,
            "dql": state["dql"].get_status() if state["dql"] else None,
        },
        "sentiment": state["sentiment_cache"].get("overall_sentiment") if state["sentiment_cache"] else None,
    }

    response = state["claude"].chat(req.message, context)
    return {"response": response}


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


@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest):
    """
    Run backtesting on trained model signals.
    Supports vectorized (fast) and event-based (with risk management) modes.
    """
    if not state["trained"]:
        raise HTTPException(400, "Models not trained. Call POST /api/train first.")

    try:
        fe = state["feature_engine"]

        # Fetch data
        ticker = yf.Ticker(req.symbol)
        df = ticker.history(period=req.period)
        if df.empty:
            raise HTTPException(404, f"No data for {req.symbol}")

        sentiment_score = 0.0
        if state["sentiment_cache"]:
            sentiment_score = state["sentiment_cache"].get("overall_sentiment", 0.0)

        features_df = fe.compute_features(df, sentiment=sentiment_score)
        feature_cols = fe.get_feature_columns()
        X = fe.scaler.transform(features_df[feature_cols].values)
        prices = features_df["Close"].values
        dates = [d.strftime("%Y-%m-%d") for d in features_df.index]

        # Generate signals from selected model
        if req.model == "dnn":
            signals = generate_model_signals(state["dnn"].model, X)
        elif req.model == "rnn":
            # RNN needs sequences - generate per-step predictions
            signals = np.zeros(len(X))
            seq_len = state["rnn"].sequence_length
            for i in range(seq_len, len(X)):
                seq = X[i - seq_len:i].reshape(1, seq_len, -1)
                prob = state["rnn"].model.predict(seq, verbose=0)[0][0]
                signals[i] = 1 if prob > 0.55 else (-1 if prob < 0.45 else 0)
        else:
            # Ensemble - combine DNN + DQL signals
            dnn_signals = generate_model_signals(state["dnn"].model, X)
            dql_signals = np.array([
                1 if state["dql"].act(x) == 1 else
                (-1 if state["dql"].act(x) == 2 else 0)
                for x in X
            ])
            signals = np.sign(dnn_signals * 0.6 + dql_signals * 0.4)

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

        return result

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
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
