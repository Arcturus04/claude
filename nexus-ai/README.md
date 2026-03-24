# NEXUS AI - Artificial Financial Intelligence Platform

A full-stack AI-powered financial analysis platform inspired by BlackRock's Aladdin,
built from the principles in **"Artificial Intelligence in Finance"** by Yves Hilpisch (O'Reilly, 2021).

## Architecture

```
React Frontend (:3000)
  Dashboard | AI Models | Risk Engine | Sentiment | Trading Bot | Chat
       |
       | REST API (proxied via Vite)
       v
FastAPI Backend (:8000)
  +-- DNN Predictor (Ch.7)    -- Dense(64) x3 with dropout
  +-- RNN Sequencer (Ch.8)    -- LSTM(64) + LSTM(32)
  +-- DQL Agent (Ch.9)        -- Deep Q-Learning, experience replay
  +-- Ensemble Meta-Learner   -- Accuracy-weighted combination
  +-- Feature Engine           -- 14 features: RSI, MACD, ATR, sentiment...
  +-- Vectorized Backtester (Ch.10) -- Fast matrix-based strategy evaluation
  +-- Event Backtester (Ch.11)      -- Stop-loss, take-profit, trade log
  +-- Sentiment Scraper        -- Reddit + Twitter/X + 4chan /biz/
  +-- Claude Analyst           -- Natural language market interpretation
```

## Quick Start

### 1. Backend Setup

```bash
cd nexus-ai/backend
python -m venv venv
source venv/bin/activate        # Mac/Linux
# .\venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
cp .env.example .env            # Then edit with your API keys
python server.py
```

Backend runs at http://localhost:8000 (Swagger docs at /docs).

### 2. Frontend Setup

```bash
cd nexus-ai/frontend
npm install
npm run dev
```

Frontend runs at http://localhost:3000 with API proxy to backend.

### 3. API Keys (optional)

You can start with zero keys - ML models and market data work without any.

- **Anthropic (Claude):** https://console.anthropic.com - for AI analysis and chat
- **Reddit:** https://www.reddit.com/prefs/apps - for Reddit sentiment
- **Twitter/X:** https://developer.x.com - for Twitter sentiment
- **4chan:** No key needed - public API

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/health | System and model status |
| GET | /api/market/{symbol} | Market data from Yahoo Finance |
| POST | /api/train | Train all ML models |
| POST | /api/predict | Get model predictions |
| POST | /api/backtest | Run backtesting on trained models |
| GET | /api/risk | Risk metrics (VaR, CVaR, skewness, kurtosis) |
| POST | /api/sentiment/scrape | Scrape Reddit, Twitter, 4chan |
| GET | /api/sentiment | Get cached sentiment |
| POST | /api/analyze | Claude's full market analysis |
| POST | /api/chat | Chat with Claude about markets |

## Book Coverage

| Part | Chapters | Implementation |
|------|----------|---------------|
| I | 1-2 | AI concepts, neural network foundations |
| II | 3-6 | Normative finance, data-driven finance, feature engineering |
| III | 7-9 | DNN, RNN/LSTM, DQL Agent - all in models.py |
| IV | 10-12 | Vectorized + event-based backtesting in backtester.py |
| V | 13-14 | Risk analysis, non-normality diagnostics |

## Disclaimer

This is an educational project. NOT financial advice.
ML models typically achieve 55-65% directional accuracy.
Never trade real money based solely on these signals.
