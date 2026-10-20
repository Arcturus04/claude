# NEXUS•AI - Financial Intelligence Platform

ML trading models, social sentiment and Claude analysis, built on the methods in
*Artificial Intelligence in Finance* by Yves Hilpisch (O'Reilly, 2021).

**Educational project. Not financial advice.**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                REACT + VITE FRONTEND (:3000)                 │
│  Dashboard │ AI Models │ Risk │ Sentiment │ Trading Bot │ Chat│
└──────────────────────────┬──────────────────────────────────┘
                           │ REST API (proxied by Vite)
┌──────────────────────────▼──────────────────────────────────┐
│                 FASTAPI BACKEND (:8000)                      │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐   │
│  │   DNN    │  │   LSTM   │  │   DQL    │  │  Ensemble  │   │
│  │ Predictor│  │ Sequencer│  │  Agent   │  │ (weighted) │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘   │
│       └─────────────┴─────────────┴──────────────┘          │
│                          │                                   │
│  ┌───────────────────────▼───────────┐  ┌────────────────┐  │
│  │ FEATURE ENGINE (13 price features)│  │  BACKTESTERS   │  │
│  └───────────────────────┬───────────┘  └────────────────┘  │
│  ┌───────────────────────▼───────────┐                      │
│  │ MARKET DATA (Yahoo Finance)       │                      │
│  └───────────────────────────────────┘                      │
│  ┌───────────────────────────────────┐                      │
│  │ SENTIMENT SCRAPER                 │──┐                   │
│  │ Reddit + X + 4chan, VADER scoring │  │                   │
│  └───────────────────────────────────┘  ▼                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ CLAUDE ANALYST: model results + sentiment + risk      │  │
│  │                  → written analysis and chat          │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- **Python 3.10+** from https://www.python.org/downloads/ (tick "Add to PATH")
- **Node.js 18+** from https://nodejs.org/

### 1. Backend

```bash
cd nexus-ai/backend
python -m venv venv

# Activate it:
.\venv\Scripts\Activate.ps1      # Windows PowerShell
.\venv\Scripts\activate.bat      # Windows CMD
source venv/bin/activate         # Mac/Linux

pip install -r requirements.txt
copy .env.example .env           # Windows  (Mac/Linux: cp .env.example .env)
python server.py
```

The backend runs at http://127.0.0.1:8000, with interactive API docs at
http://127.0.0.1:8000/docs.

### 2. Frontend

In a second terminal:

```bash
cd nexus-ai/frontend
npm install
npm run dev
```

Open http://localhost:3000.

### 3. API keys (all optional)

Edit `nexus-ai/backend/.env`:

| Key | Needed for | Cost |
|-----|-----------|------|
| `ANTHROPIC_API_KEY` | Claude analysis and chat | Pay per use, https://console.anthropic.com |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Reddit sentiment | Free, https://www.reddit.com/prefs/apps ("script" app) |
| `TWITTER_BEARER_TOKEN` | X sentiment | Paid: recent search needs X's Basic tier or higher |
| *(none)* | 4chan /biz/ sentiment | Free public API |

With no keys at all, market data, model training, predictions, backtests, risk
metrics and 4chan sentiment all work. Claude analysis uses `claude-opus-5-5` by
default; set `ANTHROPIC_MODEL` (for example `claude-sonnet-5-5`) to change it.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Model status, out-of-sample scores, training period |
| `GET` | `/api/market/{symbol}` | Market data from Yahoo Finance |
| `POST` | `/api/train` | Train all models (runs in the background) |
| `POST` | `/api/predict` | Next-session predictions from every model |
| `POST` | `/api/backtest` | Backtest a model on sessions after its training data |
| `GET` | `/api/risk` | VaR, CVaR, volatility, skew, kurtosis, normality tests |
| `POST` | `/api/sentiment/scrape` | Scrape Reddit, X and 4chan |
| `GET` | `/api/sentiment` | Latest sentiment, with per-source status |
| `POST` | `/api/analyze` | Claude's written analysis |
| `POST` | `/api/chat` | Chat with Claude about the current results |
| `GET`/`POST` | `/api/igon/*` | Igon value screener endpoints |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/train -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "period": "2y"}'
curl http://127.0.0.1:8000/api/health          # repeat until training.status is "complete"
curl -X POST http://127.0.0.1:8000/api/backtest -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "model": "dnn", "mode": "event"}'
```

## How It Works

### Features and labels
The feature engine computes 13 features from daily OHLCV data: today's and the
previous four daily log returns, SMA-20, SMA-50 and EMA-12 relative to the close,
20-day volatility, RSI, MACD, volume ratio and ATR. Each row's features use data
up to that day's close, and its label is whether the **next** session closed higher.

Social sentiment is not a model input. The scraper produces one current reading,
not a history aligned with past prices, so as a feature it would be a constant the
models cannot learn from. Sentiment goes to the Claude analyst instead.

### Models
- **DNN**: Dense(64) → Dense(64) → Dense(32) → Dense(1), with dropout
- **LSTM**: LSTM(64) → LSTM(32) → Dense(1) over 20-session windows
- **DQL agent**: deep Q-learning with experience replay and epsilon-greedy
  exploration; actions BUY / HOLD / SELL
- **Ensemble**: a weighted average of the three. A model earns weight only if
  it beats its baseline with statistical significance (below); otherwise the
  weights are equal and every ensemble signal carries a no-skill warning.

### Evaluation methodology
- **Chronological split**: models train on the first 80% of the period and are
  scored on the remaining 20%, which they never see. The scaler is fit on the
  training rows only.
- **Baseline**: each accuracy is shown next to the accuracy of always
  predicting the training period's majority direction.
- **Significance**: a one-sided exact McNemar test compares each model with the
  baseline on the same sessions. The UI reports an edge only when p < 0.05.
- **Backtests** trade only sessions after the training cutoff, charge
  transaction costs, and compare against buy-and-hold over the same sessions.
  Event-mode stops and targets are checked at the daily close and filled at that
  close, so a gap past a stop is booked at the real price.
- **Leakage test**: on random-walk prices, where nothing is predictable, a model
  trained on these features scores about 50% out of sample (see
  `tests/test_features.py`).

### Sentiment
- **Reddit**: hot posts and top comments from r/wallstreetbets, r/stocks,
  r/investing, r/options and five more
- **X**: recent-search API, financial keywords and tickers
- **4chan /biz/**: public catalog and thread API
- Scoring uses **VADER** (a sentiment lexicon) extended with financial terms such as
  moon (+2.5) and crash (-3.0). Within each source, posts are weighted by engagement;
  sources are then combined by post count.
- Each scrape reports, per source, whether it returned posts, had no credentials,
  or failed, and why.

### Claude analysis
The analyst sends the models' out-of-sample results, sentiment and risk metrics
to Claude and returns a written analysis; chat answers follow-up questions. The
prompt instructs Claude to treat models without a significant edge as no
evidence. Failed API calls are reported as errors, never replaced by canned text.

## Running the Tests

```bash
cd nexus-ai/backend
pip install -r requirements-dev.txt
python -m pytest
```

The suite runs offline (no network or API keys) and covers leakage,
backtesting, model evaluation, the Claude integration and the sentiment scraper.

## Important Disclaimers

⚠️ **This is an educational project. It is NOT financial advice.**

- Next-day direction is hard to predict. On real markets, expect out-of-sample
  accuracy close to the baseline; the app tells you when no model has a
  significant edge.
- Backtest results describe one historical period and do not predict future results.
- Social sentiment is noisy and can be manipulated.
- Never trade real money based on these signals. Consult a qualified professional.

## Troubleshooting

**"npm is not recognized"**: install Node.js from https://nodejs.org and restart the terminal.

**"python is not recognized"**: install Python from https://python.org with "Add to PATH" ticked.

**PowerShell blocks scripts**: run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.

**TensorFlow GPU warnings**: harmless; the CPU build works.

**Reddit 401/403**: check the credentials at reddit.com/prefs/apps.

**X 403**: the free X tier does not include recent search; it needs Basic or higher.
The Sentiment tab shows this on the X badge.

**4chan rate limiting**: the scraper pauses one second between threads. If blocked, wait a few minutes.

**Claude errors**: the chat shows the exact cause (invalid key, rate limit, network).
