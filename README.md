# ╔══════════════════════════════════════════════════════════════╗
# ║  NEXUS•AI — Artificial Financial Intelligence Platform     ║
# ║  ML Trading Models + Social Sentiment + Claude Analysis    ║
# ╚══════════════════════════════════════════════════════════════╝

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    REACT FRONTEND (:3000)                    │
│  Dashboard │ AI Models │ Risk Engine │ Trading Bot │ Chat   │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST API
┌──────────────────────────▼──────────────────────────────────┐
│                 FASTAPI BACKEND (:8000)                      │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │   DNN    │  │   RNN    │  │   DQL    │  │  Ensemble  │  │
│  │ Predictor│  │ Sequencer│  │  Agent   │  │  Meta-Learn│  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘  │
│       └──────────────┴─────────────┴──────────────┘          │
│                          │                                   │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │              FEATURE ENGINE                            │  │
│  │  Technical Indicators + Sentiment Score → ML Features  │  │
│  └───────────┬────────────────────────────┬──────────────┘  │
│              │                            │                  │
│  ┌───────────▼──────────┐  ┌─────────────▼──────────────┐  │
│  │   MARKET DATA        │  │   SENTIMENT SCRAPER        │  │
│  │   Yahoo Finance API  │  │   Reddit + Twitter + 4chan  │  │
│  └──────────────────────┘  └────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              CLAUDE ANALYST                           │   │
│  │  Interprets all signals → Natural language analysis   │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start Guide

### Prerequisites

- **Python 3.10+** → https://www.python.org/downloads/
- **Node.js 18+** → https://nodejs.org/
- Both must be installed and in your PATH

### Step 1 — Clone & Setup Backend

```bash
# Navigate to the backend folder
cd backend

# Create a virtual environment (recommended)
python -m venv venv

# Activate it:
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2 — Configure API Keys

Copy the example env file and fill in your keys:

```bash
copy .env.example .env     # Windows
cp .env.example .env        # Mac/Linux
```

Edit `.env` with your keys:

```
ANTHROPIC_API_KEY=sk-ant-...      # From console.anthropic.com
REDDIT_CLIENT_ID=...               # From reddit.com/prefs/apps
REDDIT_CLIENT_SECRET=...
TWITTER_BEARER_TOKEN=...           # From developer.x.com
```

**Which keys are required?**
- **Anthropic (Claude):** Needed for AI analysis & chat. Get it at https://console.anthropic.com
- **Reddit:** Needed for Reddit sentiment. Free at https://www.reddit.com/prefs/apps (create "script" type app)
- **Twitter/X:** Needed for Twitter sentiment. Apply at https://developer.x.com
- **4chan:** No key needed — public API

**You can start with ZERO keys** — the system works without them, just without sentiment/Claude features.

### Step 3 — Start the Backend

```bash
python server.py
```

The server starts at `http://localhost:8000`. You'll see:
```
══════════════════════════════════════════════════════
  NEXUS•AI — Artificial Financial Intelligence Platform
══════════════════════════════════════════════════════
```

### Step 4 — Setup Frontend

Open a NEW terminal (keep the backend running):

```bash
# If you already have the nexus-ai React app:
cd nexus-ai
npm install recharts lucide-react
npm start

# Or create fresh:
npx create-react-app nexus-ai
cd nexus-ai
npm install recharts lucide-react
# Copy the React artifact into src/App.js
npm start
```

## API Endpoints

Once the backend is running, you can use these endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Server status |
| `GET` | `/api/health` | Model & system health |
| `GET` | `/api/market/{symbol}` | Fetch market data (e.g., `/api/market/SPY`) |
| `POST` | `/api/train` | Train all ML models |
| `POST` | `/api/predict` | Get model predictions |
| `POST` | `/api/sentiment/scrape` | Trigger sentiment scrape |
| `GET` | `/api/sentiment` | Get cached sentiment data |
| `POST` | `/api/analyze` | Get Claude's full analysis |
| `POST` | `/api/chat` | Chat with Claude about markets |
| `GET` | `/api/risk` | Calculate risk metrics |

### Example: Train & Predict

```bash
# Train models on 2 years of SPY data
curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "period": "2y", "epochs_dnn": 50}'

# Wait for training to complete, then predict
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SPY"}'

# Scrape sentiment
curl -X POST http://localhost:8000/api/sentiment/scrape

# Get Claude's analysis
curl -X POST http://localhost:8000/api/analyze

# Chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What does the sentiment on Reddit suggest about NVDA?"}'
```

You can also open `http://localhost:8000/docs` in your browser to see the
interactive Swagger API documentation that FastAPI generates automatically.

## How It All Works

### 1. Data Pipeline
- **Yahoo Finance** provides OHLCV (Open/High/Low/Close/Volume) data
- **Feature Engine** computes 14 features: lagged returns, SMA, EMA, RSI, MACD, ATR, volatility, volume ratio, and social sentiment score
- Sentiment score from Reddit/Twitter/4chan is injected as a feature

### 2. ML Models (from the book)
- **DNN Predictor**: 3-layer Dense network (64→64→32→1) with dropout, predicts direction (up/down)
- **RNN Sequencer**: 2-layer LSTM (64→32→1), uses 20-step sequences to capture temporal patterns
- **DQL Agent**: Deep Q-Learning with experience replay buffer (2000), epsilon-greedy exploration (1.0→0.01), 3 actions (BUY/HOLD/SELL)
- **Ensemble**: Weighted average of all models, weights adjusted by accuracy

### 3. Sentiment Intelligence
- **Reddit**: Scrapes r/wallstreetbets, r/stocks, r/investing, r/cryptocurrency + 6 more
- **Twitter/X**: Searches for financial keywords and ticker mentions
- **4chan /biz/**: Scrapes the Business & Finance board for contrarian signals
- **VADER** sentiment analysis enhanced with financial lexicon (moon=+2.5, crash=-3.0, etc.)
- Posts are weighted by engagement (upvotes, likes, replies)

### 4. Claude Analysis
- Receives ML signals, sentiment data, and risk metrics
- Provides natural language interpretation
- Interactive chat for follow-up questions
- Acknowledges uncertainty and never gives financial advice

## Important Disclaimers

⚠️ **This is an educational project.** It is NOT financial advice.

- Past model performance does not predict future results
- The ML models typically achieve 55-65% directional accuracy — barely above coin flip
- Social sentiment is noisy and can be manipulated
- Never trade real money based solely on these signals
- Always consult qualified financial professionals

## Troubleshooting

**"npm is not recognized"** → Install Node.js from https://nodejs.org and restart your terminal

**"python is not recognized"** → Install Python from https://python.org, check "Add to PATH" during install

**PowerShell blocks scripts** → Run: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

**TensorFlow GPU errors** → The CPU version works fine. Ignore GPU warnings.

**Reddit API 403 error** → Your Reddit app credentials may be wrong. Double-check at reddit.com/prefs/apps

**Twitter API 403 error** → You need "Elevated" access on the Twitter developer portal

**4chan rate limiting** → The scraper includes 1-second delays. If blocked, wait a few minutes.
