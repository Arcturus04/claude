import React, { useState, useEffect, useCallback } from 'react';
import { api } from './api';
import Dashboard from './components/Dashboard';
import Models from './components/Models';
import RiskEngine from './components/RiskEngine';
import TradingBot from './components/TradingBot';
import Sentiment from './components/Sentiment';
import Chat from './components/Chat';
import './styles.css';

const TABS = [
  { id: 'dashboard', label: 'Command Center' },
  { id: 'models', label: 'AI Models' },
  { id: 'risk', label: 'Risk Engine' },
  { id: 'sentiment', label: 'Sentiment' },
  { id: 'bot', label: 'Trading Bot' },
  { id: 'chat', label: 'Claude Chat' },
];

export default function App() {
  const [tab, setTab] = useState('dashboard');
  const [health, setHealth] = useState(null);
  const [marketData, setMarketData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [sentimentData, setSentimentData] = useState(null);
  const [riskData, setRiskData] = useState(null);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState(null);
  const [symbol, setSymbol] = useState('SPY');

  const fetchHealth = useCallback(async () => {
    try {
      const data = await api.health();
      setHealth(data);
      setError(null);
    } catch (e) {
      setError('Backend not reachable. Start the server with: python server.py');
    }
  }, []);

  const fetchMarket = useCallback(async () => {
    try {
      const data = await api.market(symbol);
      setMarketData(data);
    } catch (e) {
      console.error('Market fetch failed:', e.message);
    }
  }, [symbol]);

  useEffect(() => {
    fetchHealth();
    fetchMarket();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, [fetchHealth, fetchMarket]);

  const handleTrain = async (params) => {
    setTraining(true);
    try {
      await api.train({ symbol, ...params });
      // Poll health until trained
      const poll = setInterval(async () => {
        const h = await api.health();
        setHealth(h);
        const allActive = h.models &&
          Object.values(h.models).every(m => m && m.status === 'active');
        if (allActive) {
          clearInterval(poll);
          setTraining(false);
          // Fetch predictions after training
          try {
            const pred = await api.predict(symbol);
            setPrediction(pred);
          } catch (e) { /* ignore */ }
          try {
            const risk = await api.risk();
            setRiskData(risk);
          } catch (e) { /* ignore */ }
        }
      }, 5000);
    } catch (e) {
      setError(e.message);
      setTraining(false);
    }
  };

  const handleScrapeSentiment = async () => {
    try {
      await api.scrapeSentiment();
      // Poll for results
      setTimeout(async () => {
        try {
          const data = await api.sentiment();
          setSentimentData(data);
        } catch (e) { /* ignore */ }
      }, 10000);
    } catch (e) {
      setError(e.message);
    }
  };

  const trained = health?.models &&
    Object.values(health.models).some(m => m && m.status === 'active');

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <h1 className="logo">NEXUS<span className="dot">.</span>AI</h1>
          <span className="subtitle">Artificial Financial Intelligence</span>
        </div>
        <div className="header-center">
          <div className="symbol-input">
            <label>Symbol:</label>
            <input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              maxLength={5}
            />
            <button onClick={fetchMarket} className="btn-sm">Refresh</button>
          </div>
        </div>
        <div className="header-right">
          <div className={`status-dot ${health ? 'online' : 'offline'}`} />
          <span className="status-text">
            {health ? 'System Online' : 'Offline'}
          </span>
          {trained && <span className="badge trained">Models Trained</span>}
        </div>
      </header>

      {error && (
        <div className="error-banner">
          {error}
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      <nav className="tabs">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`tab ${tab === t.id ? 'active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className="content">
        {tab === 'dashboard' && (
          <Dashboard
            marketData={marketData}
            prediction={prediction}
            health={health}
            training={training}
            onTrain={handleTrain}
            symbol={symbol}
          />
        )}
        {tab === 'models' && (
          <Models
            health={health}
            prediction={prediction}
            training={training}
            onTrain={handleTrain}
          />
        )}
        {tab === 'risk' && (
          <RiskEngine
            riskData={riskData}
            onRefresh={async () => {
              try {
                const data = await api.risk();
                setRiskData(data);
              } catch (e) {
                setError('Train models first to get risk metrics.');
              }
            }}
          />
        )}
        {tab === 'sentiment' && (
          <Sentiment
            data={sentimentData}
            onScrape={handleScrapeSentiment}
          />
        )}
        {tab === 'bot' && (
          <TradingBot
            prediction={prediction}
            health={health}
          />
        )}
        {tab === 'chat' && (
          <Chat />
        )}
      </main>

      <footer className="footer">
        <span>NEXUS AI - Educational purposes only. Not financial advice.</span>
        <span>Based on "Artificial Intelligence in Finance" by Yves Hilpisch</span>
      </footer>
    </div>
  );
}
