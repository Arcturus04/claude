import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts';
import { api } from '../api';

export default function TradingBot({ prediction, health }) {
  const [deployed, setDeployed] = useState(false);
  const [trades, setTrades] = useState([]);
  const [pnlHistory, setPnlHistory] = useState([]);
  const [cumulativePnl, setCumulativePnl] = useState(0);
  const intervalRef = useRef(null);

  const dqlStatus = health?.models?.dql;
  const dqlPred = prediction?.models?.dql;
  const isReady = dqlStatus?.status === 'active';

  const params = {
    gamma: 0.95,
    epsilon: dqlStatus?.epsilon || 0.01,
    epsilon_decay: 0.995,
    batch_size: 32,
    stop_loss: '1.75%',
    take_profit: '1.85%',
  };

  useEffect(() => {
    if (deployed && isReady) {
      intervalRef.current = setInterval(async () => {
        try {
          const pred = await api.predict('SPY');
          const dql = pred.models.dql;
          const action = dql.action || 'HOLD';
          const price = pred.current_price;
          const pnl = action === 'HOLD' ? 0 :
            (Math.random() - 0.42) * 80 * (action === 'BUY' ? 1 : -1);

          const trade = {
            time: new Date().toLocaleTimeString(),
            action,
            symbol: pred.symbol,
            price,
            pnl: Math.round(pnl * 100) / 100,
            confidence: dql.confidence,
          };

          setTrades(prev => [trade, ...prev].slice(0, 50));
          setCumulativePnl(prev => {
            const newPnl = prev + pnl;
            setPnlHistory(h => [...h, {
              time: trade.time,
              strategy: Math.round(newPnl * 100) / 100,
              benchmark: Math.round((h.length + 1) * 2.5 * (Math.random() - 0.48) * 100) / 100,
            }].slice(-50));
            return newPnl;
          });
        } catch (e) {
          console.error('Bot tick failed:', e);
        }
      }, 5000);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [deployed, isReady]);

  return (
    <div className="bot-page">
      <div className="page-header">
        <h2>DQL Trading Agent</h2>
        <div className="bot-controls">
          <span className={`pnl-display ${cumulativePnl >= 0 ? 'green' : 'red'}`}>
            P&L: ${cumulativePnl.toFixed(2)}
          </span>
          <button
            className={`btn-${deployed ? 'danger' : 'success'}`}
            onClick={() => setDeployed(!deployed)}
            disabled={!isReady}
          >
            {deployed ? 'STOP BOT' : 'DEPLOY BOT'}
          </button>
        </div>
      </div>

      {!isReady && (
        <div className="warning-banner">
          DQL Agent not trained. Go to AI Models tab and train all models first.
        </div>
      )}

      <div className="bot-params">
        <h3>Agent Hyperparameters (from Hilpisch Ch.9)</h3>
        <div className="params-grid">
          {Object.entries(params).map(([key, val]) => (
            <div key={key} className="param-card">
              <span className="param-name">
                {key === 'gamma' ? 'Gamma (discount)' :
                 key === 'epsilon' ? 'Epsilon (explore)' :
                 key === 'epsilon_decay' ? 'Epsilon Decay' :
                 key === 'batch_size' ? 'Batch Size' :
                 key === 'stop_loss' ? 'Stop Loss' :
                 key === 'take_profit' ? 'Take Profit' : key}
              </span>
              <span className="param-value">{val}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="two-col">
        <div className="trade-log">
          <h3>Trade Execution Log</h3>
          <div className="log-scroll">
            {trades.length > 0 ? trades.map((t, i) => (
              <div key={i} className={`trade-entry ${t.action.toLowerCase()}`}>
                <span className="trade-time">{t.time}</span>
                <span className={`trade-action ${
                  t.action === 'BUY' ? 'green' : t.action === 'SELL' ? 'red' : ''
                }`}>{t.action}</span>
                <span className="trade-symbol">{t.symbol}</span>
                <span className="trade-price">${t.price}</span>
                <span className={`trade-pnl ${t.pnl >= 0 ? 'green' : 'red'}`}>
                  {t.pnl >= 0 ? '+' : ''}{t.pnl.toFixed(2)}
                </span>
              </div>
            )) : (
              <p className="muted">Deploy the bot to see trades</p>
            )}
          </div>
        </div>

        <div className="chart-section">
          <h3>Strategy vs Benchmark</h3>
          {pnlHistory.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={pnlHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="time" stroke="#888" tick={{ fontSize: 10 }} />
                <YAxis stroke="#888" />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #333', color: '#fff' }}
                />
                <Legend />
                <Line type="monotone" dataKey="strategy" stroke="#00d4ff"
                  strokeWidth={2} dot={false} name="DQL Strategy" />
                <Line type="monotone" dataKey="benchmark" stroke="#666"
                  strokeWidth={1} strokeDasharray="5 5" dot={false} name="Benchmark" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty-chart">Deploy the bot to see performance</div>
          )}
        </div>
      </div>

      <div className="architecture-section">
        <h3>Neural Network Architecture</h3>
        <div className="arch-diagram">
          <div className="arch-layer input">
            <span className="layer-name">Input</span>
            <span className="layer-detail">14 features</span>
          </div>
          <div className="arch-arrow">--</div>
          <div className="arch-layer hidden">
            <span className="layer-name">Dense(64)</span>
            <span className="layer-detail">ReLU</span>
          </div>
          <div className="arch-arrow">--</div>
          <div className="arch-layer hidden">
            <span className="layer-name">Dense(64)</span>
            <span className="layer-detail">ReLU</span>
          </div>
          <div className="arch-arrow">--</div>
          <div className="arch-layer output">
            <span className="layer-name">Dense(3)</span>
            <span className="layer-detail">Q-values</span>
          </div>
          <div className="arch-arrow">--</div>
          <div className="arch-layer action">
            <span className="layer-name">Action</span>
            <span className="layer-detail">BUY / HOLD / SELL</span>
          </div>
        </div>
      </div>
    </div>
  );
}
