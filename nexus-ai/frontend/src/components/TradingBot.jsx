import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts';
import { api } from '../api';

// Every number on this page comes from the backend. Strategy performance is
// an out-of-sample backtest: only sessions after the training cutoff are
// traded, against a buy-and-hold benchmark over the same sessions.

const pct = (v) => (v === null || v === undefined ? 'n/a' : `${v}%`);
const num = (v) => (v === null || v === undefined ? 'n/a' : v);
const money = (v) => (v === null || v === undefined ? 'n/a' : `$${Number(v).toLocaleString()}`);

function BacktestMetrics({ result }) {
  const isEvent = result.mode === 'event';
  const rows = isEvent
    ? [
        ['Strategy return', pct(result.total_return)],
        ['Buy & hold return', pct(result.total_return_market)],
        ['Max drawdown', pct(result.max_drawdown)],
        ['Trades', num(result.total_trades)],
        ['Win rate', pct(result.win_rate)],
        ['Profit factor', num(result.profit_factor)],
        ['Payoff ratio', num(result.payoff_ratio)],
        ['Stops / targets hit', `${result.stop_loss_hits} / ${result.take_profit_hits}`],
        ['Transaction costs', money(result.transaction_costs_total)],
      ]
    : [
        ['Strategy return', pct(result.total_return_strategy)],
        ['Buy & hold return', pct(result.total_return_market)],
        ['Out/under-performance', pct(result.outperformance)],
        ['Sharpe ratio', num(result.sharpe_ratio)],
        ['Sortino ratio', num(result.sortino_ratio)],
        ['Max drawdown', pct(result.max_drawdown)],
        ['Position changes', num(result.total_trades)],
        ['In-market win rate', pct(result.win_rate)],
        ['Transaction costs', money(result.transaction_costs_total)],
      ];
  return (
    <div className="params-grid">
      {rows.map(([label, value]) => (
        <div key={label} className="param-card">
          <span className="param-name">{label}</span>
          <span className="param-value">{value}</span>
        </div>
      ))}
    </div>
  );
}

export default function TradingBot({ health, symbol = 'SPY' }) {
  const [model, setModel] = useState('dnn');
  const [mode, setMode] = useState('vectorized');
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);

  const [monitoring, setMonitoring] = useState(false);
  const [signalLog, setSignalLog] = useState([]);

  const dqlStatus = health?.models?.dql;
  const trainingInfo = health?.training;
  const isReady = trainingInfo?.status === 'complete';

  const runBacktest = async () => {
    setRunning(true);
    setError(null);
    try {
      setResult(await api.backtest({ symbol, model, mode }));
    } catch (e) {
      setResult(null);
      setError(e.message);
    } finally {
      setRunning(false);
    }
  };

  // Live signal monitor: records what the agent would do now. It places no
  // orders and invents no profit; how a signal would have traded is what the
  // backtest measures.
  useEffect(() => {
    if (!(monitoring && isReady)) return undefined;
    const tick = async () => {
      try {
        const pred = await api.predict(symbol);
        const dql = pred.models.dql;
        setSignalLog(prev => [{
          time: new Date().toLocaleTimeString(),
          action: dql.action || 'HOLD',
          symbol: pred.symbol,
          price: pred.current_price,
          confidence: dql.confidence,
          ensemble: pred.ensemble?.signal,
        }, ...prev].slice(0, 50));
      } catch (e) {
        setError(`Signal update failed: ${e.message}`);
      }
    };
    tick();
    const id = setInterval(tick, 60000);
    return () => clearInterval(id);
  }, [monitoring, isReady, symbol]);

  const chartData = (result?.equity_curve || []).map(p => ({
    date: p.date,
    strategy: p.strategy,
    market: p.market,
  }));
  const oos = result?.out_of_sample;

  return (
    <div className="bot-page">
      <div className="page-header">
        <h2>DQL Trading Agent</h2>
        <div className="bot-controls">
          <button
            className={`btn-${monitoring ? 'danger' : 'success'}`}
            onClick={() => setMonitoring(!monitoring)}
            disabled={!isReady}
          >
            {monitoring ? 'STOP SIGNAL MONITOR' : 'START SIGNAL MONITOR'}
          </button>
        </div>
      </div>

      {!isReady && (
        <div className="warning-banner">
          Models not trained. Go to the AI Models tab and train them first.
        </div>
      )}
      {error && <div className="error-banner">{error}</div>}

      <div className="chart-section">
        <h3>Out-of-Sample Backtest: {symbol}</h3>
        <div className="bot-controls" style={{ marginBottom: 12 }}>
          <select value={model} onChange={e => setModel(e.target.value)}>
            <option value="dnn">DNN</option>
            <option value="rnn">LSTM</option>
            <option value="ensemble">Ensemble (DNN + DQL)</option>
          </select>
          <select value={mode} onChange={e => setMode(e.target.value)}>
            <option value="vectorized">Vectorized (daily positions)</option>
            <option value="event">Event-based (stops and targets)</option>
          </select>
          <button className="btn-primary" onClick={runBacktest} disabled={!isReady || running}>
            {running ? 'Running...' : 'Run Backtest'}
          </button>
        </div>

        {oos && (
          <p className="muted">
            Models trained on {oos.trained_symbol} up to {oos.train_end_date}. This backtest
            trades only {oos.start} to {oos.end} ({oos.sessions} sessions), data the models never
            saw, with transaction costs included. {oos.note}
            {oos.trained_symbol && oos.trained_symbol.toUpperCase() !== symbol.toUpperCase() &&
              ` Note: the models were trained on ${oos.trained_symbol}, not ${symbol}.`}
          </p>
        )}

        {chartData.length > 0 ? (
          <>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="date" stroke="#888" tick={{ fontSize: 10 }} minTickGap={30} />
                <YAxis stroke="#888" domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #333', color: '#fff' }}
                />
                <Legend />
                <Line type="monotone" dataKey="strategy" stroke="#00d4ff"
                  strokeWidth={2} dot={false} name="Strategy" />
                <Line type="monotone" dataKey="market" stroke="#888"
                  strokeWidth={1} strokeDasharray="5 5" dot={false} name="Buy & hold" />
              </LineChart>
            </ResponsiveContainer>
            <BacktestMetrics result={result} />
          </>
        ) : (
          <div className="empty-chart">
            {isReady ? 'Run a backtest to see out-of-sample performance' : 'Train the models first'}
          </div>
        )}
      </div>

      <div className="two-col">
        <div className="trade-log">
          <h3>{result?.mode === 'event' ? 'Backtest Trades (last 50)' : 'Signal Monitor'}</h3>
          <div className="log-scroll">
            {result?.mode === 'event' ? (
              result.trades.length > 0 ? result.trades.map((t, i) => (
                <div key={i} className={`trade-entry backtest ${t.direction === 'LONG' ? 'buy' : 'sell'}`}>
                  <span className="trade-time">{t.entry_date} to {t.exit_date}</span>
                  <span className={`trade-action ${t.direction === 'LONG' ? 'green' : 'red'}`}>
                    {t.direction}
                  </span>
                  <span className="trade-price">${t.entry_price} to ${t.exit_price}</span>
                  <span className="trade-reason">{t.exit_reason.replace(/_/g, ' ')}</span>
                  <span className={`trade-pnl ${t.pnl >= 0 ? 'green' : 'red'}`}>
                    {t.pnl >= 0 ? '+' : ''}{t.pnl.toFixed(2)}
                  </span>
                </div>
              )) : <p className="muted">No trades in this period</p>
            ) : signalLog.length > 0 ? signalLog.map((s, i) => (
              <div key={i} className={`trade-entry ${s.action.toLowerCase()}`}>
                <span className="trade-time">{s.time}</span>
                <span className={`trade-action ${
                  s.action === 'BUY' ? 'green' : s.action === 'SELL' ? 'red' : ''
                }`}>{s.action}</span>
                <span className="trade-symbol">{s.symbol}</span>
                <span className="trade-price">${s.price}</span>
                <span className="trade-pnl">conf {(s.confidence * 100).toFixed(0)}%</span>
              </div>
            )) : (
              <p className="muted">
                Start the signal monitor to log the agent's live signal once a minute. Signals
                only: no orders are placed and no profit is computed.
              </p>
            )}
          </div>
        </div>

        <div className="bot-params">
          <h3>Agent Hyperparameters (from the trained model)</h3>
          {dqlStatus ? (
            <div className="params-grid">
              {[
                ['Gamma (discount)', dqlStatus.gamma],
                ['Epsilon (explore)', dqlStatus.epsilon],
                ['Epsilon Decay', dqlStatus.epsilon_decay],
                ['Batch Size', dqlStatus.batch_size],
                ['Input Features', dqlStatus.input_features],
                ['Replay Memory', dqlStatus.memory_size],
              ].map(([name, value]) => (
                <div key={name} className="param-card">
                  <span className="param-name">{name}</span>
                  <span className="param-value">{num(value)}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">Backend not reachable.</p>
          )}
        </div>
      </div>

      <div className="architecture-section">
        <h3>Neural Network Architecture</h3>
        <div className="arch-diagram">
          <div className="arch-layer input">
            <span className="layer-name">Input</span>
            <span className="layer-detail">{num(dqlStatus?.input_features)} features</span>
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
