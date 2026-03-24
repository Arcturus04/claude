import React, { useMemo } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar,
} from 'recharts';

export default function Dashboard({ marketData, prediction, health, training, onTrain, symbol }) {
  const chartData = useMemo(() => {
    if (!marketData?.data) return [];
    return marketData.data.map(d => ({
      date: d.date,
      close: d.close,
      volume: d.volume / 1e6,
    }));
  }, [marketData]);

  const lastPrice = chartData.length > 0 ? chartData[chartData.length - 1].close : null;
  const firstPrice = chartData.length > 0 ? chartData[0].close : null;
  const periodReturn = lastPrice && firstPrice
    ? ((lastPrice - firstPrice) / firstPrice * 100).toFixed(2)
    : null;

  const ensembleSignal = prediction?.ensemble?.signal || 'N/A';
  const ensembleConf = prediction?.ensemble?.confidence
    ? (prediction.ensemble.confidence * 100).toFixed(1)
    : '---';

  return (
    <div className="dashboard">
      <div className="metrics-row">
        <div className="metric-card">
          <div className="metric-label">{symbol} Price</div>
          <div className="metric-value">${lastPrice?.toFixed(2) || '---'}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Period Return</div>
          <div className={`metric-value ${periodReturn > 0 ? 'green' : 'red'}`}>
            {periodReturn ? `${periodReturn}%` : '---'}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Ensemble Signal</div>
          <div className={`metric-value ${
            ensembleSignal === 'LONG' ? 'green' :
            ensembleSignal === 'SHORT' ? 'red' : ''
          }`}>
            {ensembleSignal}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">AI Confidence</div>
          <div className="metric-value">{ensembleConf}%</div>
        </div>
      </div>

      <div className="chart-section">
        <div className="chart-header">
          <h3>{symbol} Price History - {marketData?.period || '6mo'}</h3>
          <button
            className="btn-primary"
            onClick={() => onTrain({ symbol })}
            disabled={training}
          >
            {training ? 'Training Models...' : 'Train All Models'}
          </button>
        </div>
        <ResponsiveContainer width="100%" height={350}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#00d4ff" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="date" stroke="#888" tick={{ fontSize: 11 }}
              tickFormatter={d => d.slice(5)} />
            <YAxis stroke="#888" domain={['auto', 'auto']}
              tick={{ fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: '#1a1a2e', border: '1px solid #333', color: '#fff' }}
              formatter={(v) => [`$${v.toFixed(2)}`, 'Price']}
            />
            <Area type="monotone" dataKey="close" stroke="#00d4ff"
              fill="url(#priceGrad)" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="two-col">
        <div className="chart-section">
          <h3>Volume (Millions)</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData.slice(-30)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="date" stroke="#888" tick={{ fontSize: 10 }}
                tickFormatter={d => d.slice(5)} />
              <YAxis stroke="#888" tick={{ fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: '#1a1a2e', border: '1px solid #333', color: '#fff' }}
                formatter={(v) => [`${v.toFixed(1)}M`, 'Volume']}
              />
              <Bar dataKey="volume" fill="#6c5ce7" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="system-status">
          <h3>System Status</h3>
          {health?.models ? (
            <div className="status-list">
              {Object.entries(health.models).map(([key, model]) => (
                <div key={key} className="status-item">
                  <span className={`status-indicator ${
                    model?.status === 'active' ? 'active' :
                    model?.status === 'training' ? 'training' : 'idle'
                  }`} />
                  <span className="status-name">{model?.name || key.toUpperCase()}</span>
                  <span className="status-detail">
                    {model?.status === 'active'
                      ? `${model.accuracy}% acc`
                      : model?.status || 'untrained'}
                  </span>
                </div>
              ))}
              <div className="status-item">
                <span className={`status-indicator ${health.sentiment_available ? 'active' : 'idle'}`} />
                <span className="status-name">Sentiment Engine</span>
                <span className="status-detail">
                  {health.sentiment_available ? 'Data Available' : 'No Data'}
                </span>
              </div>
              <div className="status-item">
                <span className={`status-indicator ${health.claude_available ? 'active' : 'idle'}`} />
                <span className="status-name">Claude Analyst</span>
                <span className="status-detail">
                  {health.claude_available ? 'Connected' : 'No API Key'}
                </span>
              </div>
            </div>
          ) : (
            <p className="muted">Connect to backend to see status</p>
          )}
        </div>
      </div>
    </div>
  );
}
