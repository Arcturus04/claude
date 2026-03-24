const API_BASE = '/api';

async function request(method, path, body = null) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(`${API_BASE}${path}`, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

export const api = {
  health: () => request('GET', '/health'),
  market: (symbol = 'SPY', period = '6mo') => request('GET', `/market/${symbol}?period=${period}`),
  train: (params = {}) => request('POST', '/train', {
    symbol: params.symbol || 'SPY',
    period: params.period || '2y',
    epochs_dnn: params.epochs_dnn || 50,
    epochs_rnn: params.epochs_rnn || 30,
    episodes_dql: params.episodes_dql || 50,
  }),
  predict: (symbol = 'SPY') => request('POST', '/predict', { symbol }),
  sentiment: () => request('GET', '/sentiment'),
  scrapeSentiment: () => request('POST', '/sentiment/scrape'),
  analyze: () => request('POST', '/analyze'),
  chat: (message) => request('POST', '/chat', { message }),
  risk: () => request('GET', '/risk'),
  backtest: (params = {}) => request('POST', '/backtest', {
    symbol: params.symbol || 'SPY',
    period: params.period || '2y',
    model: params.model || 'dnn',
    mode: params.mode || 'vectorized',
    stop_loss: params.stop_loss || 0.0175,
    take_profit: params.take_profit || 0.0185,
    initial_capital: params.initial_capital || 10000,
  }),
};
