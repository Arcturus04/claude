import React, { useState, useEffect } from 'react';
import { api } from '../api';

const SIGNAL_COLORS = {
  STRONG_BUY: '#00e676',
  BUY: '#4caf50',
  HOLD: '#ff9800',
  REDUCE: '#ff5722',
  SELL: '#f44336',
};

const SIGNAL_LABELS = {
  STRONG_BUY: 'STRONG BUY',
  BUY: 'BUY',
  HOLD: 'HOLD',
  REDUCE: 'REDUCE',
  SELL: 'SELL',
};

export default function IgonScreener({ symbol }) {
  const [method, setMethod] = useState(null);
  const [screenResult, setScreenResult] = useState(null);
  const [exitSchedule, setExitSchedule] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [screenSymbol, setScreenSymbol] = useState(symbol || 'TTE.PA');

  // Manual score inputs
  const [manualPE, setManualPE] = useState(8);
  const [manualROE, setManualROE] = useState(15);
  const [manualGrowth, setManualGrowth] = useState(5);
  const [manualResult, setManualResult] = useState(null);

  // Position sizing inputs
  const [portfolioValue, setPortfolioValue] = useState(100000);
  const [stockPrice, setStockPrice] = useState(50);
  const [dailyVolume, setDailyVolume] = useState(500000);
  const [positionResult, setPositionResult] = useState(null);

  useEffect(() => {
    api.igonMethod().then(setMethod).catch(() => {});
  }, []);

  const handleScreen = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.igonScreen(screenSymbol);
      setScreenResult(result);
      // Auto-fetch exit schedule
      if (result.pe > 0) {
        const exit = await api.igonExitSchedule({ pe: result.pe, position_pct: 3.0 });
        setExitSchedule(exit);
      }
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  };

  const handleManualScore = async () => {
    try {
      const result = await api.igonScore({
        pe: parseFloat(manualPE),
        roe: parseFloat(manualROE),
        revenue_growth: parseFloat(manualGrowth),
      });
      setManualResult(result);
    } catch (e) {
      setError(e.message);
    }
  };

  const handlePositionSize = async () => {
    try {
      const result = await api.igonPositionSize({
        portfolio_value: parseFloat(portfolioValue),
        stock_price: parseFloat(stockPrice),
        avg_daily_volume: parseFloat(dailyVolume),
        current_lines: 20,
      });
      setPositionResult(result);
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <div className="panel-grid">
      {/* Method Overview */}
      <div className="panel full-width">
        <h2 className="panel-title">Igon Value Screen</h2>
        <p className="panel-subtitle">
          William Igon's methodology - 4000%+ returns since 1993 (12.8% annualized)
        </p>
        {method && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
            <div className="metric-card">
              <div className="metric-label">Entry Rules</div>
              <div style={{ fontSize: '0.85rem', color: '#aaa', marginTop: '0.5rem' }}>
                <div>PE &lt; {method.entry_rules.pe.replace('< ', '')}</div>
                <div>ROE &gt; {method.entry_rules.roe.replace('> ', '')}</div>
                <div>Revenue: {method.entry_rules.revenue}</div>
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Exit Rules</div>
              <div style={{ fontSize: '0.85rem', color: '#aaa', marginTop: '0.5rem' }}>
                <div>Take profit: PE {'>'}= 15</div>
                <div>Sell 20% per 1pt PE rise</div>
                <div>Full exit: PE {'>'}= 20</div>
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Position Sizing</div>
              <div style={{ fontSize: '0.85rem', color: '#aaa', marginTop: '0.5rem' }}>
                <div>{method.position_sizing.target_lines} lines at {method.position_sizing.weight_per_line}</div>
                <div>Build in {method.position_sizing.build_method}</div>
                <div>Max {method.position_sizing.volume_limit} daily vol</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Live Stock Screener */}
      <div className="panel" style={{ gridColumn: 'span 2' }}>
        <h3 className="panel-title">Live Stock Screen</h3>
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
          <input
            value={screenSymbol}
            onChange={(e) => setScreenSymbol(e.target.value.toUpperCase())}
            placeholder="Symbol (e.g. TTE.PA)"
            style={{
              flex: 1, padding: '0.5rem', background: '#1a1a2e',
              border: '1px solid #333', color: '#fff', borderRadius: '4px',
            }}
          />
          <button onClick={handleScreen} disabled={loading} className="btn-primary">
            {loading ? 'Screening...' : 'Screen'}
          </button>
        </div>

        {error && <div style={{ color: '#f44336', marginBottom: '1rem' }}>{error}</div>}

        {screenResult && (
          <div>
            <div style={{
              display: 'flex', alignItems: 'center', gap: '1rem',
              marginBottom: '1rem', padding: '1rem',
              background: '#1a1a2e', borderRadius: '8px',
              borderLeft: `4px solid ${SIGNAL_COLORS[screenResult.signal] || '#666'}`,
            }}>
              <div style={{
                fontSize: '1.5rem', fontWeight: 'bold',
                color: SIGNAL_COLORS[screenResult.signal] || '#666',
              }}>
                {SIGNAL_LABELS[screenResult.signal] || screenResult.signal}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '1.1rem', color: '#fff' }}>
                  {screenResult.name || screenResult.symbol}
                </div>
                <div style={{ color: '#888', fontSize: '0.85rem' }}>
                  {screenResult.sector} | Score: {screenResult.score}/100
                </div>
              </div>
              {screenResult.current_price && (
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '1.2rem', color: '#fff' }}>
                    {screenResult.current_price}
                  </div>
                </div>
              )}
            </div>

            {/* Criteria Check */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.5rem', marginBottom: '1rem' }}>
              <div className="metric-card" style={{
                borderColor: screenResult.criteria?.pe_ok ? '#4caf50' : '#f44336',
                borderWidth: '1px', borderStyle: 'solid',
              }}>
                <div className="metric-label">PE Ratio</div>
                <div className="metric-value">{screenResult.pe}</div>
                <div style={{ color: screenResult.criteria?.pe_ok ? '#4caf50' : '#f44336', fontSize: '0.8rem' }}>
                  {screenResult.criteria?.pe_ok ? 'PASS (< 12)' : 'FAIL (>= 12)'}
                </div>
              </div>
              <div className="metric-card" style={{
                borderColor: screenResult.criteria?.roe_ok ? '#4caf50' : '#f44336',
                borderWidth: '1px', borderStyle: 'solid',
              }}>
                <div className="metric-label">ROE</div>
                <div className="metric-value">{screenResult.roe}%</div>
                <div style={{ color: screenResult.criteria?.roe_ok ? '#4caf50' : '#f44336', fontSize: '0.8rem' }}>
                  {screenResult.criteria?.roe_ok ? 'PASS (> 10%)' : 'FAIL (<= 10%)'}
                </div>
              </div>
              <div className="metric-card" style={{
                borderColor: screenResult.criteria?.growth_ok ? '#4caf50' : '#f44336',
                borderWidth: '1px', borderStyle: 'solid',
              }}>
                <div className="metric-label">Revenue Growth</div>
                <div className="metric-value">{screenResult.revenue_growth}%</div>
                <div style={{ color: screenResult.criteria?.growth_ok ? '#4caf50' : '#f44336', fontSize: '0.8rem' }}>
                  {screenResult.criteria?.growth_ok ? 'PASS (> 0%)' : 'FAIL (<= 0%)'}
                </div>
              </div>
            </div>

            {/* WCR if available */}
            {screenResult.wcr_pct !== null && (
              <div className="metric-card" style={{ marginBottom: '1rem' }}>
                <div className="metric-label">Working Capital Requirement</div>
                <div className="metric-value">{screenResult.wcr_pct}% of turnover</div>
                <div style={{
                  color: screenResult.wcr_pct < 20 ? '#4caf50' : screenResult.wcr_pct < 25 ? '#ff9800' : '#f44336',
                  fontSize: '0.8rem',
                }}>
                  {screenResult.wcr_pct < 20 ? 'Excellent positioning' :
                   screenResult.wcr_pct < 25 ? 'Normal range' : 'Warning - weak negotiating power'}
                </div>
              </div>
            )}

            {/* Reasons & Warnings */}
            {screenResult.reasons?.length > 0 && (
              <div style={{ marginBottom: '0.5rem' }}>
                {screenResult.reasons.map((r, i) => (
                  <div key={i} style={{ color: '#4caf50', fontSize: '0.85rem', padding: '0.2rem 0' }}>
                    + {r}
                  </div>
                ))}
              </div>
            )}
            {screenResult.warnings?.length > 0 && (
              <div>
                {screenResult.warnings.map((w, i) => (
                  <div key={i} style={{ color: '#ff9800', fontSize: '0.85rem', padding: '0.2rem 0' }}>
                    ! {w}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Exit Schedule */}
      <div className="panel">
        <h3 className="panel-title">Exit Schedule</h3>
        {exitSchedule ? (
          <div>
            <div style={{
              padding: '0.75rem', background: '#1a1a2e', borderRadius: '6px',
              marginBottom: '1rem',
              borderLeft: `3px solid ${
                exitSchedule.action === 'HOLD' ? '#4caf50' :
                exitSchedule.action === 'FULL_EXIT' ? '#f44336' : '#ff9800'
              }`,
            }}>
              <div style={{ fontSize: '1.1rem', fontWeight: 'bold', color: '#fff' }}>
                {exitSchedule.action}
              </div>
              <div style={{ color: '#aaa', fontSize: '0.85rem' }}>
                Keep {exitSchedule.position_to_keep}% | Sell {exitSchedule.position_to_sell}%
              </div>
            </div>
            {exitSchedule.schedule?.length > 0 && (
              <div>
                {exitSchedule.schedule.map((step, i) => (
                  <div key={i} style={{
                    display: 'flex', justifyContent: 'space-between',
                    padding: '0.4rem 0.5rem', fontSize: '0.8rem',
                    background: step.status === 'current' ? '#1a1a2e' :
                               step.status === 'executed' ? '#0a0a1e' : 'transparent',
                    borderRadius: '4px', marginBottom: '2px',
                    color: step.status === 'executed' ? '#666' :
                           step.status === 'current' ? '#fff' : '#888',
                  }}>
                    <span>PE {step.pe}</span>
                    <span>Sell {step.sell_pct}%</span>
                    <span>Keep {step.remaining_pct}%</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div style={{ color: '#666', textAlign: 'center', padding: '2rem' }}>
            Screen a stock to see its exit schedule
          </div>
        )}
      </div>

      {/* Manual Score Calculator */}
      <div className="panel">
        <h3 className="panel-title">Manual Score Calculator</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <label style={{ color: '#888', fontSize: '0.8rem' }}>
            PE Ratio
            <input type="number" value={manualPE} onChange={(e) => setManualPE(e.target.value)}
              style={{ width: '100%', padding: '0.4rem', background: '#1a1a2e', border: '1px solid #333', color: '#fff', borderRadius: '4px' }}
            />
          </label>
          <label style={{ color: '#888', fontSize: '0.8rem' }}>
            ROE (%)
            <input type="number" value={manualROE} onChange={(e) => setManualROE(e.target.value)}
              style={{ width: '100%', padding: '0.4rem', background: '#1a1a2e', border: '1px solid #333', color: '#fff', borderRadius: '4px' }}
            />
          </label>
          <label style={{ color: '#888', fontSize: '0.8rem' }}>
            Revenue Growth (%)
            <input type="number" value={manualGrowth} onChange={(e) => setManualGrowth(e.target.value)}
              style={{ width: '100%', padding: '0.4rem', background: '#1a1a2e', border: '1px solid #333', color: '#fff', borderRadius: '4px' }}
            />
          </label>
          <button onClick={handleManualScore} className="btn-primary" style={{ marginTop: '0.5rem' }}>
            Calculate Score
          </button>
        </div>

        {manualResult && (
          <div style={{
            marginTop: '1rem', padding: '0.75rem',
            background: '#1a1a2e', borderRadius: '6px',
            borderLeft: `3px solid ${SIGNAL_COLORS[manualResult.signal] || '#666'}`,
          }}>
            <div style={{ color: SIGNAL_COLORS[manualResult.signal], fontWeight: 'bold' }}>
              {SIGNAL_LABELS[manualResult.signal]} - Score: {manualResult.score}/100
            </div>
            {manualResult.reasons?.map((r, i) => (
              <div key={i} style={{ color: '#4caf50', fontSize: '0.8rem', marginTop: '0.3rem' }}>+ {r}</div>
            ))}
            {manualResult.warnings?.map((w, i) => (
              <div key={i} style={{ color: '#ff9800', fontSize: '0.8rem', marginTop: '0.3rem' }}>! {w}</div>
            ))}
          </div>
        )}
      </div>

      {/* Position Sizing Calculator */}
      <div className="panel">
        <h3 className="panel-title">Position Sizing</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <label style={{ color: '#888', fontSize: '0.8rem' }}>
            Portfolio Value
            <input type="number" value={portfolioValue} onChange={(e) => setPortfolioValue(e.target.value)}
              style={{ width: '100%', padding: '0.4rem', background: '#1a1a2e', border: '1px solid #333', color: '#fff', borderRadius: '4px' }}
            />
          </label>
          <label style={{ color: '#888', fontSize: '0.8rem' }}>
            Stock Price
            <input type="number" value={stockPrice} onChange={(e) => setStockPrice(e.target.value)}
              style={{ width: '100%', padding: '0.4rem', background: '#1a1a2e', border: '1px solid #333', color: '#fff', borderRadius: '4px' }}
            />
          </label>
          <label style={{ color: '#888', fontSize: '0.8rem' }}>
            Avg Daily Volume (currency)
            <input type="number" value={dailyVolume} onChange={(e) => setDailyVolume(e.target.value)}
              style={{ width: '100%', padding: '0.4rem', background: '#1a1a2e', border: '1px solid #333', color: '#fff', borderRadius: '4px' }}
            />
          </label>
          <button onClick={handlePositionSize} className="btn-primary" style={{ marginTop: '0.5rem' }}>
            Calculate Size
          </button>
        </div>

        {positionResult && (
          <div style={{ marginTop: '1rem' }}>
            <div className="metric-card">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.8rem' }}>
                <div>
                  <div style={{ color: '#888' }}>Target Weight</div>
                  <div style={{ color: '#fff' }}>{positionResult.target_weight_pct}%</div>
                </div>
                <div>
                  <div style={{ color: '#888' }}>Full Position</div>
                  <div style={{ color: '#fff' }}>{positionResult.full_position_value?.toLocaleString()}</div>
                </div>
                <div>
                  <div style={{ color: '#888' }}>Per Tranche</div>
                  <div style={{ color: '#fff' }}>{positionResult.tranche_value?.toLocaleString()}</div>
                </div>
                <div>
                  <div style={{ color: '#888' }}>Shares/Tranche</div>
                  <div style={{ color: '#fff' }}>{positionResult.tranche_shares}</div>
                </div>
                <div>
                  <div style={{ color: '#888' }}>Max Daily Buy</div>
                  <div style={{ color: '#fff' }}>{positionResult.max_daily_purchase?.toLocaleString()}</div>
                </div>
                <div>
                  <div style={{ color: '#888' }}>Days to Full</div>
                  <div style={{ color: '#fff' }}>{positionResult.days_to_full_position} days</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Key Quotes */}
      {method && (
        <div className="panel full-width">
          <h3 className="panel-title">Igon's Principles</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            {method.key_quotes?.map((q, i) => (
              <div key={i} style={{
                padding: '0.75rem', background: '#1a1a2e', borderRadius: '6px',
                borderLeft: '3px solid #6366f1', color: '#ccc', fontSize: '0.85rem',
                fontStyle: 'italic',
              }}>
                "{q}"
              </div>
            ))}
            {method.avoid?.map((a, i) => (
              <div key={i} style={{
                padding: '0.75rem', background: '#1a1a2e', borderRadius: '6px',
                borderLeft: '3px solid #f44336', color: '#999', fontSize: '0.85rem',
              }}>
                Avoid: {a}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
