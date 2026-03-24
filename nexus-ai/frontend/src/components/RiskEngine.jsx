import React from 'react';

export default function RiskEngine({ riskData, onRefresh }) {
  const metrics = [
    {
      key: 'var_95',
      label: 'Value at Risk (95%)',
      suffix: '%',
      warn: (v) => v < -2,
      desc: 'Maximum expected loss at 95% confidence',
    },
    {
      key: 'cvar_95',
      label: 'Conditional VaR (95%)',
      suffix: '%',
      warn: (v) => v < -3,
      desc: 'Expected loss in the worst 5% of scenarios',
    },
    {
      key: 'max_drawdown',
      label: 'Maximum Drawdown',
      suffix: '%',
      warn: (v) => v < -15,
      desc: 'Largest peak-to-trough decline',
    },
    {
      key: 'volatility_annual',
      label: 'Annual Volatility',
      suffix: '%',
      warn: (v) => v > 25,
      desc: 'Annualized standard deviation of returns',
    },
    {
      key: 'skewness',
      label: 'Skewness',
      suffix: '',
      warn: (v) => v < -0.3,
      desc: 'Negative = more left-tail risk (crashes)',
    },
    {
      key: 'kurtosis',
      label: 'Kurtosis',
      suffix: '',
      warn: (v) => v > 3,
      desc: 'Higher = fatter tails than normal distribution',
    },
  ];

  return (
    <div className="risk-page">
      <div className="page-header">
        <h2>Risk Engine</h2>
        <button className="btn-primary" onClick={onRefresh}>
          Calculate Risk Metrics
        </button>
      </div>

      <div className="risk-intro">
        <p>
          The book's central thesis: financial returns are <strong>not normally distributed</strong>.
          They exhibit negative skewness (more crashes than booms) and excess kurtosis
          (fat tails - extreme events happen more often than Gaussian models predict).
          This invalidates traditional models like CAPM and Black-Scholes.
        </p>
      </div>

      {riskData ? (
        <>
          <div className="risk-grid">
            {metrics.map(m => {
              const val = riskData[m.key];
              if (val === undefined || val === null) return null;
              const isWarning = m.warn(val);
              return (
                <div key={m.key} className={`risk-card ${isWarning ? 'warning' : 'ok'}`}>
                  <div className="risk-card-header">
                    <span className="risk-label">{m.label}</span>
                    <span className={`risk-indicator ${isWarning ? 'alert' : 'safe'}`}>
                      {isWarning ? 'Warning' : 'Normal'}
                    </span>
                  </div>
                  <div className="risk-value">
                    {typeof val === 'number' ? val.toFixed(4) : val}{m.suffix}
                  </div>
                  <div className="risk-desc">{m.desc}</div>
                </div>
              );
            })}
          </div>

          {riskData.atr_14d !== undefined && riskData.atr_14d !== null && (
            <div className="risk-card standalone">
              <div className="risk-card-header">
                <span className="risk-label">ATR (14-day)</span>
              </div>
              <div className="risk-value">{riskData.atr_14d.toFixed(4)}%</div>
              <div className="risk-desc">Average True Range - measures daily volatility</div>
            </div>
          )}

          {riskData.normality_tests && (
            <div className="normality-section">
              <h3>Non-Normality Diagnostics</h3>
              <p className="muted">
                All tests check if returns follow a normal (Gaussian) distribution.
                p-values below 0.05 reject normality.
              </p>
              <div className="normality-grid">
                {Object.entries(riskData.normality_tests).map(([test, pval]) => {
                  const rejected = pval < 0.05;
                  return (
                    <div key={test} className={`normality-card ${rejected ? 'rejected' : ''}`}>
                      <div className="test-name">
                        {test.replace(/_/g, ' ').replace('pvalue', '')}
                      </div>
                      <div className="test-pval">p = {pval.toFixed(6)}</div>
                      <div className={`test-result ${rejected ? 'red' : 'green'}`}>
                        {rejected ? 'REJECTED - Not Normal' : 'Cannot Reject'}
                      </div>
                    </div>
                  );
                })}
              </div>
              <p className="warning-text">
                All normality tests are expected to be rejected for financial returns.
                This confirms that standard Gaussian risk models underestimate tail risk.
                - From Hilpisch, "Artificial Intelligence in Finance" (2021)
              </p>
            </div>
          )}
        </>
      ) : (
        <div className="empty-state">
          <p>No risk data available. Train the models first, then calculate risk metrics.</p>
        </div>
      )}
    </div>
  );
}
