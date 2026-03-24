import React from 'react';

export default function Models({ health, prediction, training, onTrain }) {
  const models = health?.models || {};

  const modelCards = [
    {
      key: 'dnn',
      name: 'DNN Predictor',
      desc: 'Dense Neural Network - 3 hidden layers with dropout',
      arch: 'Dense(64) - Dense(64) - Dense(32) - Dense(1)',
      chapter: 'Chapter 7',
    },
    {
      key: 'rnn',
      name: 'RNN Sequencer',
      desc: 'LSTM Recurrent Network - temporal sequence patterns',
      arch: 'LSTM(64, seq=20) - LSTM(32) - Dense(1)',
      chapter: 'Chapter 8',
    },
    {
      key: 'dql',
      name: 'DQL Agent',
      desc: 'Deep Q-Learning - reinforcement learning trading agent',
      arch: 'Dense(64) - Dense(64) - Dense(3) [Q-values]',
      chapter: 'Chapter 9',
    },
  ];

  return (
    <div className="models-page">
      <div className="page-header">
        <h2>AI Model Ensemble</h2>
        <button
          className="btn-primary"
          onClick={() => onTrain({})}
          disabled={training}
        >
          {training ? 'Training...' : 'Train All Models'}
        </button>
      </div>

      <div className="model-grid">
        {modelCards.map(card => {
          const model = models[card.key];
          const pred = prediction?.models?.[card.key];
          const isActive = model?.status === 'active';

          return (
            <div key={card.key} className={`model-card ${isActive ? 'active' : ''}`}>
              <div className="model-card-header">
                <h3>{card.name}</h3>
                <span className={`badge ${isActive ? 'trained' : 'untrained'}`}>
                  {model?.status || 'untrained'}
                </span>
              </div>
              <p className="model-desc">{card.desc}</p>
              <div className="model-detail">
                <span className="label">Architecture:</span>
                <code>{card.arch}</code>
              </div>
              <div className="model-detail">
                <span className="label">Source:</span>
                <span>Hilpisch (2021) {card.chapter}</span>
              </div>
              {isActive && (
                <>
                  <div className="model-metrics">
                    <div className="model-metric">
                      <span className="metric-label">Accuracy</span>
                      <span className="metric-value">{model.accuracy}%</span>
                    </div>
                    {pred && (
                      <>
                        <div className="model-metric">
                          <span className="metric-label">Signal</span>
                          <span className={`metric-value ${
                            (pred.direction === 'LONG' || pred.action === 'BUY') ? 'green' :
                            (pred.direction === 'SHORT' || pred.action === 'SELL') ? 'red' : ''
                          }`}>
                            {pred.direction || pred.action || 'N/A'}
                          </span>
                        </div>
                        <div className="model-metric">
                          <span className="metric-label">Confidence</span>
                          <span className="metric-value">
                            {(pred.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </>
                    )}
                  </div>
                  {card.key === 'dql' && model.epsilon !== undefined && (
                    <div className="model-extra">
                      <span>Epsilon: {model.epsilon}</span>
                      <span>Memory: {model.memory_size} experiences</span>
                      <span>Total Reward: {model.total_reward}</span>
                    </div>
                  )}
                </>
              )}
            </div>
          );
        })}
      </div>

      {prediction?.ensemble && (
        <div className="ensemble-card">
          <h3>Ensemble Meta-Learner</h3>
          <div className="ensemble-grid">
            <div className="ensemble-metric">
              <span className="label">Combined Signal</span>
              <span className={`value large ${
                prediction.ensemble.signal === 'LONG' ? 'green' :
                prediction.ensemble.signal === 'SHORT' ? 'red' : ''
              }`}>
                {prediction.ensemble.signal}
              </span>
            </div>
            <div className="ensemble-metric">
              <span className="label">Confidence</span>
              <span className="value large">
                {(prediction.ensemble.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="ensemble-metric">
              <span className="label">Combined Score</span>
              <span className="value">{prediction.ensemble.combined_score}</span>
            </div>
          </div>
          {prediction.ensemble.components && (
            <div className="component-scores">
              <h4>Component Contributions</h4>
              <div className="score-bars">
                {Object.entries(prediction.ensemble.components).map(([key, val]) => (
                  <div key={key} className="score-bar-row">
                    <span className="bar-label">{key.toUpperCase()}</span>
                    <div className="bar-track">
                      <div
                        className={`bar-fill ${val > 0 ? 'positive' : 'negative'}`}
                        style={{
                          width: `${Math.min(Math.abs(val) * 100, 100)}%`,
                          marginLeft: val < 0 ? 'auto' : undefined,
                        }}
                      />
                    </div>
                    <span className="bar-value">{val}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
