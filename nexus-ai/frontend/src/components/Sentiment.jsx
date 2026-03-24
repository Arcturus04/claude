import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell,
} from 'recharts';

const COLORS = ['#00d4ff', '#6c5ce7', '#ff6b6b', '#ffd93d', '#51cf66', '#ff922b'];

export default function Sentiment({ data, onScrape }) {
  const sources = data?.sources || {};
  const tickers = data?.top_tickers || {};

  const tickerChart = Object.entries(tickers)
    .slice(0, 10)
    .map(([ticker, info]) => ({
      ticker,
      mentions: info.mentions,
      sentiment: info.avg_sentiment,
      bullish: info.bullish_pct,
    }));

  const sourceChart = Object.entries(sources).map(([name, info]) => ({
    name,
    posts: info.count,
    sentiment: info.avg_sentiment,
    bullish: info.bullish_pct,
    bearish: info.bearish_pct,
  }));

  return (
    <div className="sentiment-page">
      <div className="page-header">
        <h2>Social Sentiment Intelligence</h2>
        <button className="btn-primary" onClick={onScrape}>
          Scrape All Sources
        </button>
      </div>

      <div className="source-badges">
        <span className="source-badge reddit">Reddit (WSB, stocks, investing +6 more)</span>
        <span className="source-badge twitter">Twitter/X (financial tweets)</span>
        <span className="source-badge chan">4chan /biz/ (contrarian signals)</span>
      </div>

      {data && data.total_posts > 0 ? (
        <>
          <div className="sentiment-overview">
            <div className="sentiment-main">
              <div className="sentiment-score-large">
                <span className="label">Overall Sentiment</span>
                <span className={`value ${
                  data.overall_sentiment > 0.1 ? 'green' :
                  data.overall_sentiment < -0.1 ? 'red' : ''
                }`}>
                  {data.overall_sentiment?.toFixed(4)}
                </span>
                <span className={`sentiment-label ${
                  data.sentiment_label === 'BULLISH' ? 'green' :
                  data.sentiment_label === 'BEARISH' ? 'red' : ''
                }`}>
                  {data.sentiment_label}
                </span>
              </div>
              <div className="sentiment-meta">
                <span>{data.total_posts} posts analyzed</span>
                <span>Last updated: {new Date(data.timestamp).toLocaleString()}</span>
              </div>
            </div>
          </div>

          <div className="two-col">
            <div className="chart-section">
              <h3>Source Breakdown</h3>
              {sourceChart.length > 0 && (
                <div className="source-cards">
                  {sourceChart.map(s => (
                    <div key={s.name} className="source-card">
                      <div className="source-name">{s.name}</div>
                      <div className="source-stats">
                        <span>{s.posts} posts</span>
                        <span className={s.sentiment > 0 ? 'green' : 'red'}>
                          Sentiment: {s.sentiment.toFixed(4)}
                        </span>
                      </div>
                      <div className="sentiment-bar">
                        <div
                          className="bar-bullish"
                          style={{ width: `${s.bullish}%` }}
                          title={`Bullish: ${s.bullish}%`}
                        />
                        <div
                          className="bar-bearish"
                          style={{ width: `${s.bearish}%` }}
                          title={`Bearish: ${s.bearish}%`}
                        />
                      </div>
                      <div className="bar-labels">
                        <span className="green">{s.bullish}% bullish</span>
                        <span className="red">{s.bearish}% bearish</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="chart-section">
              <h3>Top Mentioned Tickers</h3>
              {tickerChart.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={tickerChart} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis type="number" stroke="#888" />
                    <YAxis dataKey="ticker" type="category" stroke="#888" width={50} />
                    <Tooltip
                      contentStyle={{ background: '#1a1a2e', border: '1px solid #333', color: '#fff' }}
                      formatter={(v, name) => {
                        if (name === 'mentions') return [v, 'Mentions'];
                        return [v.toFixed(4), 'Sentiment'];
                      }}
                    />
                    <Bar dataKey="mentions" fill="#00d4ff" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="muted">No ticker data available</p>
              )}
            </div>
          </div>

          {data.posts && data.posts.length > 0 && (
            <div className="posts-section">
              <h3>Most Significant Posts</h3>
              <div className="posts-list">
                {data.posts.slice(0, 15).map((post, i) => (
                  <div key={i} className={`post-card ${
                    post.sentiment > 0.05 ? 'bullish' :
                    post.sentiment < -0.05 ? 'bearish' : 'neutral'
                  }`}>
                    <div className="post-header">
                      <span className="post-source">{post.source}</span>
                      <span className="post-author">{post.author}</span>
                      <span className={`post-sentiment ${
                        post.sentiment > 0 ? 'green' : 'red'
                      }`}>
                        {post.sentiment.toFixed(4)}
                      </span>
                    </div>
                    <div className="post-text">{post.text.slice(0, 200)}...</div>
                    {post.tickers.length > 0 && (
                      <div className="post-tickers">
                        {post.tickers.map(t => (
                          <span key={t} className="ticker-tag">${t}</span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="empty-state">
          <p>No sentiment data yet. Click "Scrape All Sources" to gather social media sentiment.</p>
          <p className="muted">
            Requires API keys for Reddit and Twitter. 4chan works without keys.
          </p>
        </div>
      )}
    </div>
  );
}
