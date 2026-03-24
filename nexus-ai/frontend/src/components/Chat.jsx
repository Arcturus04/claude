import React, { useState, useRef, useEffect } from 'react';
import { api } from '../api';

export default function Chat() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Welcome to NEXUS AI. I can analyze market conditions, interpret ML model signals, explain sentiment data, and discuss trading strategies. What would you like to know?',
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const data = await api.chat(input);
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${e.message}. Make sure ANTHROPIC_API_KEY is set in your .env file.`,
      }]);
    } finally {
      setLoading(false);
    }
  };

  const runAnalysis = async () => {
    setAnalysisLoading(true);
    try {
      const data = await api.analyze();
      setMessages(prev => [...prev,
        { role: 'user', content: '[Requested full market analysis]' },
        { role: 'assistant', content: data.analysis },
      ]);
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Analysis failed: ${e.message}`,
      }]);
    } finally {
      setAnalysisLoading(false);
    }
  };

  return (
    <div className="chat-page">
      <div className="page-header">
        <h2>Claude Analyst</h2>
        <button
          className="btn-primary"
          onClick={runAnalysis}
          disabled={analysisLoading}
        >
          {analysisLoading ? 'Analyzing...' : 'Full Market Analysis'}
        </button>
      </div>

      <div className="chat-container" ref={scrollRef}>
        {messages.map((msg, i) => (
          <div key={i} className={`chat-message ${msg.role}`}>
            <div className="message-header">
              {msg.role === 'assistant' ? 'NEXUS AI' : 'You'}
            </div>
            <div className="message-content">
              {msg.content.split('\n').map((line, j) => (
                <React.Fragment key={j}>
                  {line}
                  {j < msg.content.split('\n').length - 1 && <br />}
                </React.Fragment>
              ))}
            </div>
          </div>
        ))}
        {loading && (
          <div className="chat-message assistant">
            <div className="message-header">NEXUS AI</div>
            <div className="message-content typing">Analyzing...</div>
          </div>
        )}
      </div>

      <div className="chat-input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Ask about markets, models, sentiment, or trading strategies..."
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>

      <div className="chat-suggestions">
        <span className="label">Try:</span>
        {[
          'What does the Reddit sentiment suggest about NVDA?',
          'Explain the DQL agent trading logic',
          'What are the main risks right now?',
          'Compare the DNN and RNN predictions',
        ].map((s, i) => (
          <button
            key={i}
            className="suggestion"
            onClick={() => { setInput(s); }}
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
