"""
╔══════════════════════════════════════════════════════════════╗
║  NEXUS•AI — Claude Intelligence Layer                       ║
║  Uses Anthropic's Claude to interpret ML signals,           ║
║  sentiment data, and provide strategic analysis             ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import json
import logging
from typing import Dict, Optional
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class ClaudeAnalyst:
    """
    Claude-powered financial analyst that interprets:
    - ML model signals (DNN, RNN, DQL outputs)
    - Social sentiment data (Reddit, Twitter, 4chan)
    - Technical indicators and risk metrics
    - Market context and macro environment
    """

    SYSTEM_PROMPT = """You are NEXUS•AI, an advanced artificial financial intelligence system.
You analyze market data, ML model outputs, and social media sentiment to provide
actionable trading intelligence.

Your analysis style:
- Be precise and quantitative. Use numbers, percentages, probabilities.
- Acknowledge uncertainty. Markets are inherently unpredictable.
- Distinguish between signal and noise in sentiment data.
- Consider the source reliability: Reddit/WSB is retail-heavy and momentum-driven,
  Twitter has faster information flow, 4chan /biz/ is contrarian and speculative.
- Reference specific model outputs (DNN accuracy, RNN sequence patterns, DQL Q-values).
- Always include risk warnings. Never frame anything as guaranteed.
- Be concise. Trading decisions require clarity, not verbosity.

CRITICAL: You are an analytical tool, NOT a financial advisor. All outputs are
for informational and educational purposes only. Always remind users that past
performance does not predict future results and that they should consult qualified
professionals before making investment decisions.

Format responses in clear sections with key metrics highlighted."""

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set — Claude analysis disabled")
            self.client = None
            return
        self.client = Anthropic(api_key=api_key)
        logger.info("Claude analyst initialized")

    def analyze(self, market_data: Dict, model_signals: Dict,
                sentiment_data: Dict, risk_metrics: Dict) -> str:
        """
        Generate comprehensive market analysis using Claude.
        Combines ML signals, sentiment, and risk data.
        """
        if not self.client:
            return self._fallback_analysis(model_signals, sentiment_data)

        prompt = self._build_analysis_prompt(
            market_data, model_signals, sentiment_data, risk_metrics
        )

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return self._fallback_analysis(model_signals, sentiment_data)

    def chat(self, user_message: str, context: Dict) -> str:
        """
        Interactive chat about market conditions.
        Users can ask questions about the current analysis.
        """
        if not self.client:
            return "Claude API is not configured. Please add your ANTHROPIC_API_KEY to the .env file."

        context_summary = json.dumps(context, indent=2, default=str)

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800,
                system=f"""{self.SYSTEM_PROMPT}

Current system context:
{context_summary}""",
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude chat error: {e}")
            return f"Analysis temporarily unavailable: {str(e)}"

    def _build_analysis_prompt(self, market_data: Dict, model_signals: Dict,
                                sentiment_data: Dict, risk_metrics: Dict) -> str:
        """Construct the analysis prompt with all available data."""
        return f"""Analyze the current market conditions based on the following data:

## ML Model Signals
{json.dumps(model_signals, indent=2, default=str)}

## Social Sentiment Analysis
- Overall sentiment: {sentiment_data.get('overall_sentiment', 'N/A')}
- Label: {sentiment_data.get('sentiment_label', 'N/A')}
- Total posts analyzed: {sentiment_data.get('total_posts', 0)}
- Source breakdown: {json.dumps(sentiment_data.get('sources', {}), indent=2)}
- Top mentioned tickers: {json.dumps(dict(list(sentiment_data.get('top_tickers', {}).items())[:10]), indent=2)}

## Risk Metrics
{json.dumps(risk_metrics, indent=2, default=str)}

## Market Data Summary
{json.dumps(market_data, indent=2, default=str)}

Provide:
1. SIGNAL SYNTHESIS: What do the combined ML models suggest?
2. SENTIMENT READING: What is the crowd saying and how reliable is it?
3. RISK ASSESSMENT: Key risks to watch right now.
4. ACTIONABLE INSIGHT: What would a prudent trader consider?
5. CONFIDENCE LEVEL: How much should one trust this analysis? (Low/Medium/High)

Remember: This is for educational purposes only, not financial advice."""

    def _fallback_analysis(self, model_signals: Dict, sentiment_data: Dict) -> str:
        """Basic analysis when Claude API is unavailable."""
        signal = model_signals.get("ensemble", {}).get("signal", "NEUTRAL")
        confidence = model_signals.get("ensemble", {}).get("confidence", 0)
        sentiment = sentiment_data.get("overall_sentiment", 0)
        sentiment_label = sentiment_data.get("sentiment_label", "NEUTRAL")

        return f"""## NEXUS•AI Analysis (Offline Mode)

**Ensemble Signal:** {signal} (confidence: {confidence:.1%})
**Social Sentiment:** {sentiment_label} (score: {sentiment:.4f})

**Note:** Claude AI analysis is offline. Configure ANTHROPIC_API_KEY for
full natural language analysis combining ML signals with sentiment interpretation.

This is for educational/informational purposes only — not financial advice."""
