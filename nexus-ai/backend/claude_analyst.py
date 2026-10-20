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
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import anthropic

logger = logging.getLogger(__name__)

# Override with the ANTHROPIC_MODEL environment variable, e.g. a Sonnet model
# for lower cost per call.
DEFAULT_MODEL = "claude-opus-5-5"

# Server-side refusal fallback: if a safety classifier declines a request,
# the API re-runs it on Anthropic's recommended fallback model within the
# same call instead of returning the refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class AnalystResult:
    """
    Outcome of one Claude call.

    llm_used is False whenever the text did not come from the model, so the
    API can never present a canned summary as Claude's analysis.
    """
    text: str
    llm_used: bool
    model: Optional[str] = None
    error: Optional[str] = None
    stop_reason: Optional[str] = None
    usage: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class ClaudeAnalyst:
    """
    Claude-powered financial analyst that interprets:
    - ML model signals (DNN, RNN, DQL outputs) with their out-of-sample scores
    - Social sentiment data (Reddit, Twitter/X, 4chan /biz/)
    - Technical indicators and risk metrics
    """

    SYSTEM_PROMPT = """You are NEXUS•AI, an analytical assistant inside an educational
financial-analysis app. You interpret market data, machine-learning model outputs and
social-media sentiment for the user.

How to read the model data you are given:
- Each model's accuracy is measured out of sample, on sessions it never trained on.
- Each accuracy comes with a majority-class baseline (always predicting the most common
  direction) and a p-value from a one-sided McNemar test against that baseline.
- A model whose edge is not significant (p >= 0.05) has shown no predictive skill. Say
  so plainly, and do not treat its signal as evidence, whatever its raw accuracy.
- If the ensemble carries a no-skill warning, the combined signal is not evidence either.

How to read sentiment:
- Sentiment scores come from a finance-tuned VADER lexicon, not from a trained model.
- Weigh sources by reliability: Reddit and WSB are retail-heavy and momentum-driven, X
  moves fast, 4chan /biz/ is contrarian and speculative.
- A post count of zero means no data was collected, not a neutral market.

Style: be quantitative and concise, state your uncertainty, separate signal from noise,
and never present anything as guaranteed. You are an analytical tool, not a financial
adviser: say that the output is educational, that past performance does not predict
future results, and that investment decisions call for a qualified professional.

Format the answer in short sections with the key figures highlighted."""

    # Current Opus models always think before answering, and thinking tokens
    # count toward max_tokens. The budget must cover the reasoning and the
    # answer; 1,000 tokens could be spent on reasoning alone.
    MAX_TOKENS = 16000
    ANALYSIS_EFFORT = "high"    # one deliberate synthesis per click
    CHAT_EFFORT = "medium"      # interactive questions

    def __init__(self, client=None):
        self.model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
        if client is not None:
            self.client = client
            return
        if not os.getenv("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY not set — Claude analysis disabled")
            self.client = None
            return
        self.client = anthropic.Anthropic()
        logger.info(f"Claude analyst initialized ({self.model})")

    @property
    def configured(self) -> bool:
        return self.client is not None

    def analyze(self, market_data: Dict, model_signals: Dict,
                sentiment_data: Dict, risk_metrics: Dict) -> AnalystResult:
        """
        Generate a market analysis combining ML signals, sentiment and risk.

        Without an API key this returns a plain summary of the same data with
        llm_used=False, labelled as such in its text.
        """
        if not self.configured:
            return AnalystResult(
                text=self._offline_summary(model_signals, sentiment_data),
                llm_used=False,
                error="ANTHROPIC_API_KEY is not set",
            )
        prompt = self._build_analysis_prompt(
            market_data, model_signals, sentiment_data, risk_metrics
        )
        return self._complete(self.SYSTEM_PROMPT, prompt, self.ANALYSIS_EFFORT)

    def chat(self, user_message: str, context: Dict) -> AnalystResult:
        """Interactive Q&A about the current models, sentiment and risk."""
        if not self.configured:
            return AnalystResult(
                text="",
                llm_used=False,
                error="ANTHROPIC_API_KEY is not set. Add it to backend/.env and restart.",
            )
        system = (
            f"{self.SYSTEM_PROMPT}\n\nCurrent system context:\n"
            f"{json.dumps(context, indent=2, default=str)}"
        )
        return self._complete(system, user_message, self.CHAT_EFFORT)

    def _complete(self, system: str, user_content: str, effort: str) -> AnalystResult:
        """
        One streamed request. Streaming with get_final_message() avoids HTTP
        timeouts on long answers while still returning a single message.
        """
        try:
            with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=self.MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user_content}],
                output_config={"effort": effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                message = stream.get_final_message()
        # Most specific first: each failure gets an actionable message, and
        # retryable failures (429, 5xx, network) stay distinguishable from
        # ones a retry cannot fix (bad key, unknown model, invalid request).
        except anthropic.AuthenticationError:
            return self._failure("Anthropic rejected the API key (401). Check ANTHROPIC_API_KEY.")
        except anthropic.PermissionDeniedError:
            return self._failure(f"This API key cannot use {self.model} (403).")
        except anthropic.NotFoundError:
            return self._failure(
                f"Model {self.model} was not found (404). Set ANTHROPIC_MODEL to a current model."
            )
        except anthropic.RateLimitError:
            return self._failure("Rate limited by the Anthropic API (429). Try again shortly.")
        except anthropic.BadRequestError as e:
            return self._failure(f"The Anthropic API rejected the request (400): {e.message}")
        except anthropic.APIStatusError as e:
            return self._failure(f"The Anthropic API returned an error ({e.status_code}). Try again later.")
        except anthropic.APITimeoutError:
            return self._failure("The Anthropic API timed out. Try again.")
        except anthropic.APIConnectionError:
            return self._failure("Could not reach the Anthropic API. Check the network connection.")

        usage = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        }
        logger.info(
            f"Claude call: model={message.model} stop={message.stop_reason} "
            f"in={usage['input_tokens']} out={usage['output_tokens']} "
            f"request_id={getattr(message, '_request_id', None)}"
        )

        # A refusal must be handled before reading content; any partial text
        # from a declined turn is discarded.
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            return AnalystResult(
                text="",
                llm_used=True,
                model=message.model,
                error="Claude declined this request" + (f" ({category})." if category else "."),
                stop_reason="refusal",
                usage=usage,
            )

        # Responses interleave thinking blocks and text blocks; only the text
        # blocks are the answer. content[0] is usually a thinking block.
        text = "".join(b.text for b in message.content if b.type == "text").strip()
        if message.stop_reason == "max_tokens":
            text += "\n\n[Response truncated: the output token limit was reached.]"
        return AnalystResult(
            text=text,
            llm_used=True,
            model=message.model,
            stop_reason=message.stop_reason,
            usage=usage,
        )

    def _failure(self, error: str) -> AnalystResult:
        logger.error(f"Claude call failed: {error}")
        return AnalystResult(text="", llm_used=False, model=self.model, error=error)

    def _build_analysis_prompt(self, market_data: Dict, model_signals: Dict,
                               sentiment_data: Dict, risk_metrics: Dict) -> str:
        """Construct the analysis prompt with all available data."""
        top_tickers = dict(list(sentiment_data.get("top_tickers", {}).items())[:10])
        return f"""Analyze the current market conditions based on the following data:

## ML Model Signals (out-of-sample scores, baselines and p-values)
{json.dumps(model_signals, indent=2, default=str)}

## Social Sentiment Analysis
- Overall sentiment: {sentiment_data.get('overall_sentiment', 'N/A')}
- Label: {sentiment_data.get('sentiment_label', 'N/A')}
- Total posts analyzed: {sentiment_data.get('total_posts', 0)}
- Source breakdown: {json.dumps(sentiment_data.get('sources', {}), indent=2)}
- Top mentioned tickers: {json.dumps(top_tickers, indent=2)}

## Risk Metrics
{json.dumps(risk_metrics, indent=2, default=str)}

## Market Data Summary
{json.dumps(market_data, indent=2, default=str)}

Provide:
1. SIGNAL SYNTHESIS: What do the models suggest, given which of them show significant skill?
2. SENTIMENT READING: What is the crowd saying, and how reliable is it?
3. RISK ASSESSMENT: Key risks to watch right now.
4. ACTIONABLE INSIGHT: What would a prudent trader consider?
5. CONFIDENCE LEVEL: How much should one trust this analysis? (Low/Medium/High)

Remember: this is for educational purposes only, not financial advice."""

    def _offline_summary(self, model_signals: Dict, sentiment_data: Dict) -> str:
        """Plain summary of the same inputs, used when no API key is set."""
        lines = ["## NEXUS•AI Summary (Claude offline: no AI analysis)", ""]
        for name in ("dnn", "rnn", "dql"):
            m = model_signals.get(name) or {}
            if m.get("accuracy") is None:
                continue
            verdict = "significant edge" if m.get("significant") else "no significant edge"
            lines.append(
                f"- **{m.get('name', name.upper())}:** {m['accuracy']}% out of sample vs "
                f"{m.get('baseline_accuracy')}% baseline ({verdict}, p = {m.get('p_value')})"
            )
        if sentiment_data.get("total_posts", 0) > 0:
            lines.append(
                f"- **Social sentiment:** {sentiment_data.get('sentiment_label', 'N/A')} "
                f"(score {sentiment_data.get('overall_sentiment', 0):.4f}, "
                f"{sentiment_data['total_posts']} posts)"
            )
        else:
            lines.append("- **Social sentiment:** no posts collected")
        lines += [
            "",
            "Set ANTHROPIC_API_KEY to get Claude's written analysis of these figures.",
            "",
            "Educational use only, not financial advice.",
        ]
        return "\n".join(lines)
