"""
Claude integration, tested against a fake client (no network, no API key).

Covers the request shape sent to the API, extraction of the answer from
mixed thinking/text blocks, refusals, and that every failure is reported as
a failure rather than as a canned answer.
"""

from types import SimpleNamespace

import anthropic
import httpx2
import pytest

import server
from claude_analyst import DEFAULT_MODEL, FALLBACK_BETA, ClaudeAnalyst


def block(kind, text=""):
    if kind == "thinking":
        return SimpleNamespace(type="thinking", thinking="", signature="sig")
    return SimpleNamespace(type="text", text=text)


def message(content, stop_reason="end_turn", stop_details=None):
    return SimpleNamespace(
        content=content, stop_reason=stop_reason, stop_details=stop_details,
        model=DEFAULT_MODEL,
        usage=SimpleNamespace(input_tokens=1200, output_tokens=340),
    )


class FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.msg


class FakeClient:
    """Mimics client.beta.messages.stream(...). Records every request."""

    def __init__(self, msg=None, error=None):
        self.msg, self.error, self.calls = msg, error, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return FakeStream(self.msg)


REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(cls, code):
    return cls("error", response=httpx2.Response(code, request=REQUEST), body=None)


def run_analysis(client):
    return ClaudeAnalyst(client=client).analyze({}, {}, {"total_posts": 0}, {})


def test_request_uses_a_current_model_with_room_to_think():
    client = FakeClient(message([block("text", "ok")]))
    run_analysis(client)
    req = client.calls[0]
    assert req["model"] == DEFAULT_MODEL == "claude-opus-5-5"
    assert req["max_tokens"] >= 16000  # thinking counts toward max_tokens
    assert req["output_config"] == {"effort": ClaudeAnalyst.ANALYSIS_EFFORT}
    assert req["betas"] == [FALLBACK_BETA] and req["fallbacks"] == "default"
    assert "thinking" not in req  # always on for this model; disabling it is a 400


def test_model_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    client = FakeClient(message([block("text", "ok")]))
    run_analysis(client)
    assert client.calls[0]["model"] == "claude-sonnet-5-5"


def test_answer_is_read_from_text_blocks_not_content_zero():
    # content[0] is a thinking block on current models; it has no .text.
    client = FakeClient(message([block("thinking"), block("text", "Part one. "),
                                 block("text", "Part two.")]))
    result = run_analysis(client)
    assert result.text == "Part one. Part two."
    assert result.llm_used is True and result.error is None
    assert result.usage == {"input_tokens": 1200, "output_tokens": 340}


def test_refusal_is_reported_and_partial_text_discarded():
    msg = message([block("text", "partial")], stop_reason="refusal",
                  stop_details=SimpleNamespace(category="cyber"))
    result = run_analysis(FakeClient(msg))
    assert result.text == ""
    assert result.stop_reason == "refusal" and "declined" in result.error
    assert "cyber" in result.error


def test_truncation_is_flagged():
    result = run_analysis(FakeClient(message([block("text", "half")], stop_reason="max_tokens")))
    assert result.text.startswith("half") and "truncated" in result.text


@pytest.mark.parametrize("error, phrase", [
    (status_error(anthropic.AuthenticationError, 401), "API key"),
    (status_error(anthropic.PermissionDeniedError, 403), "403"),
    (status_error(anthropic.NotFoundError, 404), "ANTHROPIC_MODEL"),
    (status_error(anthropic.RateLimitError, 429), "Rate limited"),
    (status_error(anthropic.InternalServerError, 500), "500"),
    (anthropic.APITimeoutError(request=REQUEST), "timed out"),
    (anthropic.APIConnectionError(request=REQUEST), "Could not reach"),
])
def test_every_failure_is_reported_not_disguised(error, phrase):
    result = run_analysis(FakeClient(error=error))
    assert result.llm_used is False
    assert result.text == ""  # no canned analysis in place of a failed call
    assert phrase in result.error


def test_without_a_key_the_offline_summary_is_labelled(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    analyst = ClaudeAnalyst()
    signals = {"dnn": {"name": "DNN", "accuracy": 56.0, "baseline_accuracy": 53.8,
                       "p_value": 0.41, "significant": False}}
    result = analyst.analyze({}, signals, {"total_posts": 0}, {})
    assert result.llm_used is False and "not set" in result.error
    assert "offline" in result.text and "no significant edge" in result.text
    assert "no posts collected" in result.text


# ─── HTTP layer ──────────────────────────────────────────────────────────────

def test_analyze_without_key_is_flagged_offline(client):
    body = client.post("/api/analyze").json()
    assert body["llm_used"] is False
    assert "offline" in body["analysis"]


def test_analyze_with_client_returns_claudes_text(client):
    server.state["claude"] = ClaudeAnalyst(client=FakeClient(message([block("text", "Analysis.")])))
    body = client.post("/api/analyze").json()
    assert body["llm_used"] is True and body["analysis"] == "Analysis."
    assert body["model"] == DEFAULT_MODEL


def test_api_failure_is_an_http_error_not_a_canned_answer(client):
    server.state["claude"] = ClaudeAnalyst(
        client=FakeClient(error=status_error(anthropic.RateLimitError, 429)))
    for path, body in (("/api/analyze", None), ("/api/chat", {"message": "hi"})):
        r = client.post(path, json=body)
        assert r.status_code == 502 and "Rate limited" in r.json()["detail"]


def test_chat_rejects_an_empty_message(client):
    assert client.post("/api/chat", json={"message": "   "}).status_code == 400
