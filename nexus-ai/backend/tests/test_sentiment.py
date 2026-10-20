"""Sentiment scraper, offline: fake HTTP for X and 4chan, fake objects for Reddit."""

from types import SimpleNamespace

import pytest
import requests

import sentiment_scraper as ss
from sentiment_scraper import SentimentScraper


@pytest.fixture
def scraper(monkeypatch):
    monkeypatch.setattr(ss.time, "sleep", lambda s: None)  # 4chan politeness delay
    return SentimentScraper()


class FakeResponse:
    def __init__(self, payload=None, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


# ─── Text handling ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "The weather today is lovely",              # "ath"
    "I belong to a book club",                  # "long"
    "My father passed away, the death of a parent",
    "I took a bath and read a bulletin",        # "ath", "bull"
    "She got her diploma",                      # "dip"
    "I'll reply shortly from the household",    # "short", "hold"
])
def test_everyday_text_is_not_financial(scraper, text):
    assert not scraper.is_financial(text)


@pytest.mark.parametrize("text", [
    "$TSLA to the moon",
    "Very bullish on NVDA here",
    "Bought calls, buying the dip",
    "Bears are shorting everything after the Fed",
    "new ATH incoming, squeezed shorts",
])
def test_market_talk_is_financial(scraper, text):
    assert scraper.is_financial(text)


def test_html_entities_are_decoded_so_negation_survives(scraper):
    raw = "don&#039;t buy this<br>it&#039;s going to &gt;crash"
    assert scraper.clean_html(raw) == "don't buy this it's going to >crash"


# ─── X / Twitter ─────────────────────────────────────────────────────────────

def test_x_without_token_reports_no_credentials(scraper):
    scraper.twitter_token = None
    assert scraper.scrape_twitter() == []
    assert scraper.source_status["twitter"]["status"] == "no_credentials"


def test_x_max_results_never_drops_below_ten(scraper, monkeypatch):
    sent = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        sent.update(params=params, timeout=timeout)
        return FakeResponse({"data": []})

    scraper.twitter_token = "token"
    monkeypatch.setattr(ss.requests, "get", fake_get)
    scraper.scrape_twitter(limit=5)
    assert sent["params"]["max_results"] == 10
    assert sent["timeout"] is not None
    assert scraper.source_status["twitter"]["status"] == "empty"


def test_x_403_names_the_paid_tier(scraper, monkeypatch):
    scraper.twitter_token = "token"
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: FakeResponse(status=403))
    assert scraper.scrape_twitter() == []
    status = scraper.source_status["twitter"]
    assert status["status"] == "failed" and "Basic tier" in status["error"]


def test_x_tweets_are_scored(scraper, monkeypatch):
    payload = {"data": [{
        "id": "1", "text": "$NVDA breakout, very bullish", "author_id": "a",
        "created_at": "2026-10-19T12:00:00Z",
        "public_metrics": {"like_count": 10, "retweet_count": 2, "reply_count": 1},
    }]}
    scraper.twitter_token = "token"
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: FakeResponse(payload))
    posts = scraper.scrape_twitter()
    assert len(posts) == 1 and posts[0].sentiment > 0 and posts[0].tickers == ["NVDA"]
    assert posts[0].score == 10 + 2 * 2 + 1


# ─── Reddit ──────────────────────────────────────────────────────────────────

class FakeReddit:
    def __init__(self):
        self.limits = []

    def subreddit(self, name):
        def hot(limit):
            self.limits.append(limit)
            return [SimpleNamespace(
                title=f"$GME moon in r/{name}", selftext="", author="ape",
                created_utc=1_760_000_000, score=100, permalink=f"/r/{name}/1",
                comments=FakeComments(),
            )]
        return SimpleNamespace(hot=hot)


class FakeComments(list):
    def replace_more(self, limit=0):
        return None


def test_reddit_without_credentials_reports_it(scraper):
    scraper.reddit = None
    assert scraper.scrape_reddit() == []
    assert scraper.source_status["reddit"]["status"] == "no_credentials"


def test_reddit_small_limit_still_fetches_every_subreddit(scraper):
    scraper.reddit = FakeReddit()
    posts = scraper.scrape_reddit(limit=5)  # fewer than the 9 subreddits
    assert scraper.reddit.limits == [1] * 9
    assert len(posts) == 9
    assert scraper.source_status["reddit"] == {"status": "ok", "posts": 9, "error": None}


# ─── Aggregation ─────────────────────────────────────────────────────────────

def test_no_posts_is_no_data_not_neutral(scraper, monkeypatch):
    scraper.reddit, scraper.twitter_token = None, None

    def down(*a, **k):
        raise requests.ConnectionError("network unreachable")

    monkeypatch.setattr(ss.requests, "get", down)
    result = scraper.scrape_all()
    assert result["total_posts"] == 0
    assert result["overall_sentiment"] is None and result["sentiment_label"] == "NO_DATA"
    assert result["source_status"]["4chan"]["status"] == "failed"
    assert result["source_status"]["reddit"]["status"] == "no_credentials"


def test_overall_agrees_with_the_per_source_figures(scraper, monkeypatch):
    posts = [
        ss.SentimentPost("reddit", "a", "x", "t", 4200, 0.67, []),
        ss.SentimentPost("reddit", "b", "x", "t", 88, -0.84, []),
        ss.SentimentPost("reddit", "c", "x", "t", 910, -0.88, []),
    ]
    monkeypatch.setattr(scraper, "scrape_reddit", lambda limit: posts)
    monkeypatch.setattr(scraper, "scrape_twitter", lambda limit: [])
    monkeypatch.setattr(scraper, "scrape_4chan", lambda limit: [])
    result = scraper.scrape_all()
    # These posts used to give -0.35 BEARISH overall against +0.37 for Reddit.
    reddit = result["sources"]["reddit"]["avg_sentiment"]
    assert reddit > 0
    assert result["overall_sentiment"] == pytest.approx(reddit, abs=1e-4)
    assert result["sentiment_label"] == "BULLISH"


def test_4chan_posts_are_cleaned_and_scored(scraper, monkeypatch):
    catalog = [{"threads": [{"no": 7, "sub": "BTC thread", "com": "bullish on $BTC"}]}]
    thread = {"posts": [{"com": "I&#039;m not selling, $BTC is going to the moon this year",
                         "time": 1_760_000_000, "replies": 3}]}

    def fake_get(url, timeout=None, **_):
        return FakeResponse(catalog if url.endswith("catalog.json") else thread)

    monkeypatch.setattr(ss.requests, "get", fake_get)
    posts = scraper.scrape_4chan()
    assert len(posts) == 1
    assert posts[0].text.startswith("I'm not selling")
    assert posts[0].tickers == ["BTC"] and posts[0].sentiment > 0
    assert scraper.source_status["4chan"]["status"] == "ok"
