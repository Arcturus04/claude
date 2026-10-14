"""
╔══════════════════════════════════════════════════════════════╗
║  NEXUS•AI — Social Sentiment Intelligence Module            ║
║  Scrapes Reddit, Twitter/X, 4chan for market sentiment       ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import re
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

import requests
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class SentimentPost:
    """A single post/comment with sentiment data."""
    source: str          # "reddit", "twitter", "4chan"
    text: str
    author: str
    timestamp: str
    score: float         # upvotes, likes, etc.
    sentiment: float     # -1.0 to 1.0
    tickers: List[str]   # extracted ticker symbols
    subreddit: str = ""  # reddit-specific
    url: str = ""


class SentimentScraper:
    """
    Multi-source sentiment aggregator.
    Collects posts from Reddit, Twitter/X, and 4chan's /biz/ board,
    then scores them using VADER sentiment analysis.
    """

    # Common ticker patterns and financial keywords
    TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b')
    TICKER_WORDS = re.compile(r'\b(SPY|QQQ|AAPL|TSLA|MSFT|AMZN|GOOG|GOOGL|META|NVDA|AMD|GME|AMC|BTC|ETH|SOL|XRP)\b')

    FINANCIAL_KEYWORDS = [
        "bull", "bear", "calls", "puts", "short", "long", "moon",
        "crash", "pump", "dump", "squeeze", "tendies", "diamond hands",
        "buy", "sell", "hold", "yolo", "fomo", "dip", "rally",
        "earnings", "fed", "inflation", "recession", "ath", "btfd",
        "breakout", "support", "resistance", "volume", "whale",
    ]

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

        # ─── Add custom financial lexicon to VADER ───
        financial_lexicon = {
            "moon": 2.5, "mooning": 3.0, "rocket": 2.0,
            "tendies": 2.0, "diamond": 1.5, "squeeze": 1.0,
            "bull": 1.5, "bullish": 2.5, "calls": 1.0,
            "bear": -1.5, "bearish": -2.5, "puts": -1.0,
            "crash": -3.0, "dump": -2.5, "rug": -3.0,
            "scam": -2.5, "ponzi": -3.0, "bagholding": -2.0,
            "fomo": -0.5, "fud": -1.5, "rekt": -2.5,
            "pump": 1.0, "rally": 2.0, "breakout": 2.0,
            "dip": -1.0, "btfd": 1.5, "yolo": 0.5,
            "ath": 2.0, "recession": -2.0, "inflation": -1.0,
            "default": -2.5, "bankruptcy": -3.0,
        }
        self.vader.lexicon.update(financial_lexicon)

        # ─── Initialize Reddit client ───
        self.reddit = None
        try:
            client_id = os.getenv("REDDIT_CLIENT_ID")
            client_secret = os.getenv("REDDIT_CLIENT_SECRET")
            user_agent = os.getenv("REDDIT_USER_AGENT", "NexusAI/1.0")
            if client_id and client_secret:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                )
                logger.info("Reddit client initialized successfully")
            else:
                logger.warning("Reddit credentials not found — Reddit scraping disabled")
        except Exception as e:
            logger.error(f"Reddit init failed: {e}")

        # ─── Twitter bearer token ───
        self.twitter_token = os.getenv("TWITTER_BEARER_TOKEN")
        if not self.twitter_token:
            logger.warning("Twitter bearer token not found — Twitter scraping disabled")

    def analyze_sentiment(self, text: str) -> float:
        """Score text sentiment from -1.0 (bearish) to +1.0 (bullish)."""
        scores = self.vader.polarity_scores(text)
        return scores["compound"]

    def extract_tickers(self, text: str) -> List[str]:
        """Extract stock/crypto tickers from text."""
        tickers = set()
        # Match $TICKER format
        for match in self.TICKER_PATTERN.finditer(text):
            tickers.add(match.group(1))
        # Match known tickers without $ prefix
        for match in self.TICKER_WORDS.finditer(text.upper()):
            tickers.add(match.group(1))
        return list(tickers)

    def is_financial(self, text: str) -> bool:
        """Check if text contains financial discussion."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.FINANCIAL_KEYWORDS) or bool(self.TICKER_PATTERN.search(text))

    # ═══════════════════════════════════════════════════════════
    #  REDDIT SCRAPER
    # ═══════════════════════════════════════════════════════════

    def scrape_reddit(self, limit: int = 100) -> List[SentimentPost]:
        """
        Scrape financial subreddits for sentiment.
        Targets: r/wallstreetbets, r/stocks, r/investing,
                 r/options, r/cryptocurrency, r/forex
        """
        if not self.reddit:
            logger.warning("Reddit client not available")
            return []

        posts = []
        subreddits = [
            "wallstreetbets", "stocks", "investing",
            "options", "cryptocurrency", "forex",
            "stockmarket", "pennystocks", "thetagang",
        ]

        for sub_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(sub_name)
                for submission in subreddit.hot(limit=limit // len(subreddits)):
                    full_text = f"{submission.title} {submission.selftext}"

                    if not self.is_financial(full_text):
                        continue

                    sentiment = self.analyze_sentiment(full_text)
                    tickers = self.extract_tickers(full_text)

                    posts.append(SentimentPost(
                        source="reddit",
                        text=full_text[:500],  # Truncate for storage
                        author=str(submission.author) if submission.author else "[deleted]",
                        timestamp=datetime.fromtimestamp(submission.created_utc).isoformat(),
                        score=submission.score,
                        sentiment=sentiment,
                        tickers=tickers,
                        subreddit=sub_name,
                        url=f"https://reddit.com{submission.permalink}",
                    ))

                    # Also grab top comments
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments[:5]:
                        comment_text = comment.body
                        if self.is_financial(comment_text):
                            posts.append(SentimentPost(
                                source="reddit",
                                text=comment_text[:500],
                                author=str(comment.author) if comment.author else "[deleted]",
                                timestamp=datetime.fromtimestamp(comment.created_utc).isoformat(),
                                score=comment.score,
                                sentiment=self.analyze_sentiment(comment_text),
                                tickers=self.extract_tickers(comment_text),
                                subreddit=sub_name,
                                url=f"https://reddit.com{comment.permalink}",
                            ))

                logger.info(f"Scraped {sub_name}: found {len(posts)} financial posts")

            except Exception as e:
                logger.error(f"Error scraping r/{sub_name}: {e}")

        return posts

    # ═══════════════════════════════════════════════════════════
    #  TWITTER/X SCRAPER
    # ═══════════════════════════════════════════════════════════

    def scrape_twitter(self, query: str = "stocks OR trading OR $SPY OR $BTC",
                       limit: int = 50) -> List[SentimentPost]:
        """
        Scrape Twitter/X using v2 API for financial sentiment.
        Requires a Bearer Token from developer.x.com
        """
        if not self.twitter_token:
            logger.warning("Twitter token not available")
            return []

        posts = []
        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {self.twitter_token}"}

        params = {
            "query": f"({query}) lang:en -is:retweet",
            "max_results": min(limit, 100),
            "tweet.fields": "created_at,public_metrics,author_id",
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            for tweet in data.get("data", []):
                text = tweet["text"]
                if not self.is_financial(text):
                    continue

                metrics = tweet.get("public_metrics", {})
                engagement = (
                    metrics.get("like_count", 0) +
                    metrics.get("retweet_count", 0) * 2 +
                    metrics.get("reply_count", 0)
                )

                posts.append(SentimentPost(
                    source="twitter",
                    text=text[:500],
                    author=tweet.get("author_id", "unknown"),
                    timestamp=tweet.get("created_at", datetime.now().isoformat()),
                    score=engagement,
                    sentiment=self.analyze_sentiment(text),
                    tickers=self.extract_tickers(text),
                    url=f"https://x.com/i/status/{tweet['id']}",
                ))

            logger.info(f"Scraped Twitter: found {len(posts)} financial tweets")

        except Exception as e:
            logger.error(f"Twitter scraping error: {e}")

        return posts

    # ═══════════════════════════════════════════════════════════
    #  4CHAN /biz/ SCRAPER
    # ═══════════════════════════════════════════════════════════

    def scrape_4chan(self, limit: int = 50) -> List[SentimentPost]:
        """
        Scrape 4chan's /biz/ (Business & Finance) board.
        Uses the public JSON API — no authentication needed.
        API docs: https://github.com/4chan/4chan-API
        """
        posts = []
        base_url = "https://a.4cdn.org/biz"

        try:
            # Get thread catalog
            catalog_resp = requests.get(f"{base_url}/catalog.json", timeout=10)
            catalog_resp.raise_for_status()
            catalog = catalog_resp.json()

            thread_ids = []
            for page in catalog:
                for thread in page.get("threads", []):
                    # Filter for financial discussion threads
                    subject = thread.get("sub", "")
                    comment = thread.get("com", "")
                    full_text = f"{subject} {comment}"
                    # Remove HTML tags
                    full_text = re.sub(r'<[^>]+>', ' ', full_text)

                    if self.is_financial(full_text):
                        thread_ids.append(thread["no"])

                    if len(thread_ids) >= limit // 3:
                        break

            # Scrape individual threads
            for thread_id in thread_ids[:15]:  # Limit to avoid rate limiting
                try:
                    thread_resp = requests.get(
                        f"{base_url}/thread/{thread_id}.json",
                        timeout=10
                    )
                    thread_resp.raise_for_status()
                    thread_data = thread_resp.json()

                    for post in thread_data.get("posts", [])[:10]:
                        comment = post.get("com", "")
                        # Strip HTML
                        comment = re.sub(r'<[^>]+>', ' ', comment)
                        comment = re.sub(r'&[a-z]+;', ' ', comment)
                        comment = comment.strip()

                        if not comment or len(comment) < 20:
                            continue

                        if not self.is_financial(comment):
                            continue

                        posts.append(SentimentPost(
                            source="4chan",
                            text=comment[:500],
                            author="Anonymous",
                            timestamp=datetime.fromtimestamp(post.get("time", 0)).isoformat(),
                            score=post.get("replies", 0),
                            sentiment=self.analyze_sentiment(comment),
                            tickers=self.extract_tickers(comment),
                            url=f"https://boards.4chan.org/biz/thread/{thread_id}",
                        ))

                    time.sleep(1)  # Rate limiting — be respectful

                except Exception as e:
                    logger.error(f"Error scraping 4chan thread {thread_id}: {e}")

            logger.info(f"Scraped 4chan /biz/: found {len(posts)} financial posts")

        except Exception as e:
            logger.error(f"4chan catalog error: {e}")

        return posts

    # ═══════════════════════════════════════════════════════════
    #  AGGREGATED SENTIMENT
    # ═══════════════════════════════════════════════════════════

    def scrape_all(self, limit_per_source: int = 50) -> Dict:
        """
        Scrape all sources and return aggregated sentiment analysis.
        """
        logger.info("Starting full sentiment scrape...")

        reddit_posts = self.scrape_reddit(limit=limit_per_source)
        twitter_posts = self.scrape_twitter(limit=limit_per_source)
        chan_posts = self.scrape_4chan(limit=limit_per_source)

        all_posts = reddit_posts + twitter_posts + chan_posts

        if not all_posts:
            return {
                "timestamp": datetime.now().isoformat(),
                "total_posts": 0,
                "overall_sentiment": 0.0,
                "sources": {},
                "top_tickers": {},
                "posts": [],
            }

        # Aggregate by source
        source_sentiments = {}
        for source in ["reddit", "twitter", "4chan"]:
            source_posts = [p for p in all_posts if p.source == source]
            if source_posts:
                # Weight by engagement score
                total_score = sum(max(p.score, 1) for p in source_posts)
                weighted_sentiment = sum(
                    p.sentiment * max(p.score, 1) / total_score
                    for p in source_posts
                )
                source_sentiments[source] = {
                    "count": len(source_posts),
                    "avg_sentiment": round(weighted_sentiment, 4),
                    "bullish_pct": round(
                        len([p for p in source_posts if p.sentiment > 0.05]) / len(source_posts) * 100, 1
                    ),
                    "bearish_pct": round(
                        len([p for p in source_posts if p.sentiment < -0.05]) / len(source_posts) * 100, 1
                    ),
                }

        # Aggregate tickers
        ticker_counts = {}
        ticker_sentiments = {}
        for post in all_posts:
            for ticker in post.tickers:
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
                if ticker not in ticker_sentiments:
                    ticker_sentiments[ticker] = []
                ticker_sentiments[ticker].append(post.sentiment)

        top_tickers = {}
        for ticker, count in sorted(ticker_counts.items(), key=lambda x: -x[1])[:20]:
            sentiments = ticker_sentiments[ticker]
            top_tickers[ticker] = {
                "mentions": count,
                "avg_sentiment": round(sum(sentiments) / len(sentiments), 4),
                "bullish_pct": round(len([s for s in sentiments if s > 0.05]) / len(sentiments) * 100, 1),
            }

        # Overall weighted sentiment
        overall = sum(p.sentiment for p in all_posts) / len(all_posts)

        result = {
            "timestamp": datetime.now().isoformat(),
            "total_posts": len(all_posts),
            "overall_sentiment": round(overall, 4),
            "sentiment_label": "BULLISH" if overall > 0.1 else "BEARISH" if overall < -0.1 else "NEUTRAL",
            "sources": source_sentiments,
            "top_tickers": top_tickers,
            "posts": [asdict(p) for p in sorted(all_posts, key=lambda x: -abs(x.sentiment))[:50]],
        }

        logger.info(
            f"Sentiment scrape complete: {len(all_posts)} posts, "
            f"overall={overall:.4f} ({result['sentiment_label']})"
        )

        return result
