"""
Sentiment data collector — news, social media, fear & greed index.
Uses NewsAPI + FinBERT-style scoring.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

NEWS_API_BASE = "https://newsapi.org/v2/everything"
FEAR_GREED_API = "https://api.alternative.me/fng/"
COINGECKO_TRENDING = "https://api.coingecko.com/api/v3/search/trending"


class SentimentDataCollector:
    """Collects and scores sentiment data from news and social sources."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.news_api_key = self.settings.data.news_api_key
        self._sentiment_model = None  # Lazy-loaded

    def _load_sentiment_model(self):
        """Lazy-load FinBERT sentiment model."""
        if self._sentiment_model is None:
            try:
                from transformers import pipeline
                self._sentiment_model = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    truncation=True,
                    max_length=512,
                )
                logger.info("FinBERT sentiment model loaded")
            except Exception as e:
                logger.warning(f"Could not load FinBERT: {e}. Using lexicon fallback.")
        return self._sentiment_model

    def score_text(self, text: str) -> float:
        """
        Score text sentiment. Returns float in [-1, 1].
        1 = very positive, -1 = very negative.
        """
        model = self._load_sentiment_model()
        if model is None:
            return self._lexicon_score(text)

        try:
            result = model(text[:512])[0]
            label = result["label"].lower()
            score = result["score"]
            if label == "positive":
                return score
            elif label == "negative":
                return -score
            return 0.0
        except Exception as e:
            logger.debug(f"FinBERT scoring failed: {e}")
            return self._lexicon_score(text)

    @staticmethod
    def _lexicon_score(text: str) -> float:
        """Simple lexicon-based fallback sentiment scorer."""
        text_lower = text.lower()
        positive_words = {
            "bull", "bullish", "surge", "rally", "growth", "profit", "gain",
            "up", "rise", "positive", "strong", "breakout", "ath", "record",
        }
        negative_words = {
            "bear", "bearish", "crash", "dump", "loss", "decline", "fall",
            "down", "drop", "negative", "weak", "sell-off", "capitulation",
        }
        pos = sum(1 for w in positive_words if w in text_lower)
        neg = sum(1 for w in negative_words if w in text_lower)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    # ------------------------------------------------------------------ #
    #  News API                                                            #
    # ------------------------------------------------------------------ #

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
    def fetch_news(
        self,
        query: str,
        days_back: int = 7,
        language: str = "en",
        page_size: int = 100,
    ) -> list[dict]:
        """Fetch recent news articles for a query via NewsAPI."""
        if not self.news_api_key:
            logger.warning("NEWS_API_KEY not set; skipping news fetch")
            return []

        from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        params = {
            "q": query,
            "from": from_date,
            "language": language,
            "pageSize": page_size,
            "sortBy": "relevancy",
            "apiKey": self.news_api_key,
        }
        resp = requests.get(NEWS_API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        logger.debug(f"News fetched for '{query}': {len(articles)} articles")
        return articles

    def get_news_sentiment(self, symbol: str, days_back: int = 7) -> dict:
        """
        Return aggregated sentiment stats for a symbol from news.
        Returns: {symbol, mean_score, std_score, positive_pct, negative_pct, article_count}
        """
        articles = self.fetch_news(symbol, days_back=days_back)
        if not articles:
            return {"symbol": symbol, "mean_score": 0.0, "std_score": 0.0,
                    "positive_pct": 0.5, "negative_pct": 0.5, "article_count": 0}

        scores = []
        for art in articles:
            text = f"{art.get('title', '')} {art.get('description', '')}"
            if text.strip():
                scores.append(self.score_text(text))

        if not scores:
            return {"symbol": symbol, "mean_score": 0.0, "std_score": 0.0,
                    "positive_pct": 0.5, "negative_pct": 0.5, "article_count": 0}

        scores_arr = np.array(scores)
        return {
            "symbol": symbol,
            "mean_score": float(np.mean(scores_arr)),
            "std_score": float(np.std(scores_arr)),
            "positive_pct": float(np.mean(scores_arr > 0.1)),
            "negative_pct": float(np.mean(scores_arr < -0.1)),
            "article_count": len(scores),
        }

    # ------------------------------------------------------------------ #
    #  Crypto Fear & Greed Index                                           #
    # ------------------------------------------------------------------ #

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
    def fetch_fear_greed_index(self, limit: int = 30) -> pd.DataFrame:
        """Fetch Crypto Fear & Greed Index (Alternative.me)."""
        resp = requests.get(FEAR_GREED_API, params={"limit": limit}, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)[["timestamp", "value", "value_classification"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
        df["value"] = df["value"].astype(int)
        df = df.set_index("timestamp").sort_index()
        logger.debug(f"Fear & Greed: {len(df)} days fetched")
        return df

    # ------------------------------------------------------------------ #
    #  CoinGecko trending                                                  #
    # ------------------------------------------------------------------ #

    def fetch_crypto_trending(self) -> list[str]:
        """Return list of trending crypto symbols (top 7 on CoinGecko)."""
        try:
            resp = requests.get(COINGECKO_TRENDING, timeout=15)
            resp.raise_for_status()
            coins = resp.json().get("coins", [])
            symbols = [c["item"]["symbol"].upper() for c in coins]
            logger.debug(f"Trending crypto: {symbols}")
            return symbols
        except Exception as e:
            logger.warning(f"Trending fetch failed: {e}")
            return []

    def build_sentiment_dataframe(
        self, symbols: list[str], days_back: int = 7
    ) -> pd.DataFrame:
        """Build a DataFrame of sentiment scores for a list of symbols."""
        rows = []
        for symbol in symbols:
            row = self.get_news_sentiment(symbol, days_back=days_back)
            rows.append(row)
        return pd.DataFrame(rows).set_index("symbol")
