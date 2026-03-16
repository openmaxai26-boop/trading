"""
Market data collector — OHLCV, order book, tick data.
Supports crypto via ccxt and equities via yfinance.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class MarketDataCollector:
    """Collects OHLCV and order book data from multiple sources."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self._exchange = self._init_exchange()

    def _init_exchange(self) -> ccxt.Exchange:
        cfg = self.settings.data
        exchange_class = getattr(ccxt, cfg.exchange_id, ccxt.binance)
        exchange = exchange_class(
            {
                "apiKey": cfg.exchange_api_key,
                "secret": cfg.exchange_api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        return exchange

    # ------------------------------------------------------------------ #
    #  Crypto data via ccxt                                                #
    # ------------------------------------------------------------------ #

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def fetch_ohlcv_crypto(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a crypto symbol from the configured exchange."""
        since_ms: Optional[int] = None
        if since:
            since_ms = int(since.timestamp() * 1000)
        elif not since:
            # Default: lookback_days
            lookback = self.settings.data.lookback_days
            since_ms = int((datetime.now(timezone.utc) - timedelta(days=lookback)).timestamp() * 1000)

        raw = self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not raw:
            logger.warning(f"No OHLCV data returned for {symbol} {timeframe}")
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        logger.debug(f"Fetched {len(df)} bars for {symbol} {timeframe}")
        return df

    def fetch_order_book(self, symbol: str, depth: int = 20) -> dict:
        """Fetch current order book snapshot."""
        try:
            book = self._exchange.fetch_order_book(symbol, limit=depth)
            return {
                "timestamp": datetime.now(timezone.utc),
                "symbol": symbol,
                "bids": book["bids"][:depth],
                "asks": book["asks"][:depth],
                "bid_ask_spread": (
                    book["asks"][0][0] - book["bids"][0][0]
                    if book["asks"] and book["bids"]
                    else None
                ),
            }
        except Exception as e:
            logger.error(f"Order book fetch failed for {symbol}: {e}")
            return {}

    def get_available_crypto_symbols(self, quote_currency: str = "USDT") -> list[str]:
        """Return all available symbols quoted in the given currency."""
        try:
            markets = self._exchange.load_markets()
            symbols = [
                s for s, m in markets.items()
                if m.get("quote") == quote_currency and m.get("active")
            ]
            limit = self.settings.data.crypto_symbols_limit
            logger.info(f"Found {len(symbols)} active {quote_currency} symbols (cap: {limit})")
            return symbols[:limit]
        except Exception as e:
            logger.error(f"Failed to load crypto markets: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Equity data via yfinance                                            #
    # ------------------------------------------------------------------ #

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def fetch_ohlcv_equity(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data for an equity via yfinance."""
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if data.empty:
                logger.warning(f"No equity data returned for {ticker}")
                return pd.DataFrame()

            df = data.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "timestamp"
            logger.debug(f"Fetched {len(df)} bars for equity {ticker}")
            return df
        except Exception as e:
            logger.error(f"Equity fetch failed for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_batch_ohlcv(
        self,
        symbols: list[str],
        asset_type: str = "crypto",
        timeframe: str = "1d",
        delay_ms: float = 100,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for a list of symbols with rate-limit delay."""
        results: dict[str, pd.DataFrame] = {}
        for i, symbol in enumerate(symbols):
            try:
                if asset_type == "crypto":
                    df = self.fetch_ohlcv_crypto(symbol, timeframe=timeframe)
                else:
                    df = self.fetch_ohlcv_equity(symbol)

                if not df.empty:
                    results[symbol] = df

            except Exception as e:
                logger.warning(f"Skipping {symbol}: {e}")

            if delay_ms > 0:
                time.sleep(delay_ms / 1000)

            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i+1}/{len(symbols)} symbols fetched")

        logger.info(f"Batch fetch complete: {len(results)}/{len(symbols)} succeeded")
        return results

    # ------------------------------------------------------------------ #
    #  Volatility and spread metrics                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_realized_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Compute rolling realized volatility (annualized)."""
        log_returns = np.log(df["close"] / df["close"].shift(1))
        rv = log_returns.rolling(window).std() * np.sqrt(252)
        return rv.rename("realized_volatility")

    @staticmethod
    def compute_amihud_illiquidity(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Amihud illiquidity ratio: |return| / volume."""
        returns = df["close"].pct_change().abs()
        illiq = (returns / df["volume"].replace(0, np.nan)).rolling(window).mean()
        return illiq.rename("amihud_illiquidity")
