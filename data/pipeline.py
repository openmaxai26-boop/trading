"""
Main data ingestion pipeline — orchestrates all collectors and storage.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from config.settings import Settings
from data.collectors.market_data import MarketDataCollector
from data.collectors.fundamental_data import FundamentalDataCollector
from data.collectors.sentiment_data import SentimentDataCollector
from data.collectors.onchain_data import OnChainDataCollector
from data.storage.database import Database, init_db
from data.storage.cache import CacheManager
from utils.logger import get_logger

logger = get_logger(__name__)


class DataPipeline:
    """
    Orchestrates end-to-end data collection for all asset classes.

    Usage:
        pipeline = DataPipeline()
        pipeline.run(asset_types=["crypto", "equity"])
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.db = init_db(self.settings)
        self.cache = CacheManager(self.settings)
        self.market = MarketDataCollector(self.settings)
        self.fundamental = FundamentalDataCollector(self.settings)
        self.sentiment = SentimentDataCollector(self.settings)
        self.onchain = OnChainDataCollector(self.settings)

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def run(
        self,
        asset_types: Optional[list[str]] = None,
        timeframes: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> dict[str, int]:
        """
        Full data ingestion run.
        Returns dict with counts of assets processed per type.
        """
        asset_types = asset_types or ["crypto", "equity"]
        timeframes = timeframes or [self.settings.data.primary_timeframe]

        logger.info(f"Data pipeline starting — asset_types={asset_types}, timeframes={timeframes}")
        counts: dict[str, int] = {}

        if "crypto" in asset_types:
            counts["crypto"] = self._run_crypto(timeframes=timeframes, limit=limit)

        if "equity" in asset_types:
            counts["equity"] = self._run_equity(limit=limit)

        # Macro / fundamental (not per-asset)
        try:
            self._run_macro()
            counts["macro"] = 1
        except Exception as e:
            logger.error(f"Macro data failed: {e}")

        logger.info(f"Data pipeline complete: {counts}")
        return counts

    # ------------------------------------------------------------------ #
    #  Crypto pipeline                                                     #
    # ------------------------------------------------------------------ #

    def _run_crypto(self, timeframes: list[str], limit: Optional[int]) -> int:
        logger.info("Collecting crypto OHLCV data…")
        symbols = self.market.get_available_crypto_symbols()
        if limit:
            symbols = symbols[:limit]

        success = 0
        for tf in timeframes:
            batch = self.market.fetch_batch_ohlcv(symbols, asset_type="crypto", timeframe=tf)
            for symbol, df in batch.items():
                try:
                    self.db.save_ohlcv(symbol, tf, df)
                    success += 1
                except Exception as e:
                    logger.warning(f"Failed to save {symbol} {tf}: {e}")

        logger.info(f"Crypto OHLCV: {success} datasets saved")

        # Sentiment
        logger.info("Collecting crypto sentiment…")
        try:
            crypto_names = [s.split("/")[0] for s in symbols[:50]]  # Top 50 for sentiment
            sent_df = self.sentiment.build_sentiment_dataframe(crypto_names, days_back=3)
            if not sent_df.empty:
                path = os.path.join(self.settings.data.data_lake_path, "sentiment", "crypto_sentiment.parquet")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                sent_df.to_parquet(path)
                logger.info(f"Crypto sentiment saved: {len(sent_df)} rows")
        except Exception as e:
            logger.warning(f"Crypto sentiment collection failed: {e}")

        # Fear & Greed
        try:
            fg = self.sentiment.fetch_fear_greed_index(limit=90)
            if not fg.empty:
                path = os.path.join(self.settings.data.data_lake_path, "sentiment", "fear_greed.parquet")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                fg.to_parquet(path)
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")

        # On-chain for major assets
        major_assets = ["BTC", "ETH"]
        for asset in major_assets:
            try:
                metrics = self.onchain.fetch_key_onchain_metrics(asset)
                if metrics:
                    for name, series in metrics.items():
                        path = os.path.join(
                            self.settings.data.data_lake_path, "onchain", asset, f"{name}.parquet"
                        )
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        series.to_frame().to_parquet(path)
                    logger.info(f"On-chain metrics saved for {asset}")
            except Exception as e:
                logger.warning(f"On-chain collection failed for {asset}: {e}")

        return success

    # ------------------------------------------------------------------ #
    #  Equity pipeline                                                     #
    # ------------------------------------------------------------------ #

    def _run_equity(self, limit: Optional[int]) -> int:
        # Default universe: S&P 500 sample + global indices
        default_tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM",
            "JNJ", "V", "WMT", "MA", "PG", "HD", "BAC", "XOM", "CVX", "ABBV",
            "MRK", "LLY", "SPY", "QQQ", "IWM", "GLD", "SLV", "USO", "TLT",
            "^GSPC", "^DJI", "^IXIC", "^VIX", "^TNX",
        ]
        tickers = default_tickers[:limit] if limit else default_tickers
        logger.info(f"Collecting equity data for {len(tickers)} tickers…")

        success = 0
        batch = self.market.fetch_batch_ohlcv(tickers, asset_type="equity", timeframe="1d")
        for ticker, df in batch.items():
            try:
                self.db.save_ohlcv(ticker, "1d", df)
                success += 1
            except Exception as e:
                logger.warning(f"Failed to save equity {ticker}: {e}")

        logger.info(f"Equity OHLCV: {success} saved")
        return success

    # ------------------------------------------------------------------ #
    #  Macro pipeline                                                      #
    # ------------------------------------------------------------------ #

    def _run_macro(self) -> None:
        logger.info("Collecting macro data…")
        macro_df = self.fundamental.fetch_all_macro()
        if not macro_df.empty:
            slope = self.fundamental.compute_yield_curve_slope(macro_df)
            macro_df["yield_curve_slope"] = slope
            path = os.path.join(self.settings.data.data_lake_path, "macro", "macro_indicators.parquet")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            macro_df.to_parquet(path)
            logger.info(f"Macro data saved: {macro_df.shape}")

    # ------------------------------------------------------------------ #
    #  Data loading helpers (used by feature pipeline)                    #
    # ------------------------------------------------------------------ #

    def load_ohlcv(
        self, symbol: str, timeframe: str = "1d", lookback_days: Optional[int] = None
    ) -> pd.DataFrame:
        """Load OHLCV from Parquet, optionally limited to recent days."""
        start = None
        if lookback_days:
            start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        return self.db.load_ohlcv(symbol, timeframe, start=start)

    def load_macro(self) -> pd.DataFrame:
        path = os.path.join(self.settings.data.data_lake_path, "macro", "macro_indicators.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        return pd.DataFrame()

    def load_sentiment(self, asset_class: str = "crypto") -> pd.DataFrame:
        path = os.path.join(
            self.settings.data.data_lake_path, "sentiment", f"{asset_class}_sentiment.parquet"
        )
        if os.path.exists(path):
            return pd.read_parquet(path)
        return pd.DataFrame()
