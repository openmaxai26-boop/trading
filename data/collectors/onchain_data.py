"""
On-chain data collector for crypto assets.
Sources: CoinGecko (free), Glassnode (premium), blockchain explorers.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
GLASSNODE_BASE = "https://api.glassnode.com/v1/metrics"


class OnChainDataCollector:
    """Collects on-chain metrics for crypto assets."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.coingecko_key = self.settings.data.coingecko_api_key
        self.glassnode_key = self.settings.data.glassnode_api_key

    def _cg_headers(self) -> dict:
        if self.coingecko_key:
            return {"x-cg-pro-api-key": self.coingecko_key}
        return {}

    # ------------------------------------------------------------------ #
    #  CoinGecko on-chain metrics                                          #
    # ------------------------------------------------------------------ #

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    def fetch_coin_info(self, coin_id: str) -> dict:
        """Fetch comprehensive coin data from CoinGecko."""
        url = f"{COINGECKO_BASE}/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "true",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "false",
        }
        resp = requests.get(url, params=params, headers=self._cg_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()

        market = data.get("market_data", {})
        community = data.get("community_data", {})
        tickers = data.get("tickers", [])

        # Compute exchange flow proxy: sum volume across all exchanges
        total_exchange_volume = sum(t.get("volume", 0) or 0 for t in tickers)

        return {
            "coin_id": coin_id,
            "symbol": data.get("symbol", "").upper(),
            "market_cap": market.get("market_cap", {}).get("usd"),
            "fully_diluted_valuation": market.get("fully_diluted_valuation", {}).get("usd"),
            "total_volume_usd": market.get("total_volume", {}).get("usd"),
            "circulating_supply": market.get("circulating_supply"),
            "total_supply": market.get("total_supply"),
            "max_supply": market.get("max_supply"),
            "price_change_24h_pct": market.get("price_change_percentage_24h"),
            "price_change_7d_pct": market.get("price_change_percentage_7d"),
            "price_change_30d_pct": market.get("price_change_percentage_30d"),
            "ath": market.get("ath", {}).get("usd"),
            "ath_change_pct": market.get("ath_change_percentage", {}).get("usd"),
            "twitter_followers": community.get("twitter_followers"),
            "reddit_subscribers": community.get("reddit_subscribers"),
            "exchange_volume_proxy": total_exchange_volume,
            "num_exchanges": len(tickers),
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    def fetch_market_chart(
        self,
        coin_id: str,
        days: int = 90,
        vs_currency: str = "usd",
    ) -> pd.DataFrame:
        """Fetch market chart data (price, market cap, volume) for a coin."""
        url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
        params = {"vs_currency": vs_currency, "days": days, "interval": "daily"}
        resp = requests.get(url, params=params, headers=self._cg_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()

        prices = pd.DataFrame(data.get("prices", []), columns=["ts", "price"])
        market_caps = pd.DataFrame(data.get("market_caps", []), columns=["ts", "market_cap"])
        volumes = pd.DataFrame(data.get("total_volumes", []), columns=["ts", "volume_usd"])

        df = prices.merge(market_caps, on="ts").merge(volumes, on="ts")
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.drop("ts", axis=1, inplace=True)
        df.set_index("timestamp", inplace=True)
        df["nvt_ratio"] = df["market_cap"] / df["volume_usd"].replace(0, float("nan"))
        logger.debug(f"Market chart for {coin_id}: {len(df)} days")
        return df

    # ------------------------------------------------------------------ #
    #  Glassnode on-chain metrics (premium)                                #
    # ------------------------------------------------------------------ #

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=5, max=30))
    def fetch_glassnode_metric(
        self,
        asset: str,
        metric_path: str,
        since: Optional[datetime] = None,
        resolution: str = "24h",
    ) -> pd.Series:
        """
        Fetch a Glassnode metric.
        metric_path examples: 'addresses/active_count', 'supply/current'
        """
        if not self.glassnode_key:
            logger.warning("GLASSNODE_API_KEY not set; skipping Glassnode metric")
            return pd.Series(dtype=float)

        since_ts = int((since or datetime.now(timezone.utc) - timedelta(days=365)).timestamp())
        url = f"{GLASSNODE_BASE}/{metric_path}"
        params = {
            "a": asset.upper(),
            "api_key": self.glassnode_key,
            "s": since_ts,
            "i": resolution,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return pd.Series(dtype=float)

        df = pd.DataFrame(data)
        df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
        s = df.set_index("t")["v"].rename(metric_path.replace("/", "_"))
        return s

    def fetch_key_onchain_metrics(self, asset: str = "BTC") -> dict[str, pd.Series]:
        """Fetch a bundle of key on-chain metrics for an asset."""
        metrics = {
            "active_addresses": "addresses/active_count",
            "exchange_inflow": "transactions/transfers_volume_to_exchanges_sum",
            "exchange_outflow": "transactions/transfers_volume_from_exchanges_sum",
            "miner_outflow": "mining/revenue_sum",
            "sopr": "indicators/sopr",  # Spent Output Profit Ratio
            "nupl": "indicators/nupl",  # Net Unrealized Profit/Loss
        }
        results = {}
        for name, path in metrics.items():
            try:
                s = self.fetch_glassnode_metric(asset, path)
                if not s.empty:
                    results[name] = s
            except Exception as e:
                logger.warning(f"Glassnode {name} failed: {e}")
        return results

    # ------------------------------------------------------------------ #
    #  Exchange flow analysis                                              #
    # ------------------------------------------------------------------ #

    def compute_exchange_flow_ratio(
        self,
        inflow: pd.Series,
        outflow: pd.Series,
    ) -> pd.Series:
        """
        Exchange flow ratio = inflow / (inflow + outflow).
        > 0.5 = more inflow (bearish — coins moving to exchanges to sell)
        < 0.5 = more outflow (bullish — coins leaving exchanges)
        """
        total = inflow + outflow
        return (inflow / total.replace(0, float("nan"))).rename("exchange_flow_ratio")
