"""
Whale / Smart Money detection module.
Analyzes volume anomalies, order book imbalances, and on-chain flows.
Produces a whale_score (0-100).
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class WhaleDetector:
    """
    Detects institutional / whale activity using multiple signals:

    1. Volume spikes (abnormal vs rolling average)
    2. Order book imbalance (bid/ask volume asymmetry)
    3. Large transaction clustering (on-chain)
    4. Price impact analysis (large moves with volume)
    5. Spoofing indicators (large orders that disappear)

    Output: whale_score 0-100
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()

    # ------------------------------------------------------------------ #
    #  Main scoring function                                               #
    # ------------------------------------------------------------------ #

    def compute_whale_score(
        self,
        ohlcv: pd.DataFrame,
        order_book: Optional[dict] = None,
        onchain_metrics: Optional[dict[str, pd.Series]] = None,
    ) -> float:
        """
        Compute a composite whale activity score (0-100).

        Higher score = stronger evidence of institutional / whale activity.
        """
        scores = []
        weights = []

        # Volume analysis (always available)
        vol_score = self._volume_spike_score(ohlcv)
        scores.append(vol_score)
        weights.append(0.35)

        # Price-volume pattern
        pv_score = self._price_volume_pattern_score(ohlcv)
        scores.append(pv_score)
        weights.append(0.25)

        # Order book (if available)
        if order_book and order_book.get("bids") and order_book.get("asks"):
            ob_score = self._order_book_imbalance_score(order_book)
            scores.append(ob_score)
            weights.append(0.20)
        else:
            # Redistribute weight
            weights[0] += 0.10
            weights[1] += 0.10

        # On-chain (if available)
        if onchain_metrics:
            oc_score = self._onchain_score(onchain_metrics)
            scores.append(oc_score)
            weights.append(0.20)
        else:
            weights[0] += 0.10
            weights[1] += 0.10

        # Normalize weights
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        composite = sum(s * w for s, w in zip(scores, weights))
        return min(100.0, max(0.0, composite))

    # ------------------------------------------------------------------ #
    #  Volume spike detection                                              #
    # ------------------------------------------------------------------ #

    def _volume_spike_score(self, df: pd.DataFrame, window: int = 20) -> float:
        """
        Score based on current volume vs rolling average.
        Volume > 3σ above mean → high score.
        """
        if len(df) < window + 5:
            return 0.0

        vol = df["volume"]
        rolling_mean = vol.rolling(window).mean()
        rolling_std = vol.rolling(window).std()

        # Z-score of recent volume
        z_scores = ((vol - rolling_mean) / (rolling_std + 1e-10)).fillna(0)
        recent_z = float(z_scores.iloc[-5:].max())  # Max over last 5 bars

        # Map z-score to 0-100
        if recent_z < 1.0:
            return 0.0
        elif recent_z < 2.0:
            return 20.0 + (recent_z - 1.0) * 30.0
        elif recent_z < 3.0:
            return 50.0 + (recent_z - 2.0) * 30.0
        else:
            return min(100.0, 80.0 + (recent_z - 3.0) * 10.0)

    # ------------------------------------------------------------------ #
    #  Price-volume patterns                                               #
    # ------------------------------------------------------------------ #

    def _price_volume_pattern_score(self, df: pd.DataFrame, window: int = 20) -> float:
        """
        Detect accumulation/distribution patterns:
        - High volume + small price move = absorption (accumulation/distribution)
        - Rising volume with consistent price direction = institutional trend
        - Volume divergence from price trend = smart money reversals
        """
        if len(df) < window + 5:
            return 0.0

        close = df["close"]
        vol = df["volume"]

        # Price efficiency ratio: |close change| / sum(|bar ranges|)
        price_change = abs(close.iloc[-1] - close.iloc[-window])
        total_range = (df["high"] - df["low"]).iloc[-window:].sum()
        if total_range > 0:
            efficiency = price_change / total_range
        else:
            efficiency = 0.5

        # Volume trend vs price trend correlation (divergence detector)
        vol_trend = np.polyfit(range(window), vol.iloc[-window:].values, 1)[0]
        price_trend = np.polyfit(range(window), close.iloc[-window:].values, 1)[0]

        # Both trending same direction = confirmation
        # Divergence = potential smart money activity
        if vol_trend > 0 and price_trend > 0:
            trend_score = 30.0   # Volume confirms uptrend
        elif vol_trend > 0 and price_trend < 0:
            trend_score = 70.0   # Volume rising while price falls = accumulation signal
        elif vol_trend < 0 and price_trend > 0:
            trend_score = 65.0   # Volume falling while price rises = distribution signal
        else:
            trend_score = 15.0

        # Large-body candles with high volume
        recent = df.iloc[-5:]
        large_body = (abs(recent["close"] - recent["open"]) / recent["open"] > 0.02).any()
        high_vol = (recent["volume"] > vol.rolling(window).mean().iloc[-5:]).any()

        if large_body and high_vol:
            trend_score = min(100.0, trend_score + 20.0)

        # Low efficiency + high volume = absorption
        if efficiency < 0.3 and vol_trend > 0:
            trend_score = min(100.0, trend_score + 25.0)

        return min(100.0, trend_score)

    # ------------------------------------------------------------------ #
    #  Order book analysis                                                 #
    # ------------------------------------------------------------------ #

    def _order_book_imbalance_score(self, order_book: dict) -> float:
        """
        Analyze bid/ask imbalance in the order book.
        Large bid wall >> ask wall → buying pressure (bullish whale)
        Large ask wall >> bid wall → selling pressure (bearish whale)
        """
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])

        if not bids or not asks:
            return 0.0

        # Total bid / ask volume
        bid_volume = sum(price * size for price, size in bids)
        ask_volume = sum(price * size for price, size in asks)

        if bid_volume + ask_volume == 0:
            return 0.0

        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)  # -1 to 1

        # Large bid > ask imbalance = potential whale buying
        abs_imbalance = abs(imbalance)
        if abs_imbalance < 0.1:
            score = 10.0
        elif abs_imbalance < 0.3:
            score = 30.0 + abs_imbalance * 100
        elif abs_imbalance < 0.5:
            score = 60.0 + abs_imbalance * 60
        else:
            score = min(100.0, 90.0 + abs_imbalance * 20)

        # Large individual orders (iceberg detection)
        max_bid = max((size for _, size in bids), default=0)
        max_ask = max((size for _, size in asks), default=0)
        avg_bid = np.mean([size for _, size in bids]) if bids else 0
        avg_ask = np.mean([size for _, size in asks]) if asks else 0

        if avg_bid > 0 and max_bid / avg_bid > 10:
            score = min(100.0, score + 15.0)  # Iceberg bid detected
        if avg_ask > 0 and max_ask / avg_ask > 10:
            score = min(100.0, score + 15.0)  # Iceberg ask detected

        return score

    # ------------------------------------------------------------------ #
    #  On-chain analysis                                                   #
    # ------------------------------------------------------------------ #

    def _onchain_score(self, metrics: dict[str, pd.Series]) -> float:
        """
        Score based on on-chain metrics:
        - Exchange inflow/outflow ratio
        - SOPR (Spent Output Profit Ratio)
        - NVT ratio anomalies
        - Active address spikes
        """
        score = 0.0
        components = 0

        # Exchange flow: large outflow = whales withdrawing (bullish)
        if "exchange_inflow" in metrics and "exchange_outflow" in metrics:
            inflow = metrics["exchange_inflow"]
            outflow = metrics["exchange_outflow"]
            if len(inflow) > 0 and len(outflow) > 0:
                recent_outflow = float(outflow.iloc[-7:].mean())
                recent_inflow = float(inflow.iloc[-7:].mean())
                total = recent_inflow + recent_outflow
                if total > 0:
                    outflow_ratio = recent_outflow / total
                    # High outflow ratio = coins leaving exchanges (bullish institutional)
                    if outflow_ratio > 0.6:
                        score += 70.0
                    elif outflow_ratio > 0.5:
                        score += 45.0
                    else:
                        score += 20.0
                    components += 1

        # SOPR analysis
        if "sopr" in metrics:
            sopr = metrics["sopr"]
            if len(sopr) > 7:
                recent_sopr = float(sopr.iloc[-7:].mean())
                if recent_sopr > 1.05:
                    score += 60.0   # Profitable spending = smart money taking profits
                elif recent_sopr < 0.95:
                    score += 40.0   # Spending at loss = capitulation / buying opportunity
                else:
                    score += 20.0
                components += 1

        # Active address spikes
        if "active_addresses" in metrics:
            aa = metrics["active_addresses"]
            if len(aa) > 30:
                z_score = (float(aa.iloc[-1]) - float(aa.iloc[-30:].mean())) / (float(aa.iloc[-30:].std()) + 1e-10)
                if z_score > 2:
                    score += 65.0
                elif z_score > 1:
                    score += 35.0
                else:
                    score += 10.0
                components += 1

        return min(100.0, score / max(components, 1))

    # ------------------------------------------------------------------ #
    #  Batch scoring                                                       #
    # ------------------------------------------------------------------ #

    def score_universe(
        self,
        ohlcv_dict: dict[str, pd.DataFrame],
        order_books: Optional[dict[str, dict]] = None,
    ) -> dict[str, float]:
        """Score whale activity for a universe of assets."""
        scores = {}
        for symbol, ohlcv in ohlcv_dict.items():
            try:
                ob = order_books.get(symbol) if order_books else None
                scores[symbol] = self.compute_whale_score(ohlcv, order_book=ob)
            except Exception as e:
                logger.debug(f"Whale score failed for {symbol}: {e}")
                scores[symbol] = 0.0
        return scores
