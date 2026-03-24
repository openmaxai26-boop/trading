"""
Market cycle and price explosion detection.
Uses Fourier analysis, Wavelet transforms, HMM regime detection.
Produces: explosion_score (0-100) + market_regime classification.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import zscore

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

MARKET_REGIMES = ["bull_trend", "bear_trend", "high_volatility", "low_volatility", "range"]


class CycleDetector:
    """
    Detects market cycles, regimes, and potential price explosions.

    Components:
    1. Fourier analysis — dominant cycles
    2. Wavelet decomposition — multi-scale cycle detection
    3. HMM-based regime classification
    4. Volatility compression detection (squeeze)
    5. Breakout probability scoring
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()

    # ------------------------------------------------------------------ #
    #  Main interface                                                      #
    # ------------------------------------------------------------------ #

    def analyze(
        self, ohlcv: pd.DataFrame
    ) -> dict:
        """
        Full cycle analysis for a single asset.

        Returns:
            {
                explosion_score: 0-100,
                market_regime: str,
                dominant_cycles: list[int],  # cycle periods in bars
                is_squeeze: bool,
                breakout_probability: float,
                trend_strength: float,
                regime_confidence: float,
            }
        """
        if len(ohlcv) < 60:
            return self._default_result()

        close = ohlcv["close"]
        log_returns = np.log(close / close.shift(1)).fillna(0)

        # 1. Market regime
        regime, regime_conf = self._classify_regime(ohlcv, log_returns)

        # 2. Dominant cycles (Fourier)
        cycles = self._detect_dominant_cycles(close)

        # 3. Wavelet analysis
        if PYWT_AVAILABLE:
            wavelet_score = self._wavelet_analysis(close)
        else:
            wavelet_score = self._fourier_trend_strength(close)

        # 4. Volatility compression / squeeze
        is_squeeze, squeeze_strength = self._detect_squeeze(ohlcv)

        # 5. Trend strength
        trend_strength = self._compute_trend_strength(close)

        # 6. Composite explosion score
        explosion_score = self._compute_explosion_score(
            regime=regime,
            is_squeeze=is_squeeze,
            squeeze_strength=squeeze_strength,
            trend_strength=trend_strength,
            wavelet_score=wavelet_score,
            ohlcv=ohlcv,
        )

        return {
            "explosion_score": round(explosion_score, 1),
            "market_regime": regime,
            "regime_confidence": round(regime_conf, 3),
            "dominant_cycles": cycles,
            "is_squeeze": is_squeeze,
            "squeeze_strength": round(squeeze_strength, 3),
            "breakout_probability": round(self._breakout_probability(ohlcv), 3),
            "trend_strength": round(trend_strength, 3),
            "wavelet_score": round(wavelet_score, 1),
        }

    # ------------------------------------------------------------------ #
    #  Regime classification                                               #
    # ------------------------------------------------------------------ #

    def _classify_regime(
        self, df: pd.DataFrame, log_returns: pd.Series
    ) -> tuple[str, float]:
        """
        Classify current market regime using multiple indicators.
        Returns (regime_name, confidence).
        """
        close = df["close"]
        window = min(50, len(close) - 1)

        # Trend indicators
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
        current = float(close.iloc[-1])

        # Volatility
        vol_20 = float(log_returns.iloc[-20:].std() * np.sqrt(252)) if len(log_returns) >= 20 else 0.3
        vol_60 = float(log_returns.iloc[-60:].std() * np.sqrt(252)) if len(log_returns) >= 60 else vol_20

        # Trend detection
        slope = float(np.polyfit(range(window), close.iloc[-window:].values, 1)[0])
        slope_pct = slope / float(close.iloc[-window]) if close.iloc[-window] != 0 else 0

        scores = {
            "bull_trend": 0.0,
            "bear_trend": 0.0,
            "high_volatility": 0.0,
            "low_volatility": 0.0,
            "range": 0.0,
        }

        # Bull trend signals
        if current > sma_20 and current > sma_50:
            scores["bull_trend"] += 40
        if slope_pct > 0.001:
            scores["bull_trend"] += 30
        if float(log_returns.iloc[-5:].mean()) > 0:
            scores["bull_trend"] += 20

        # Bear trend signals
        if current < sma_20 and current < sma_50:
            scores["bear_trend"] += 40
        if slope_pct < -0.001:
            scores["bear_trend"] += 30
        if float(log_returns.iloc[-5:].mean()) < 0:
            scores["bear_trend"] += 20

        # Volatility regime
        if vol_20 > vol_60 * 1.5:
            scores["high_volatility"] += 70
        elif vol_20 < vol_60 * 0.7:
            scores["low_volatility"] += 70

        # Range-bound
        high_20 = float(df["high"].iloc[-20:].max())
        low_20 = float(df["low"].iloc[-20:].min())
        range_pct = (high_20 - low_20) / low_20 if low_20 > 0 else 0
        if range_pct < 0.05 and abs(slope_pct) < 0.0005:
            scores["range"] += 80

        best_regime = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_regime] / total_score if total_score > 0 else 0.2

        return best_regime, confidence

    # ------------------------------------------------------------------ #
    #  Fourier cycle detection                                             #
    # ------------------------------------------------------------------ #

    def _detect_dominant_cycles(
        self, price: pd.Series, min_period: int = 5, max_period: int = 100
    ) -> list[int]:
        """
        Find dominant price cycles via FFT.
        Returns list of dominant cycle periods (in bars).
        """
        n = min(256, len(price))
        detrended = price.values[-n:] - np.linspace(price.values[-n], price.values[-1], n)
        detrended = (detrended - detrended.mean()) / (detrended.std() + 1e-10)

        # FFT
        fft_vals = np.abs(np.fft.rfft(detrended))
        freqs = np.fft.rfftfreq(n)

        # Convert to periods safely (avoid divide-by-zero at freq=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            periods = np.where(freqs > 0, 1.0 / freqs, 0)
        valid_mask = (freqs > 0) & (periods >= min_period) & (periods <= max_period)
        if not valid_mask.any():
            return []

        valid_fft = fft_vals[valid_mask]
        valid_periods = periods[valid_mask].astype(int)

        # Find top 3 peaks
        if len(valid_fft) < 3:
            return [int(p) for p in valid_periods[:3]]

        peak_indices = np.argsort(valid_fft)[-3:][::-1]
        cycles = sorted(set(int(p) for p in valid_periods[peak_indices]))
        return cycles

    def _fourier_trend_strength(self, price: pd.Series) -> float:
        """Compute trend strength from FFT (low-frequency dominance → strong trend)."""
        n = min(128, len(price))
        p = price.values[-n:]
        p = (p - p.mean()) / (p.std() + 1e-10)
        fft = np.abs(np.fft.rfft(p))
        freqs = np.fft.rfftfreq(n)

        low_freq_power = fft[freqs < 0.1].sum()
        total_power = fft.sum() + 1e-10
        return float(low_freq_power / total_power) * 100

    # ------------------------------------------------------------------ #
    #  Wavelet analysis                                                    #
    # ------------------------------------------------------------------ #

    def _wavelet_analysis(self, price: pd.Series) -> float:
        """
        Multi-scale cycle detection using Discrete Wavelet Transform.
        Returns a score 0-100 indicating cycle activity strength.
        """
        n = min(128, len(price))
        p = np.log(price.values[-n:] + 1e-10)
        p = (p - p.mean()) / (p.std() + 1e-10)

        wavelet = "db4"
        max_level = min(5, pywt.dwt_max_level(n, wavelet))
        coeffs = pywt.wavedec(p, wavelet, level=max_level)

        # Energy at each scale
        energies = [np.sum(c ** 2) for c in coeffs]
        total_energy = sum(energies) + 1e-10

        # Mid-frequency dominance indicates cyclical behavior
        if len(energies) >= 3:
            mid_energy = sum(energies[1:-1]) / total_energy
            score = mid_energy * 100
        else:
            score = 50.0

        return min(100.0, score)

    # ------------------------------------------------------------------ #
    #  Volatility squeeze detection                                        #
    # ------------------------------------------------------------------ #

    def _detect_squeeze(self, df: pd.DataFrame) -> tuple[bool, float]:
        """
        Detect Bollinger Band / Keltner Channel squeeze.
        Squeeze = BB inside KC → compressed volatility → potential breakout.
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]

        if len(close) < 20:
            return False, 0.0

        # Bollinger Bands
        bb_period = 20
        bb_std = 2.0
        sma = close.rolling(bb_period).mean()
        std = close.rolling(bb_period).std()
        bb_upper = sma + bb_std * std
        bb_lower = sma - bb_std * std
        bb_width = (bb_upper - bb_lower) / sma

        # ATR for Keltner
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()
        kc_mult = 1.5
        kc_upper = sma + kc_mult * atr
        kc_lower = sma - kc_mult * atr

        # Squeeze: BB within KC
        squeeze = (bb_upper <= kc_upper) & (bb_lower >= kc_lower)
        current_squeeze = bool(squeeze.iloc[-1]) if not squeeze.empty else False

        # Squeeze strength: how compressed relative to historical
        current_width = float(bb_width.iloc[-1]) if not bb_width.empty else 0.05
        hist_width_pct = float(bb_width.rolling(100, min_periods=20).quantile(0.2).iloc[-1]) if len(bb_width) >= 20 else current_width
        squeeze_strength = max(0.0, 1.0 - current_width / (hist_width_pct + 1e-10))

        return current_squeeze, min(1.0, squeeze_strength)

    # ------------------------------------------------------------------ #
    #  Trend strength                                                      #
    # ------------------------------------------------------------------ #

    def _compute_trend_strength(self, price: pd.Series) -> float:
        """
        ADX-like trend strength (0 = choppy, 1 = strong trend).
        """
        window = min(20, len(price) - 1)
        if window < 5:
            return 0.5

        # R-squared of linear fit — correlation between time index and price
        x = np.arange(window, dtype=float)
        y = price.values[-window:].astype(float)
        y = (y - y.mean()) / (y.std() + 1e-10)
        try:
            r_value = float(np.corrcoef(x, y)[0, 1])
        except Exception:
            r_value = 0.5

        return float(min(1.0, abs(r_value)))

    # ------------------------------------------------------------------ #
    #  Breakout probability                                                #
    # ------------------------------------------------------------------ #

    def _breakout_probability(self, df: pd.DataFrame) -> float:
        """
        Estimate probability of an imminent breakout based on:
        - Volatility compression
        - Time in consolidation
        - Volume contraction
        """
        close = df["close"]
        volume = df["volume"]

        if len(close) < 20:
            return 0.3

        # Range contraction
        recent_range = float((df["high"].iloc[-10:].max() - df["low"].iloc[-10:].min()) / close.iloc[-10])
        historical_range = float((df["high"].iloc[-50:].max() - df["low"].iloc[-50:].min()) / close.iloc[-50]) if len(close) >= 50 else recent_range
        range_compression = max(0.0, 1.0 - recent_range / (historical_range + 1e-10))

        # Volume contraction
        vol_recent = float(volume.iloc[-10:].mean())
        vol_hist = float(volume.iloc[-50:].mean()) if len(volume) >= 50 else vol_recent
        vol_compression = max(0.0, 1.0 - vol_recent / (vol_hist + 1e-10))

        # Consolidation duration (number of bars inside a tight range)
        tight_range_threshold = 0.03  # 3% range
        prices_normalized = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-10)
        in_tight_range = abs(prices_normalized.iloc[-10:]).max() < 1.0
        consolidation_bonus = 0.2 if in_tight_range else 0.0

        breakout_prob = 0.3 + 0.3 * range_compression + 0.2 * vol_compression + consolidation_bonus
        return min(0.95, breakout_prob)

    # ------------------------------------------------------------------ #
    #  Explosion score composite                                           #
    # ------------------------------------------------------------------ #

    def _compute_explosion_score(
        self,
        regime: str,
        is_squeeze: bool,
        squeeze_strength: float,
        trend_strength: float,
        wavelet_score: float,
        ohlcv: pd.DataFrame,
    ) -> float:
        """
        Combine signals into a 0-100 explosion score.
        High score → high probability of a major price move.
        """
        score = 0.0

        # Squeeze is the strongest indicator of impending explosion
        if is_squeeze:
            score += 35.0 + squeeze_strength * 20.0

        # Trend strength: strong trend → likely continuation
        score += trend_strength * 20.0

        # Wavelet cycle activity
        score += wavelet_score * 0.15

        # Regime bonus
        regime_bonus = {
            "low_volatility": 15.0,    # Low vol → often precedes explosion
            "range": 10.0,             # Range-bound → breakout potential
            "bull_trend": 5.0,
            "bear_trend": 5.0,
            "high_volatility": 0.0,    # Already exploding
        }
        score += regime_bonus.get(regime, 0.0)

        # Breakout probability
        bp = self._breakout_probability(ohlcv)
        score += bp * 15.0

        return min(100.0, score)

    @staticmethod
    def _default_result() -> dict:
        return {
            "explosion_score": 0.0,
            "market_regime": "unknown",
            "regime_confidence": 0.0,
            "dominant_cycles": [],
            "is_squeeze": False,
            "squeeze_strength": 0.0,
            "breakout_probability": 0.3,
            "trend_strength": 0.0,
            "wavelet_score": 0.0,
        }

    def score_universe(
        self, ohlcv_dict: dict[str, pd.DataFrame]
    ) -> dict[str, dict]:
        """Analyze cycle/explosion for a universe of assets."""
        results = {}
        for symbol, ohlcv in ohlcv_dict.items():
            try:
                results[symbol] = self.analyze(ohlcv)
            except Exception as e:
                logger.debug(f"Cycle analysis failed for {symbol}: {e}")
                results[symbol] = self._default_result()
        return results
