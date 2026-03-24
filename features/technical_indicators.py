"""
Technical indicator computation.
Uses the `ta` library as primary source, with custom implementations.
"""

import numpy as np
import pandas as pd

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

from utils.logger import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    """Computes 50+ technical indicators on OHLCV data."""

    def __init__(
        self,
        rsi_periods: list[int] | None = None,
        ma_periods: list[int] | None = None,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ) -> None:
        self.rsi_periods = rsi_periods or [7, 14, 21]
        self.ma_periods = ma_periods or [5, 10, 20, 50, 100, 200]
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all indicators on an OHLCV DataFrame.
        Returns the DataFrame with new indicator columns appended.
        """
        if df.empty or len(df) < 50:
            logger.warning("Insufficient bars for indicator computation")
            return df

        result = df.copy()

        result = self._add_moving_averages(result)
        result = self._add_rsi(result)
        result = self._add_macd(result)
        result = self._add_bollinger_bands(result)
        result = self._add_atr(result)
        result = self._add_volume_indicators(result)
        result = self._add_momentum(result)
        result = self._add_price_patterns(result)
        result = self._add_volatility_regime(result)

        return result

    # ------------------------------------------------------------------ #
    #  Moving averages                                                     #
    # ------------------------------------------------------------------ #

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        for p in self.ma_periods:
            df[f"sma_{p}"] = close.rolling(p).mean()
            df[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()

        # Golden cross / death cross signals
        if 50 in self.ma_periods and 200 in self.ma_periods:
            df["golden_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)
            df["ma_50_200_ratio"] = df["sma_50"] / df["sma_200"]

        # Price relative to key MAs
        for p in [20, 50, 200]:
            if f"sma_{p}" in df.columns:
                df[f"price_vs_sma{p}"] = close / df[f"sma_{p}"] - 1

        # VWAP (daily session)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        df["price_vs_vwap"] = close / df["vwap"] - 1

        return df

    # ------------------------------------------------------------------ #
    #  RSI                                                                 #
    # ------------------------------------------------------------------ #

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        for period in self.rsi_periods:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
            avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # RSI divergence (price makes new high but RSI doesn't)
        if "rsi_14" in df.columns:
            df["rsi_14_oversold"] = (df["rsi_14"] < 30).astype(int)
            df["rsi_14_overbought"] = (df["rsi_14"] > 70).astype(int)

        return df

    # ------------------------------------------------------------------ #
    #  MACD                                                                #
    # ------------------------------------------------------------------ #

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=self.macd_signal, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        df["macd_crossover"] = (
            (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
        ).astype(int)
        df["macd_crossunder"] = (
            (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))
        ).astype(int)
        return df

    # ------------------------------------------------------------------ #
    #  Bollinger Bands                                                     #
    # ------------------------------------------------------------------ #

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        sma = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        df["bb_upper"] = sma + self.bb_std * std
        df["bb_lower"] = sma - self.bb_std * std
        df["bb_middle"] = sma
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
        df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(50).quantile(0.2)).astype(int)
        return df

    # ------------------------------------------------------------------ #
    #  ATR                                                                 #
    # ------------------------------------------------------------------ #

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.ewm(span=self.atr_period, adjust=False).mean()
        df["atr_pct"] = df["atr"] / df["close"]

        # Normalized ATR channels
        df["atr_upper"] = df["close"] + 2 * df["atr"]
        df["atr_lower"] = df["close"] - 2 * df["atr"]
        return df

    # ------------------------------------------------------------------ #
    #  Volume indicators                                                   #
    # ------------------------------------------------------------------ #

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        vol = df["volume"]
        close = df["close"]

        # OBV
        price_change = close.diff()
        obv = (np.sign(price_change) * vol).fillna(0).cumsum()
        df["obv"] = obv
        df["obv_ema_20"] = obv.ewm(span=20, adjust=False).mean()
        df["obv_vs_ema"] = obv / df["obv_ema_20"] - 1

        # Volume moving averages
        df["vol_sma_20"] = vol.rolling(20).mean()
        df["vol_ratio"] = vol / df["vol_sma_20"]
        df["volume_spike"] = (df["vol_ratio"] > 2.0).astype(int)

        # CMF (Chaikin Money Flow)
        mf_multiplier = ((close - df["low"]) - (df["high"] - close)) / (
            (df["high"] - df["low"]).replace(0, np.nan)
        )
        mf_volume = mf_multiplier * vol
        df["cmf"] = mf_volume.rolling(20).sum() / vol.rolling(20).sum()

        # VWMA
        df["vwma_20"] = (close * vol).rolling(20).sum() / vol.rolling(20).sum()

        return df

    # ------------------------------------------------------------------ #
    #  Momentum                                                            #
    # ------------------------------------------------------------------ #

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        # Rate of change
        for period in [1, 5, 10, 21, 63]:
            df[f"roc_{period}"] = close.pct_change(period)

        # Stochastic Oscillator
        for period in [14, 21]:
            low_min = df["low"].rolling(period).min()
            high_max = df["high"].rolling(period).max()
            df[f"stoch_k_{period}"] = 100 * (close - low_min) / (high_max - low_min).replace(0, np.nan)
            df[f"stoch_d_{period}"] = df[f"stoch_k_{period}"].rolling(3).mean()

        # Williams %R
        for period in [14]:
            high_max = df["high"].rolling(period).max()
            low_min = df["low"].rolling(period).min()
            df[f"williams_r_{period}"] = -100 * (high_max - close) / (high_max - low_min).replace(0, np.nan)

        # CCI
        typical = (df["high"] + df["low"] + close) / 3
        mean_dev = typical.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        df["cci"] = (typical - typical.rolling(20).mean()) / (0.015 * mean_dev)

        return df

    # ------------------------------------------------------------------ #
    #  Price patterns                                                      #
    # ------------------------------------------------------------------ #

    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Candlestick features
        df["body_size"] = abs(df["close"] - df["open"]) / df["open"]
        df["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / df["open"]
        df["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / df["open"]
        df["is_bullish"] = (df["close"] > df["open"]).astype(int)

        # Support / resistance proximity (rolling)
        df["resistance_52w"] = df["high"].rolling(252).max()
        df["support_52w"] = df["low"].rolling(252).min()
        df["pct_from_52w_high"] = df["close"] / df["resistance_52w"] - 1
        df["pct_from_52w_low"] = df["close"] / df["support_52w"] - 1

        # Inside bar
        df["inside_bar"] = (
            (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
        ).astype(int)

        return df

    # ------------------------------------------------------------------ #
    #  Volatility regime                                                   #
    # ------------------------------------------------------------------ #

    def _add_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        log_returns = np.log(df["close"] / df["close"].shift(1))

        # Rolling realized volatility
        for window in [5, 10, 20, 60]:
            df[f"rv_{window}"] = log_returns.rolling(window).std() * np.sqrt(252)

        # Volatility regime: expansion vs contraction
        df["vol_regime_expanding"] = (df["rv_20"] > df["rv_60"]).astype(int)
        df["vol_ratio_short_long"] = df["rv_5"] / df["rv_20"]

        # Garman-Klass volatility estimator
        df["gk_vol"] = np.sqrt(
            0.5 * np.log(df["high"] / df["low"]) ** 2
            - (2 * np.log(2) - 1) * np.log(df["close"] / df["open"]) ** 2
        ).rolling(20).mean() * np.sqrt(252)

        return df
