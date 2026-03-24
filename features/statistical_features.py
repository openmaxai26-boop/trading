"""
Statistical feature engineering — autocorrelation, distributional stats,
volatility clustering, cross-asset correlations.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf, adfuller

from utils.logger import get_logger

logger = get_logger(__name__)


class StatisticalFeatures:
    """Computes statistical properties of return series."""

    def compute(self, df: pd.DataFrame, macro_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Add statistical features to an OHLCV DataFrame.
        Optionally attach macro correlation features.
        """
        if df.empty or len(df) < 30:
            return df

        result = df.copy()
        result = self._add_return_distribution(result)
        result = self._add_autocorrelation(result)
        result = self._add_volatility_clustering(result)
        result = self._add_drawdown_features(result)
        result = self._add_entropy_features(result)

        if macro_df is not None and not macro_df.empty:
            result = self._add_macro_correlations(result, macro_df)

        return result

    # ------------------------------------------------------------------ #
    #  Return distribution stats                                           #
    # ------------------------------------------------------------------ #

    def _add_return_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        log_ret = np.log(df["close"] / df["close"].shift(1))

        for window in [20, 60]:
            df[f"mean_return_{window}"] = log_ret.rolling(window).mean()
            df[f"std_return_{window}"] = log_ret.rolling(window).std()
            df[f"skewness_{window}"] = log_ret.rolling(window).skew()
            df[f"kurtosis_{window}"] = log_ret.rolling(window).kurt()

            # Z-score of current return vs rolling distribution
            df[f"return_zscore_{window}"] = (
                (log_ret - df[f"mean_return_{window}"]) / df[f"std_return_{window}"]
            )

        # Tail risk (CVaR 5%)
        df["cvar_5pct"] = log_ret.rolling(60).apply(
            lambda x: x[x <= np.percentile(x, 5)].mean() if len(x) > 0 else np.nan,
            raw=True,
        )

        # Daily log return
        df["log_return"] = log_ret
        df["return_1d"] = df["close"].pct_change(1)

        return df

    # ------------------------------------------------------------------ #
    #  Autocorrelation                                                     #
    # ------------------------------------------------------------------ #

    def _add_autocorrelation(self, df: pd.DataFrame) -> pd.DataFrame:
        log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0)

        for lag in [1, 5, 10]:
            df[f"autocorr_lag{lag}"] = log_ret.rolling(60).apply(
                lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else np.nan,
                raw=False,
            )

        # Squared return autocorrelation (volatility clustering indicator)
        sq_ret = log_ret ** 2
        for lag in [1, 5]:
            df[f"sq_autocorr_lag{lag}"] = sq_ret.rolling(60).apply(
                lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else np.nan,
                raw=False,
            )

        # Hurst exponent (trending vs mean-reverting)
        df["hurst"] = log_ret.rolling(100).apply(
            self._hurst_exponent, raw=True
        )

        return df

    @staticmethod
    def _hurst_exponent(series: np.ndarray) -> float:
        """
        Compute the Hurst exponent via R/S analysis.
        H > 0.5 = trending, H < 0.5 = mean-reverting, H ≈ 0.5 = random walk.
        """
        if len(series) < 20:
            return np.nan
        try:
            lags = range(2, min(len(series) // 2, 20))
            tau = []
            for lag in lags:
                ts_lag = series[lag:]
                ts_prev = series[:-lag]
                diff = ts_lag - ts_prev
                tau.append(np.sqrt(np.var(diff)))

            if len(tau) < 2 or np.var(tau) == 0:
                return np.nan

            log_lags = np.log(list(lags))
            log_tau = np.log(tau)
            slope, _ = np.polyfit(log_lags, log_tau, 1)
            return float(slope)
        except Exception:
            return np.nan

    # ------------------------------------------------------------------ #
    #  Volatility clustering                                               #
    # ------------------------------------------------------------------ #

    def _add_volatility_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0)

        # GARCH-like conditional variance proxy
        sq = log_ret ** 2
        df["ewma_variance"] = sq.ewm(span=30, adjust=False).mean()
        df["vol_of_vol"] = df["ewma_variance"].rolling(20).std()

        # Volatility regime: high / medium / low
        vol_20 = log_ret.rolling(20).std() * np.sqrt(252)
        vol_q33 = vol_20.rolling(252).quantile(0.33)
        vol_q66 = vol_20.rolling(252).quantile(0.66)
        df["vol_regime"] = np.select(
            [vol_20 <= vol_q33, vol_20 <= vol_q66],
            [0, 1],  # 0=low, 1=medium, 2=high
            default=2,
        )

        return df

    # ------------------------------------------------------------------ #
    #  Drawdown features                                                   #
    # ------------------------------------------------------------------ #

    def _add_drawdown_features(self, df: pd.DataFrame) -> pd.DataFrame:
        rolling_max = df["close"].rolling(252, min_periods=1).max()
        df["drawdown_pct"] = df["close"] / rolling_max - 1

        # Max drawdown in rolling window
        df["max_drawdown_60"] = df["drawdown_pct"].rolling(60).min()

        # Recovery factor
        df["recovery_factor"] = df["close"] / df["close"].shift(60) - 1

        return df

    # ------------------------------------------------------------------ #
    #  Entropy features                                                    #
    # ------------------------------------------------------------------ #

    def _add_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0)

        # Approximate entropy (complexity measure)
        df["approx_entropy"] = log_ret.rolling(50).apply(
            self._approx_entropy, raw=True
        )

        return df

    @staticmethod
    def _approx_entropy(series: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
        """Compute approximate entropy of a series."""
        if len(series) < m + 1:
            return np.nan
        try:
            std = np.std(series)
            if std == 0:
                return 0.0
            r = r_factor * std
            n = len(series)

            def phi(m_val: int) -> float:
                patterns = np.array([series[i:i + m_val] for i in range(n - m_val + 1)])
                count = np.array([
                    np.sum(np.max(np.abs(patterns - patterns[j]), axis=1) <= r)
                    for j in range(len(patterns))
                ])
                return np.sum(np.log(count / (n - m_val + 1))) / (n - m_val + 1)

            return abs(phi(m) - phi(m + 1))
        except Exception:
            return np.nan

    # ------------------------------------------------------------------ #
    #  Macro correlations                                                  #
    # ------------------------------------------------------------------ #

    def _add_macro_correlations(self, df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling correlation of asset returns with macro series."""
        asset_ret = df["close"].pct_change()
        macro_daily = macro_df.resample("D").ffill().reindex(df.index, method="ffill")

        for col in ["VIX", "DXY", "US10Y"]:
            if col in macro_daily.columns:
                macro_ret = macro_daily[col].pct_change()
                df[f"corr_{col}_30d"] = asset_ret.rolling(30).corr(macro_ret)

        return df
