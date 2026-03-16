"""
Feature pipeline — orchestrates all feature engineering for a given asset.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from config.settings import Settings
from features.technical_indicators import TechnicalIndicators
from features.statistical_features import StatisticalFeatures
from features.temporal_embeddings import TemporalEmbeddingGenerator
from utils.logger import get_logger

logger = get_logger(__name__)


class FeaturePipeline:
    """
    Full feature engineering pipeline for a single asset.

    Usage:
        pipeline = FeaturePipeline()
        features_df = pipeline.transform(ohlcv_df, macro_df=macro_df)
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        cfg = settings.features if settings else self.settings.features

        self.tech = TechnicalIndicators(
            rsi_periods=cfg.rsi_periods,
            ma_periods=cfg.ma_periods,
            bb_period=cfg.bb_period,
            bb_std=cfg.bb_std,
            atr_period=cfg.atr_period,
            macd_fast=cfg.macd_fast,
            macd_slow=cfg.macd_slow,
            macd_signal=cfg.macd_signal,
        )
        self.stat = StatisticalFeatures()
        self.embedder = TemporalEmbeddingGenerator(settings)
        self.scaler = RobustScaler()
        self._fitted = False

    def transform(
        self,
        ohlcv: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
        include_embeddings: bool = True,
        scale: bool = False,
    ) -> pd.DataFrame:
        """
        Full feature engineering on an OHLCV DataFrame.

        Args:
            ohlcv:              Raw OHLCV data
            macro_df:           Optional macro indicators aligned by date
            include_embeddings: Include autoencoder temporal embeddings
            scale:              Normalize features with RobustScaler

        Returns:
            DataFrame with all features, index aligned to ohlcv.
        """
        if ohlcv.empty:
            logger.warning("Empty OHLCV — returning empty feature DataFrame")
            return pd.DataFrame()

        # Validate columns
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(ohlcv.columns):
            raise ValueError(f"OHLCV must contain: {required}")

        logger.debug(f"Computing features on {len(ohlcv)} bars")

        # 1. Technical indicators
        df = self.tech.compute(ohlcv)

        # 2. Statistical features
        df = self.stat.compute(df, macro_df=macro_df)

        # 3. Temporal embeddings (optional, slow)
        if include_embeddings:
            try:
                emb_df = self.embedder.generate_embeddings(df)
                if not emb_df.empty:
                    df = df.join(emb_df, how="left")
            except Exception as e:
                logger.warning(f"Temporal embeddings failed: {e}")

        # 4. Macro alignment
        if macro_df is not None and not macro_df.empty:
            macro_aligned = macro_df.resample("D").ffill().reindex(df.index, method="ffill")
            for col in macro_aligned.columns:
                df[f"macro_{col}"] = macro_aligned[col]

        # 5. Clean up
        df = self._clean(df)

        # 6. Optional scaling
        if scale:
            df = self._scale(df)

        logger.debug(f"Feature pipeline complete: {df.shape[1]} features, {df.shape[0]} rows")
        return df

    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """Return list of non-OHLCV feature column names."""
        base_cols = {"open", "high", "low", "close", "volume"}
        return [c for c in df.columns if c not in base_cols]

    def build_training_dataset(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        target_type: str = "direction",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build (X, y) arrays from a feature DataFrame.

        Args:
            df:           Feature DataFrame (output of transform)
            horizon:      Prediction horizon in bars
            target_type:  'direction' (0/1) or 'return' (float)

        Returns:
            (X, y) numpy arrays, NaN rows removed.
        """
        feature_cols = self.get_feature_names(df)
        X = df[feature_cols].copy()

        # Compute target
        future_return = df["close"].pct_change(horizon).shift(-horizon)
        if target_type == "direction":
            y = (future_return > 0).astype(int)
        else:
            y = future_return

        # Align and drop NaN
        combined = pd.concat([X, y.rename("target")], axis=1).dropna()
        X_arr = combined[feature_cols].values.astype(np.float32)
        y_arr = combined["target"].values

        logger.debug(f"Training dataset: X={X_arr.shape}, y={y_arr.shape}")
        return X_arr, y_arr

    def build_sequence_dataset(
        self,
        df: pd.DataFrame,
        seq_len: Optional[int] = None,
        horizon: int = 1,
        target_type: str = "direction",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build (X, y) with X as 3D sequences for LSTM/Transformer models.

        Returns:
            X: (n_samples, seq_len, n_features)
            y: (n_samples,)
        """
        seq_len = seq_len or self.settings.features.sequence_length
        feature_cols = self.get_feature_names(df)
        data = df[feature_cols].fillna(0).values.astype(np.float32)

        future_return = df["close"].pct_change(horizon).shift(-horizon).values
        if target_type == "direction":
            targets = (future_return > 0).astype(np.float32)
        else:
            targets = future_return.astype(np.float32)

        X_seqs, y_seqs = [], []
        for i in range(seq_len, len(data) - horizon):
            if not np.isnan(targets[i]):
                X_seqs.append(data[i - seq_len: i])
                y_seqs.append(targets[i])

        if not X_seqs:
            return np.empty((0, seq_len, len(feature_cols))), np.empty(0)

        return np.array(X_seqs), np.array(y_seqs)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill, clip extreme outliers, and drop all-NaN columns."""
        # Drop columns where >50% are NaN
        thresh = int(len(df) * 0.5)
        df = df.dropna(axis=1, thresh=thresh)

        # Forward-fill remaining NaN
        df = df.ffill().bfill()

        # Clip extreme values at 5 std from mean (per column)
        numeric = df.select_dtypes(include=[np.number])
        for col in numeric.columns:
            mean = numeric[col].mean()
            std = numeric[col].std()
            if std > 0:
                df[col] = df[col].clip(lower=mean - 5 * std, upper=mean + 5 * std)

        return df

    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply RobustScaler to numeric feature columns."""
        base_cols = {"open", "high", "low", "close", "volume"}
        feature_cols = [c for c in df.columns if c not in base_cols]
        numeric = df[feature_cols].select_dtypes(include=[np.number])

        if not self._fitted:
            scaled = self.scaler.fit_transform(numeric)
            self._fitted = True
        else:
            scaled = self.scaler.transform(numeric)

        df[numeric.columns] = scaled
        return df
