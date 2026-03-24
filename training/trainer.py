"""
Model training orchestrator.
Handles data splitting, training all models, evaluation, and checkpointing.
"""

import json
import os
import random
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config.settings import Settings
from data.pipeline import DataPipeline
from features.feature_pipeline import FeaturePipeline
from models.ensemble import EnsembleModel
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Orchestrates end-to-end training for the ensemble model.

    Workflow:
    1. Load OHLCV + macro data
    2. Build features
    3. Split into train/val/test
    4. Train ensemble
    5. Evaluate and save metadata
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self._set_seeds()
        self.data_pipeline = DataPipeline(settings)
        self.feature_pipeline = FeaturePipeline(settings)
        self.ensemble = EnsembleModel(settings)
        self.metadata_dir = os.path.join(self.settings.data.models_path, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)

    def _set_seeds(self) -> None:
        seed = self.settings.random_seed
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    #  Main training entry                                                 #
    # ------------------------------------------------------------------ #

    def train_all(
        self,
        symbols: Optional[list[str]] = None,
        horizon: int = 1,
    ) -> dict:
        """
        Train the ensemble model on all available symbols.
        Returns training summary.
        """
        if symbols is None:
            # Use default crypto + equity universe
            symbols = self._get_default_symbols()

        logger.info(f"Training on {len(symbols)} symbols, horizon={horizon} bars")

        all_X_seq, all_X_tab, all_y = [], [], []
        for symbol in symbols:
            try:
                X_seq, X_tab, y = self._prepare_data_for_symbol(symbol, horizon)
                if len(y) > 0:
                    all_X_seq.append(X_seq)
                    all_X_tab.append(X_tab)
                    all_y.append(y)
            except Exception as e:
                logger.warning(f"Skipping {symbol}: {e}")

        if not all_y:
            logger.error("No data available for training")
            return {"error": "no_data"}

        X_seq = np.concatenate(all_X_seq, axis=0)
        X_tab = np.concatenate(all_X_tab, axis=0)
        y = np.concatenate(all_y, axis=0)

        logger.info(f"Combined dataset: X_seq={X_seq.shape}, X_tab={X_tab.shape}, y={y.shape}")

        # Shuffle (preserving temporal structure within each symbol is done by concatenation order)
        idx = np.random.permutation(len(y))
        X_seq, X_tab, y = X_seq[idx], X_tab[idx], y[idx]

        # Train/val/test split (no shuffling to respect time order)
        n = len(y)
        train_end = int(n * (1 - self.settings.model.validation_split - self.settings.model.test_split))
        val_end = int(n * (1 - self.settings.model.test_split))

        X_seq_train, X_seq_val, X_seq_test = X_seq[:train_end], X_seq[train_end:val_end], X_seq[val_end:]
        X_tab_train, X_tab_val, X_tab_test = X_tab[:train_end], X_tab[train_end:val_end], X_tab[val_end:]
        y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

        logger.info(f"Splits — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")

        # Train
        history = self.ensemble.fit(
            X_seq_train, X_seq_val, X_tab_train, X_tab_val, y_train, y_val
        )

        # Test evaluation
        test_preds = self.ensemble.predict(X_seq_test, X_tab_test)
        test_accuracy = float(np.mean(test_preds == y_test))
        logger.info(f"Test accuracy: {test_accuracy:.4f}")

        # Save metadata
        metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "n_symbols": len(symbols),
            "horizon": horizon,
            "test_accuracy": test_accuracy,
            "ensemble_weights": self.ensemble._weight_summary(),
            "training_history": {k: v.get("best_val_acc", 0) for k, v in history.items()},
        }
        self._save_metadata(metadata)

        return metadata

    def train_single_symbol(
        self, symbol: str, horizon: int = 1
    ) -> dict:
        """Train on a single symbol (useful for fine-tuning)."""
        X_seq, X_tab, y = self._prepare_data_for_symbol(symbol, horizon)
        if len(y) == 0:
            return {"error": "insufficient_data"}

        n = len(y)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        history = self.ensemble.fit(
            X_seq[:train_end], X_seq[train_end:val_end],
            X_tab[:train_end], X_tab[train_end:val_end],
            y[:train_end], y[train_end:val_end],
        )
        logger.info(f"Single-symbol training complete for {symbol}")
        return history

    # ------------------------------------------------------------------ #
    #  Data preparation                                                    #
    # ------------------------------------------------------------------ #

    def _prepare_data_for_symbol(
        self, symbol: str, horizon: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load, featurize, and build (X_seq, X_tab, y) for a symbol."""
        ohlcv = self.data_pipeline.load_ohlcv(
            symbol, lookback_days=self.settings.data.lookback_days
        )
        if ohlcv.empty or len(ohlcv) < self.settings.data.min_bars_required:
            return np.empty((0,)), np.empty((0,)), np.empty((0,))

        macro_df = self.data_pipeline.load_macro()
        features_df = self.feature_pipeline.transform(
            ohlcv, macro_df=macro_df if not macro_df.empty else None
        )

        if features_df.empty:
            return np.empty((0,)), np.empty((0,)), np.empty((0,))

        # Sequence dataset (for DL models)
        X_seq, y = self.feature_pipeline.build_sequence_dataset(
            features_df, horizon=horizon, target_type="direction"
        )

        # Tabular dataset (for XGBoost — last feature vector per sequence)
        X_tab, _ = self.feature_pipeline.build_training_dataset(
            features_df, horizon=horizon, target_type="direction"
        )
        # Align lengths (sequence dataset is shorter by seq_len)
        min_n = min(len(X_seq), len(X_tab))
        offset = len(X_tab) - min_n
        X_tab = X_tab[offset:]
        y_aligned = y[:min_n]

        return X_seq[:min_n], X_tab, y_aligned

    def _get_default_symbols(self) -> list[str]:
        """Return available symbols from the data lake."""
        lake = self.settings.data.data_lake_path
        ohlcv_dir = os.path.join(lake, "ohlcv", "1d")
        if not os.path.exists(ohlcv_dir):
            return ["BTC/USDT", "ETH/USDT", "SPY", "QQQ"]
        files = [f.replace("_", "/").replace(".parquet", "") for f in os.listdir(ohlcv_dir) if f.endswith(".parquet")]
        return files[:200]  # Cap for training

    def _save_metadata(self, metadata: dict) -> None:
        path = os.path.join(self.metadata_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Training metadata saved: {path}")
