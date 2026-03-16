"""
Ensemble model — combines LSTM, Transformer, CNN, GNN + XGBoost.
Uses weighted averaging based on recent validation performance.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from config.settings import Settings
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.cnn_model import CNNModel
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelWeight:
    model_name: str
    weight: float = 1.0
    val_accuracy: float = 0.0
    predictions_count: int = 0


class EnsembleModel:
    """
    Stacking / weighted averaging ensemble for direction prediction.

    Base models:
    - LSTM (bidirectional with attention)
    - Transformer (temporal)
    - CNN (multi-scale 1D)
    - XGBoost (tabular features)

    Meta-learner: Logistic Regression (stacking)
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.lstm = LSTMModel(settings, model_name="lstm")
        self.gru = LSTMModel(settings, use_gru=True, model_name="gru")
        self.transformer = TransformerModel(settings)
        self.cnn = CNNModel(settings)

        # XGBoost for tabular (last-row feature vector)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.settings.random_seed,
            eval_metric="logloss",
            early_stopping_rounds=30,
            use_label_encoder=False,
        )

        # Meta-learner for stacking
        self.meta_learner = LogisticRegression(C=1.0, max_iter=1000)
        self._meta_trained = False

        # Model weights (updated based on performance)
        self.weights: dict[str, ModelWeight] = {
            "lstm": ModelWeight("lstm", weight=1.0),
            "gru": ModelWeight("gru", weight=1.0),
            "transformer": ModelWeight("transformer", weight=1.0),
            "cnn": ModelWeight("cnn", weight=1.0),
            "xgb": ModelWeight("xgb", weight=1.0),
        }

        self._weights_path = os.path.join(
            self.settings.data.models_path, "ensemble_weights.json"
        )
        self._load_weights()

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_seq_train: np.ndarray,
        X_seq_val: np.ndarray,
        X_tab_train: np.ndarray,
        X_tab_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """
        Train all base models.

        Args:
            X_seq_*:  3D sequences (n, seq_len, features) for DL models
            X_tab_*:  2D tabular (n, features) for XGBoost
            y_*:      Binary labels
        """
        logger.info("Training ensemble base models…")
        history = {}

        # LSTM
        logger.info("Training LSTM…")
        h = self.lstm.fit(X_seq_train, y_train, X_seq_val, y_val)
        self.weights["lstm"].val_accuracy = h.get("best_val_acc", 0.5)
        history["lstm"] = h

        # GRU
        logger.info("Training GRU…")
        h = self.gru.fit(X_seq_train, y_train, X_seq_val, y_val)
        self.weights["gru"].val_accuracy = h.get("best_val_acc", 0.5)
        history["gru"] = h

        # Transformer
        logger.info("Training Transformer…")
        h = self.transformer.fit(X_seq_train, y_train, X_seq_val, y_val)
        self.weights["transformer"].val_accuracy = h.get("best_val_acc", 0.5)
        history["transformer"] = h

        # CNN
        logger.info("Training CNN…")
        h = self.cnn.fit(X_seq_train, y_train, X_seq_val, y_val)
        self.weights["cnn"].val_accuracy = h.get("best_val_acc", 0.5)
        history["cnn"] = h

        # XGBoost (tabular)
        logger.info("Training XGBoost…")
        self.xgb_model.fit(
            X_tab_train, y_train,
            eval_set=[(X_tab_val, y_val)],
            verbose=False,
        )
        xgb_preds = self.xgb_model.predict(X_tab_val)
        xgb_acc = np.mean(xgb_preds == y_val)
        self.weights["xgb"].val_accuracy = float(xgb_acc)
        history["xgb"] = {"best_val_acc": xgb_acc}
        logger.info(f"XGBoost val accuracy: {xgb_acc:.4f}")

        # Update weights from accuracy
        self._update_weights_from_accuracy()
        self._save_weights()

        # Train meta-learner (stacking)
        if self.settings.model.ensemble_method == "stacking":
            logger.info("Training stacking meta-learner…")
            meta_X = self._get_base_probs(X_seq_val, X_tab_val)
            self.meta_learner.fit(meta_X, y_val)
            self._meta_trained = True

        logger.info(f"Ensemble training complete. Model weights: {self._weight_summary()}")
        return history

    # ------------------------------------------------------------------ #
    #  Prediction                                                          #
    # ------------------------------------------------------------------ #

    def predict_proba(
        self,
        X_seq: np.ndarray,
        X_tab: np.ndarray,
    ) -> np.ndarray:
        """
        Get class probabilities from ensemble.
        Returns (n_samples, 2) array.
        """
        base_probs = self._get_base_probs(X_seq, X_tab)

        if self.settings.model.ensemble_method == "stacking" and self._meta_trained:
            return self.meta_learner.predict_proba(base_probs)

        # Weighted averaging
        weights = np.array([
            self.weights["lstm"].weight,
            self.weights["gru"].weight,
            self.weights["transformer"].weight,
            self.weights["cnn"].weight,
            self.weights["xgb"].weight,
        ])
        weights = weights / weights.sum()

        # base_probs: (n, 10) — 5 models × 2 classes each
        n = len(X_seq)
        model_probs = base_probs.reshape(n, 5, 2)  # (n, 5, 2)
        ensemble_probs = (model_probs * weights[:, np.newaxis]).sum(axis=1)  # (n, 2)
        return ensemble_probs

    def predict(self, X_seq: np.ndarray, X_tab: np.ndarray) -> np.ndarray:
        return self.predict_proba(X_seq, X_tab).argmax(axis=1)

    def _get_base_probs(self, X_seq: np.ndarray, X_tab: np.ndarray) -> np.ndarray:
        """Collect predictions from all base models. Returns (n, 10) array."""
        probs_list = []
        for model, name in [
            (self.lstm, "lstm"),
            (self.gru, "gru"),
            (self.transformer, "transformer"),
            (self.cnn, "cnn"),
        ]:
            try:
                p = model.predict_proba(X_seq)  # (n, 2)
                probs_list.append(p)
            except Exception as e:
                logger.warning(f"{name} prediction failed: {e}; using uniform")
                probs_list.append(np.full((len(X_seq), 2), 0.5))

        # XGBoost tabular
        try:
            p = self.xgb_model.predict_proba(X_tab)
            probs_list.append(p)
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}; using uniform")
            probs_list.append(np.full((len(X_tab), 2), 0.5))

        return np.concatenate(probs_list, axis=1)  # (n, 10)

    # ------------------------------------------------------------------ #
    #  Confidence calibration                                              #
    # ------------------------------------------------------------------ #

    def get_confidence(self, probs: np.ndarray) -> float:
        """
        Compute prediction confidence from probability distribution.
        High confidence = probabilities far from 0.5.
        """
        max_prob = np.max(probs)
        return float(max(0.0, (max_prob - 0.5) * 2))   # Scale to [0, 1]

    # ------------------------------------------------------------------ #
    #  Weight management                                                   #
    # ------------------------------------------------------------------ #

    def _update_weights_from_accuracy(self) -> None:
        """Set model weights proportional to their validation accuracy."""
        min_acc = 0.5  # Accuracy below this = zero weight
        for name, mw in self.weights.items():
            acc = mw.val_accuracy
            mw.weight = max(0.0, acc - min_acc) ** 2  # Squared to emphasize good models

        # Normalize
        total = sum(mw.weight for mw in self.weights.values())
        if total > 0:
            for mw in self.weights.values():
                mw.weight /= total
        else:
            for mw in self.weights.values():
                mw.weight = 1.0 / len(self.weights)

    def update_weight_online(self, model_name: str, recent_accuracy: float) -> None:
        """Update a single model's weight based on recent live performance."""
        if model_name in self.weights:
            old_weight = self.weights[model_name].val_accuracy
            # Exponential moving average
            self.weights[model_name].val_accuracy = 0.7 * old_weight + 0.3 * recent_accuracy
            self._update_weights_from_accuracy()
            self._save_weights()

    def _weight_summary(self) -> dict[str, float]:
        return {name: round(mw.weight, 4) for name, mw in self.weights.items()}

    def _save_weights(self) -> None:
        os.makedirs(os.path.dirname(self._weights_path), exist_ok=True)
        data = {
            name: {"weight": mw.weight, "val_accuracy": mw.val_accuracy}
            for name, mw in self.weights.items()
        }
        with open(self._weights_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_weights(self) -> None:
        if os.path.exists(self._weights_path):
            with open(self._weights_path) as f:
                data = json.load(f)
            for name, vals in data.items():
                if name in self.weights:
                    self.weights[name].weight = vals.get("weight", 1.0)
                    self.weights[name].val_accuracy = vals.get("val_accuracy", 0.5)
            logger.info(f"Loaded ensemble weights: {self._weight_summary()}")
