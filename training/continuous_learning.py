"""
Continuous learning module.
Analyzes prediction errors, detects accuracy degradation, and triggers retraining.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import Settings
from data.storage.database import Database, init_db
from data.pipeline import DataPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


class ContinuousLearner:
    """
    Monitors model performance over time and triggers adaptive retraining.

    Workflow:
    1. Load stored predictions with actual outcomes
    2. Compute accuracy metrics per model / asset class
    3. Identify degradation or systematic biases
    4. Trigger retraining if threshold crossed
    5. Update ensemble weights online
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.db = init_db(settings)
        self.cfg = self.settings.continuous_learning
        self.reports_dir = self.settings.data.predictions_path
        os.makedirs(self.reports_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Outcome recording                                                   #
    # ------------------------------------------------------------------ #

    def record_outcomes(self, data_pipeline: Optional[DataPipeline] = None) -> int:
        """
        Match stored predictions to actual price outcomes.
        Updates prediction records with actual_return and correct_direction.
        Returns number of records updated.
        """
        dp = data_pipeline or DataPipeline(self.settings)

        # Load predictions that don't yet have outcomes
        with self.db.get_session() as session:
            from data.storage.database import PredictionRecord
            pending = (
                session.query(PredictionRecord)
                .filter(PredictionRecord.actual_return.is_(None))
                .filter(
                    PredictionRecord.predicted_at
                    <= datetime.now(timezone.utc) - timedelta(days=1)
                )
                .all()
            )

        updated = 0
        for pred in pending:
            try:
                horizon_map = {"1h": "1h", "4h": "4h", "1d": "1d", "1w": "1d"}
                tf = horizon_map.get(pred.horizon or "1d", "1d")
                ohlcv = dp.load_ohlcv(pred.asset, timeframe=tf)
                if ohlcv.empty:
                    continue

                # Find the bar at prediction time
                pred_ts = pd.Timestamp(pred.predicted_at)
                future_bars = {"1h": 1, "4h": 4, "1d": 1, "1w": 5}
                n_bars = future_bars.get(pred.horizon or "1d", 1)

                after = ohlcv[ohlcv.index >= pred_ts]
                if len(after) <= n_bars:
                    continue

                entry_price = after.iloc[0]["close"]
                exit_price = after.iloc[n_bars]["close"]
                actual_return = (exit_price - entry_price) / entry_price
                correct = 1 if (
                    (pred.prob_up > 0.5 and actual_return > 0) or
                    (pred.prob_down > 0.5 and actual_return < 0)
                ) else 0

                self.db.update_prediction_outcome(pred.id, float(actual_return), correct)
                updated += 1

            except Exception as e:
                logger.debug(f"Could not record outcome for prediction {pred.id}: {e}")

        logger.info(f"Recorded outcomes for {updated} predictions")
        return updated

    # ------------------------------------------------------------------ #
    #  Performance analysis                                                #
    # ------------------------------------------------------------------ #

    def analyze_performance(self, days_back: int = 30) -> dict:
        """
        Analyze prediction accuracy over the last N days.
        Returns performance report dict.
        """
        df = self.db.load_predictions_for_evaluation(days_back=days_back)
        if df.empty:
            logger.warning("No evaluated predictions found")
            return {"status": "no_data"}

        overall_accuracy = float(df["correct_direction"].mean())
        mean_confidence = float(df["confidence"].mean())

        # By asset class
        by_class = (
            df.groupby("asset_class")["correct_direction"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "n_predictions"})
        ).to_dict(orient="index")

        # By horizon
        by_horizon = (
            df.groupby("horizon")["correct_direction"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "n_predictions"})
        ).to_dict(orient="index")

        # Confidence calibration check
        df["conf_bin"] = pd.cut(df["confidence"], bins=5)
        calibration = df.groupby("conf_bin")["correct_direction"].mean().to_dict()

        # Worst-performing assets
        asset_acc = df.groupby("asset")["correct_direction"].mean()
        worst_assets = asset_acc.nsmallest(10).to_dict()

        report = {
            "analysis_date": datetime.now(timezone.utc).isoformat(),
            "days_analyzed": days_back,
            "n_predictions": len(df),
            "overall_accuracy": overall_accuracy,
            "mean_confidence": mean_confidence,
            "by_asset_class": by_class,
            "by_horizon": by_horizon,
            "confidence_calibration": {str(k): v for k, v in calibration.items()},
            "worst_assets": worst_assets,
            "needs_retraining": self._should_retrain(overall_accuracy),
        }

        self._save_report(report)
        logger.info(
            f"Performance analysis: accuracy={overall_accuracy:.3f}, "
            f"n={len(df)}, retraining_needed={report['needs_retraining']}"
        )
        return report

    def _should_retrain(self, current_accuracy: float) -> bool:
        """Check if accuracy has degraded below the threshold."""
        threshold = self.cfg.accuracy_degradation_threshold
        # Target accuracy (60%+ is acceptable, below 55% triggers retrain)
        return current_accuracy < (0.60 - threshold)

    # ------------------------------------------------------------------ #
    #  Analyze and retrain                                                 #
    # ------------------------------------------------------------------ #

    def analyze_and_retrain(
        self,
        force: bool = False,
        data_pipeline: Optional[DataPipeline] = None,
    ) -> dict:
        """
        Full continuous learning cycle:
        1. Record outcomes
        2. Analyze performance
        3. Retrain if needed
        """
        # 1. Record outcomes for pending predictions
        n_updated = self.record_outcomes(data_pipeline)
        logger.info(f"Updated {n_updated} prediction outcomes")

        # 2. Analyze
        report = self.analyze_performance(days_back=self.cfg.eval_lookback_days)

        if report.get("status") == "no_data":
            return report

        # 3. Conditionally retrain
        if force or report.get("needs_retraining", False):
            logger.info("Performance degradation detected — triggering retraining…")
            from training.trainer import ModelTrainer
            trainer = ModelTrainer(self.settings)
            training_result = trainer.train_all()
            report["retraining"] = training_result
            report["retrained"] = True
        else:
            logger.info("Model performance acceptable — no retraining needed")
            report["retrained"] = False

        return report

    # ------------------------------------------------------------------ #
    #  Bias detection                                                      #
    # ------------------------------------------------------------------ #

    def detect_systematic_bias(self, days_back: int = 60) -> dict:
        """
        Detect if the model has systematic biases:
        - Always predicting up (class imbalance in predictions)
        - Overconfident in certain conditions
        - Worse in high-volatility periods
        """
        df = self.db.load_predictions_for_evaluation(days_back=days_back)
        if df.empty:
            return {}

        # Direction bias
        up_preds = (df["prob_up"] > 0.5).mean()
        direction_bias = float(up_preds)

        # Accuracy by confidence quartile
        df["conf_quartile"] = pd.qcut(df["confidence"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
        acc_by_conf = df.groupby("conf_quartile")["correct_direction"].mean().to_dict()

        # Check for overconfidence (high confidence but low accuracy)
        overconfident = float(df["confidence"].mean()) > 0.75 and float(df["correct_direction"].mean()) < 0.55

        bias_report = {
            "direction_bias_up": direction_bias,
            "direction_bias_balanced": abs(direction_bias - 0.5) < 0.1,
            "accuracy_by_confidence_quartile": {str(k): v for k, v in acc_by_conf.items()},
            "overconfident": overconfident,
        }

        logger.info(f"Bias analysis: direction_bias={direction_bias:.2f}, overconfident={overconfident}")
        return bias_report

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _save_report(self, report: dict) -> None:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.reports_dir, f"performance_report_{date_str}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.debug(f"Performance report saved: {path}")
