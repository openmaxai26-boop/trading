"""
Output formatter — generates structured reports from TradingSignals.
Supports JSON, CSV, and human-readable text formats.
"""

import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Optional

import pandas as pd

from config.settings import Settings
from predictions.signal_generator import TradingSignal
from utils.logger import get_logger

logger = get_logger(__name__)


class OutputFormatter:
    """Formats and exports trading signals in multiple output formats."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.output_dir = os.path.join(self.settings.data.predictions_path, "reports")
        os.makedirs(self.output_dir, exist_ok=True)

    def format_all(
        self,
        signals: list[TradingSignal],
        formats: Optional[list[str]] = None,
        run_id: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Export signals in all requested formats.
        Returns {format: file_path} dict.
        """
        formats = formats or self.settings.signal.output_formats
        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_paths = {}

        if "json" in formats:
            path = self._to_json(signals, run_id)
            output_paths["json"] = path

        if "csv" in formats:
            path = self._to_csv(signals, run_id)
            output_paths["csv"] = path

        if "text" in formats:
            path = self._to_text(signals, run_id)
            output_paths["text"] = path

        return output_paths

    # ------------------------------------------------------------------ #
    #  JSON output                                                         #
    # ------------------------------------------------------------------ #

    def _to_json(self, signals: list[TradingSignal], run_id: str) -> str:
        path = os.path.join(self.output_dir, f"signals_{run_id}.json")
        payload = {
            "run_id": run_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "signal_count": len(signals),
            "signals": [asdict(s) for s in signals],
            "summary": self._build_summary(signals),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"JSON report: {path}")
        return path

    # ------------------------------------------------------------------ #
    #  CSV output                                                          #
    # ------------------------------------------------------------------ #

    def _to_csv(self, signals: list[TradingSignal], run_id: str) -> str:
        path = os.path.join(self.output_dir, f"signals_{run_id}.csv")
        if not signals:
            return path

        fieldnames = [
            "asset", "asset_class", "horizon", "prob_up", "prob_down",
            "confidence", "whale_score", "explosion_score", "market_regime",
            "recommended_strategy", "position_size_pct", "stop_loss_pct",
            "take_profit_pct", "current_price", "timestamp",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sig in signals:
                row = {k: getattr(sig, k, "") for k in fieldnames}
                writer.writerow(row)
        logger.info(f"CSV report: {path}")
        return path

    # ------------------------------------------------------------------ #
    #  Human-readable text summary                                         #
    # ------------------------------------------------------------------ #

    def _to_text(self, signals: list[TradingSignal], run_id: str) -> str:
        path = os.path.join(self.output_dir, f"signals_{run_id}.txt")
        lines = [
            f"{'='*80}",
            f"AUTONOMOUS QUANT AI — TRADING SIGNALS REPORT",
            f"Run ID: {run_id}",
            f"Generated: {datetime.utcnow().isoformat()}Z",
            f"Total Signals: {len(signals)}",
            f"{'='*80}",
            "",
        ]

        # Summary statistics
        summary = self._build_summary(signals)
        lines += [
            "SUMMARY",
            f"  Bullish signals:   {summary['bullish_count']}",
            f"  Bearish signals:   {summary['bearish_count']}",
            f"  High confidence:   {summary['high_confidence_count']} (≥80%)",
            f"  Whale activity:    {summary['whale_active_count']} (score ≥65)",
            f"  Explosion alerts:  {summary['explosion_count']} (score ≥70)",
            f"  Regimes: {summary['regime_distribution']}",
            "",
            f"{'─'*80}",
            "TOP SIGNALS",
            f"{'─'*80}",
        ]

        # Top 20 signals
        for i, sig in enumerate(signals[:20], 1):
            direction = "▲ BULL" if sig.prob_up > 0.5 else "▼ BEAR"
            lines += [
                f"\n#{i:02d} {sig.asset} [{sig.asset_class}] — {direction}",
                f"    Price: {sig.current_price:.6g}  |  Horizon: {sig.horizon}",
                f"    P(up)={sig.prob_up:.1%}  P(down)={sig.prob_down:.1%}  Confidence={sig.confidence:.1%}",
                f"    Whale Score: {sig.whale_score:.0f}/100  |  Explosion Score: {sig.explosion_score:.0f}/100",
                f"    Regime: {sig.market_regime}  |  Strategy: {sig.recommended_strategy}",
                f"    Position: {sig.position_size_pct:.1f}%  |  SL: -{sig.stop_loss_pct:.1f}%  |  TP: +{sig.take_profit_pct:.1f}%",
                f"    Cycles: {sig.detected_cycles}",
                f"    Factors: {sig.explanation}",
            ]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Text report: {path}")
        return path

    # ------------------------------------------------------------------ #
    #  Summary statistics                                                  #
    # ------------------------------------------------------------------ #

    def _build_summary(self, signals: list[TradingSignal]) -> dict:
        if not signals:
            return {}

        bullish = sum(1 for s in signals if s.prob_up > 0.5)
        high_conf = sum(1 for s in signals if s.confidence >= 0.8)
        whale_active = sum(1 for s in signals if s.whale_score >= 65)
        explosion = sum(1 for s in signals if s.explosion_score >= 70)

        regime_dist: dict[str, int] = {}
        for s in signals:
            regime_dist[s.market_regime] = regime_dist.get(s.market_regime, 0) + 1

        return {
            "total": len(signals),
            "bullish_count": bullish,
            "bearish_count": len(signals) - bullish,
            "high_confidence_count": high_conf,
            "whale_active_count": whale_active,
            "explosion_count": explosion,
            "regime_distribution": regime_dist,
            "avg_confidence": round(sum(s.confidence for s in signals) / len(signals), 3),
            "avg_whale_score": round(sum(s.whale_score for s in signals) / len(signals), 1),
            "avg_explosion_score": round(sum(s.explosion_score for s in signals) / len(signals), 1),
        }

    # ------------------------------------------------------------------ #
    #  DataFrame conversion (for notebooks / analysis)                    #
    # ------------------------------------------------------------------ #

    def to_dataframe(self, signals: list[TradingSignal]) -> pd.DataFrame:
        """Convert signals list to a pandas DataFrame."""
        return pd.DataFrame([asdict(s) for s in signals])
