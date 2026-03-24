"""
Signal generator — combines model predictions, whale score, cycle analysis
into structured TradingSignal objects.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from backtesting.risk_manager import RiskManager
from config.settings import Settings
from data.pipeline import DataPipeline
from data.storage.cache import CacheManager
from features.feature_pipeline import FeaturePipeline
from models.ensemble import EnsembleModel
from utils.cycle_detector import CycleDetector
from utils.whale_detector import WhaleDetector
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradingSignal:
    asset: str
    asset_class: str                # equity | crypto | commodity | forex
    horizon: str                    # 1h | 4h | 1d | 1w
    prob_up: float                  # 0.0 - 1.0
    prob_down: float                # 0.0 - 1.0
    confidence: float               # 0.0 - 1.0
    whale_score: float              # 0.0 - 100.0
    explosion_score: float          # 0.0 - 100.0
    market_regime: str
    detected_cycles: list           # Dominant cycle periods
    recommended_strategy: str
    position_size_pct: float        # % of portfolio
    stop_loss_pct: float
    take_profit_pct: float
    timestamp: str
    current_price: float
    explanation: dict               # Human-readable factors


class SignalGenerator:
    """
    Produces TradingSignal for each asset in the universe.

    Pipeline:
    1. Load OHLCV + features from cache/DB
    2. Run ensemble model → (prob_up, prob_down, confidence)
    3. Compute whale_score + explosion_score
    4. Determine position size and stops via RiskManager
    5. Assemble and cache TradingSignal
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.data_pipeline = DataPipeline(settings)
        self.feature_pipeline = FeaturePipeline(settings)
        self.ensemble = EnsembleModel(settings)
        self.whale_detector = WhaleDetector(settings)
        self.cycle_detector = CycleDetector(settings)
        self.risk_manager = RiskManager(settings)
        self.cache = CacheManager(settings)
        self.signal_cfg = self.settings.signal
        self.output_dir = self.settings.data.predictions_path
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def run(
        self,
        symbols: Optional[list[str]] = None,
        asset_classes: Optional[dict[str, str]] = None,
        horizon: str = "1d",
        save_to_db: bool = True,
    ) -> list[TradingSignal]:
        """
        Generate signals for all symbols.

        Args:
            symbols:       List of asset symbols (defaults to full universe)
            asset_classes: {symbol: asset_class} mapping
            horizon:       Prediction horizon
            save_to_db:    Store prediction records in database

        Returns:
            List of TradingSignal objects sorted by confidence.
        """
        symbols = symbols or self._get_all_symbols()
        asset_classes = asset_classes or {}
        logger.info(f"Generating signals for {len(symbols)} symbols, horizon={horizon}")

        signals: list[TradingSignal] = []
        current_positions: dict[str, float] = {}

        for i, symbol in enumerate(symbols):
            try:
                signal = self._generate_signal(
                    symbol,
                    asset_class=asset_classes.get(symbol, self._infer_asset_class(symbol)),
                    horizon=horizon,
                    current_positions=current_positions,
                )
                if signal is not None:
                    signals.append(signal)
                    current_positions[symbol] = signal.position_size_pct

                    # Cache signal
                    self.cache.cache_signal(symbol, asdict(signal), ttl=3600)

                    # Save to DB
                    if save_to_db:
                        self._save_prediction_to_db(signal)

            except Exception as e:
                logger.warning(f"Signal generation failed for {symbol}: {e}")

            if (i + 1) % 100 == 0:
                logger.info(f"Signal progress: {i+1}/{len(symbols)}")

        # Sort by confidence (highest first)
        signals.sort(key=lambda s: s.confidence, reverse=True)

        # Save batch report
        self._save_signal_report(signals)

        logger.info(f"Signal generation complete: {len(signals)} signals produced")
        return signals

    # ------------------------------------------------------------------ #
    #  Per-symbol signal generation                                        #
    # ------------------------------------------------------------------ #

    def _generate_signal(
        self,
        symbol: str,
        asset_class: str,
        horizon: str,
        current_positions: dict[str, float],
    ) -> Optional[TradingSignal]:
        """Generate a single TradingSignal for one symbol."""
        # Load data
        ohlcv = self.data_pipeline.load_ohlcv(
            symbol, timeframe="1d", lookback_days=self.settings.data.lookback_days
        )
        if ohlcv.empty or len(ohlcv) < 60:
            return None

        macro_df = self.data_pipeline.load_macro()

        # Features
        features_df = self.feature_pipeline.transform(
            ohlcv, macro_df=macro_df if not macro_df.empty else None,
            include_embeddings=False,  # Skip for speed
        )
        if features_df.empty:
            return None

        # Current price + ATR
        current_price = float(ohlcv["close"].iloc[-1])
        atr = float(features_df.get("atr", pd.Series([current_price * 0.02])).iloc[-1])

        # Model prediction
        horizon_map = {"1h": 1, "4h": 4, "1d": 1, "1w": 5}
        n_bars = horizon_map.get(horizon, 1)
        seq_len = self.settings.features.sequence_length

        if len(features_df) < seq_len + n_bars:
            return None

        try:
            X_seq, _ = self.feature_pipeline.build_sequence_dataset(features_df, seq_len=seq_len, horizon=n_bars)
            X_tab, _ = self.feature_pipeline.build_training_dataset(features_df, horizon=n_bars)
            if len(X_seq) == 0:
                return None

            # Use only the last sample (most recent)
            probs = self.ensemble.predict_proba(X_seq[-1:], X_tab[-1:])
            prob_down = float(probs[0][0])
            prob_up = float(probs[0][1])
            confidence = self.ensemble.get_confidence(probs[0])
        except Exception as e:
            logger.debug(f"Model prediction failed for {symbol}: {e}")
            # Fallback: neutral
            prob_up, prob_down, confidence = 0.5, 0.5, 0.0

        # Confidence filter
        if confidence < self.signal_cfg.min_confidence_to_trade:
            return None

        # Whale score
        whale_score = self.whale_detector.compute_whale_score(ohlcv)

        # Cycle analysis
        cycle_analysis = self.cycle_detector.analyze(ohlcv)
        explosion_score = cycle_analysis["explosion_score"]
        market_regime = cycle_analysis["market_regime"]
        detected_cycles = cycle_analysis["dominant_cycles"]

        # Position sizing
        sizing = self.risk_manager.compute_position_size(
            symbol=symbol,
            current_price=current_price,
            prob_up=prob_up,
            prob_down=prob_down,
            atr=atr,
            portfolio_value=self.settings.backtest.initial_capital,
            current_positions=current_positions,
        )

        # Compute stop/TP as percentages
        stop_pct = abs(current_price - sizing.stop_loss_price) / current_price
        tp_pct = abs(current_price - sizing.take_profit_price) / current_price

        # Select recommended strategy
        strategy = self._select_strategy(
            market_regime, explosion_score, whale_score, prob_up
        )

        # Build explanation
        explanation = {
            "dominant_direction": "bullish" if prob_up > 0.5 else "bearish",
            "regime": market_regime,
            "is_squeeze": cycle_analysis["is_squeeze"],
            "breakout_probability": cycle_analysis["breakout_probability"],
            "trend_strength": cycle_analysis["trend_strength"],
            "smart_money_active": whale_score >= self.signal_cfg.whale_score_threshold,
            "explosion_likely": explosion_score >= self.signal_cfg.explosion_score_threshold,
        }

        return TradingSignal(
            asset=symbol,
            asset_class=asset_class,
            horizon=horizon,
            prob_up=round(prob_up, 4),
            prob_down=round(prob_down, 4),
            confidence=round(confidence, 4),
            whale_score=round(whale_score, 1),
            explosion_score=round(explosion_score, 1),
            market_regime=market_regime,
            detected_cycles=detected_cycles,
            recommended_strategy=strategy,
            position_size_pct=round(sizing.position_size_pct * 100, 2),
            stop_loss_pct=round(stop_pct * 100, 2),
            take_profit_pct=round(tp_pct * 100, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
            current_price=round(current_price, 8),
            explanation=explanation,
        )

    # ------------------------------------------------------------------ #
    #  Strategy selection logic                                            #
    # ------------------------------------------------------------------ #

    def _select_strategy(
        self,
        regime: str,
        explosion_score: float,
        whale_score: float,
        prob_up: float,
    ) -> str:
        """Select the most appropriate strategy given market conditions."""
        if explosion_score >= 70:
            return "volatility_breakout"
        if whale_score >= 70:
            return "smart_money_follow"
        if regime == "bull_trend" and prob_up > 0.6:
            return "trend_following_long"
        if regime == "bear_trend" and prob_up < 0.4:
            return "trend_following_short"
        if regime in ("range", "low_volatility"):
            return "mean_reversion"
        if regime == "high_volatility":
            return "volatility_scalping"
        return "neutral_hold"

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _infer_asset_class(self, symbol: str) -> str:
        if "/" in symbol or symbol.endswith("USDT") or symbol.endswith("BTC"):
            return "crypto"
        if symbol.startswith("^"):
            return "index"
        return "equity"

    def _get_all_symbols(self) -> list[str]:
        lake = self.settings.data.data_lake_path
        ohlcv_dir = os.path.join(lake, "ohlcv", "1d")
        if not os.path.exists(ohlcv_dir):
            return []
        symbols = [f.replace("_", "/").replace(".parquet", "") for f in os.listdir(ohlcv_dir) if f.endswith(".parquet")]
        return symbols[:self.signal_cfg.max_signals_per_run * 5]

    def _save_prediction_to_db(self, signal: TradingSignal) -> None:
        try:
            record = {
                "asset": signal.asset,
                "asset_class": signal.asset_class,
                "horizon": signal.horizon,
                "prob_up": signal.prob_up,
                "prob_down": signal.prob_down,
                "confidence": signal.confidence,
                "whale_score": signal.whale_score,
                "explosion_score": signal.explosion_score,
                "market_regime": signal.market_regime,
                "recommended_strategy": signal.recommended_strategy,
                "position_size_pct": signal.position_size_pct,
                "predicted_at": datetime.now(timezone.utc),
            }
            self.data_pipeline.db.save_prediction(record)
        except Exception as e:
            logger.debug(f"DB save failed for {signal.asset}: {e}")

    def _save_signal_report(self, signals: list[TradingSignal]) -> None:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"signals_{date_str}.json")
        data = [asdict(s) for s in signals[:self.signal_cfg.max_signals_per_run]]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Signal report saved: {path} ({len(signals)} signals)")
