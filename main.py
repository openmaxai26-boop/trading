"""
Main entry point for the Autonomous Quant AI Trading System.

Usage:
    python main.py                          # Full pipeline run
    python main.py --assets crypto          # Crypto only
    python main.py --assets equity          # Equities only
    python main.py --limit 100              # Limit to 100 assets
    python main.py --dry-run               # No DB writes
    python main.py --mode data             # Data collection only
    python main.py --mode train            # Model training only
    python main.py --mode signals          # Signal generation only
    python main.py --mode learn            # Continuous learning only
    python main.py --mode discover         # Strategy discovery
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

from config.settings import Settings
from utils.logger import get_logger, setup_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous Quant AI Trading System")
    parser.add_argument("--mode", choices=["full", "data", "train", "signals", "learn", "discover"],
                        default="full", help="Pipeline mode")
    parser.add_argument("--assets", choices=["all", "crypto", "equity"], default="all",
                        help="Asset universe to process")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of assets")
    parser.add_argument("--horizon", default="1d", help="Prediction horizon")
    parser.add_argument("--dry-run", action="store_true", help="Skip DB writes")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def run_data_pipeline(settings: Settings, asset_types: list[str], limit: int | None) -> None:
    """Run data collection pipeline."""
    from data.pipeline import DataPipeline
    logger.info("=" * 60)
    logger.info("PHASE 1: DATA COLLECTION")
    logger.info("=" * 60)
    pipeline = DataPipeline(settings)
    counts = pipeline.run(asset_types=asset_types, limit=limit)
    logger.info(f"Data collection complete: {counts}")


def run_training(settings: Settings) -> None:
    """Train all models."""
    from training.trainer import ModelTrainer
    logger.info("=" * 60)
    logger.info("PHASE 2: MODEL TRAINING")
    logger.info("=" * 60)
    trainer = ModelTrainer(settings)
    result = trainer.train_all()
    logger.info(f"Training complete: {result}")


def run_signal_generation(
    settings: Settings, horizon: str, save_to_db: bool
) -> list:
    """Generate trading signals."""
    from predictions.signal_generator import SignalGenerator
    from predictions.output_formatter import OutputFormatter
    logger.info("=" * 60)
    logger.info("PHASE 3: SIGNAL GENERATION")
    logger.info("=" * 60)
    generator = SignalGenerator(settings)
    signals = generator.run(horizon=horizon, save_to_db=save_to_db)

    if signals:
        formatter = OutputFormatter(settings)
        paths = formatter.format_all(signals, formats=["json", "csv", "text"])
        logger.info(f"Signal reports generated: {paths}")

    return signals


def run_continuous_learning(settings: Settings) -> None:
    """Run continuous learning cycle."""
    from training.continuous_learning import ContinuousLearner
    logger.info("=" * 60)
    logger.info("PHASE 4: CONTINUOUS LEARNING")
    logger.info("=" * 60)
    learner = ContinuousLearner(settings)
    result = learner.analyze_and_retrain()
    logger.info(f"Continuous learning result: {result}")


def run_strategy_discovery(settings: Settings) -> None:
    """Run strategy discovery (GA + RL)."""
    from data.pipeline import DataPipeline
    from utils.strategy_discovery import StrategyDiscovery
    logger.info("=" * 60)
    logger.info("STRATEGY DISCOVERY")
    logger.info("=" * 60)

    dp = DataPipeline(settings)
    # Use BTC as sample for discovery
    ohlcv = dp.load_ohlcv("BTC/USDT", timeframe="1d")
    if ohlcv.empty:
        logger.warning("No data for strategy discovery — run data pipeline first")
        return

    from features.feature_pipeline import FeaturePipeline
    fp = FeaturePipeline(settings)
    features = fp.transform(ohlcv)

    discovery = StrategyDiscovery(settings)
    strategies = discovery.discover_strategies(ohlcv, features=features)
    logger.info(f"Discovered {len(strategies)} valid strategies")

    for strat in strategies:
        logger.info(f"  Strategy: {strat['name']} | Sharpe: {strat.get('sharpe', 'N/A')} | "
                    f"PF: {strat.get('profit_factor', 'N/A')}")


def main() -> int:
    args = parse_args()

    # Configure logging
    setup_logger(log_level=args.log_level)

    # Build settings
    settings = Settings()
    if args.dry_run:
        settings.dry_run = True
        logger.info("DRY-RUN mode enabled — no database writes")

    # Determine asset types
    asset_types = {
        "all": ["crypto", "equity"],
        "crypto": ["crypto"],
        "equity": ["equity"],
    }.get(args.assets, ["crypto", "equity"])

    start_time = time.time()
    logger.info(f"{'='*60}")
    logger.info(f"AUTONOMOUS QUANT AI TRADING SYSTEM")
    logger.info(f"Mode: {args.mode} | Assets: {args.assets} | Horizon: {args.horizon}")
    logger.info(f"Started: {datetime.now(timezone.utc).isoformat()}")
    logger.info(f"{'='*60}")

    try:
        mode = args.mode

        if mode in ("full", "data"):
            run_data_pipeline(settings, asset_types, args.limit)

        if mode in ("full", "train"):
            run_training(settings)

        if mode in ("full", "signals"):
            run_signal_generation(settings, args.horizon, save_to_db=not args.dry_run)

        if mode in ("full", "learn"):
            run_continuous_learning(settings)

        if mode == "discover":
            run_strategy_discovery(settings)

        elapsed = time.time() - start_time
        logger.info(f"\nPipeline complete in {elapsed:.1f}s")
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
