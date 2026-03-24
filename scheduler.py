"""
Automated scheduler for daily pipeline execution.

Schedules:
- 06:00 UTC: Data collection (OHLCV, sentiment, on-chain)
- 07:00 UTC: Signal generation + report export
- 08:00 UTC: Continuous learning (outcome recording + drift detection)
- Sunday 02:00 UTC: Weekly model retraining
- 1st of month 03:00 UTC: Full model retrain + strategy discovery
"""

import sys
import os
from datetime import datetime, timezone

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

from config.settings import Settings
from utils.logger import get_logger, setup_logger

logger = get_logger(__name__)


def job_data_collection() -> None:
    """Daily data ingestion job."""
    logger.info("SCHEDULER: Starting data collection job")
    try:
        from data.pipeline import DataPipeline
        settings = Settings()
        pipeline = DataPipeline(settings)
        counts = pipeline.run(asset_types=["crypto", "equity"])
        logger.info(f"SCHEDULER: Data collection complete — {counts}")
    except Exception as e:
        logger.exception(f"SCHEDULER: Data collection failed: {e}")
        raise


def job_signal_generation() -> None:
    """Daily signal generation job."""
    logger.info("SCHEDULER: Starting signal generation job")
    try:
        from predictions.signal_generator import SignalGenerator
        from predictions.output_formatter import OutputFormatter
        settings = Settings()
        gen = SignalGenerator(settings)
        signals = gen.run(horizon="1d")
        if signals:
            fmt = OutputFormatter(settings)
            paths = fmt.format_all(signals, formats=["json", "csv", "text"])
            logger.info(f"SCHEDULER: {len(signals)} signals generated — reports: {paths}")
    except Exception as e:
        logger.exception(f"SCHEDULER: Signal generation failed: {e}")
        raise


def job_continuous_learning() -> None:
    """Daily continuous learning job."""
    logger.info("SCHEDULER: Starting continuous learning job")
    try:
        from training.continuous_learning import ContinuousLearner
        settings = Settings()
        learner = ContinuousLearner(settings)
        result = learner.analyze_and_retrain(force=False)
        if result.get("retrained"):
            logger.info("SCHEDULER: Models retrained due to performance degradation")
        else:
            logger.info("SCHEDULER: Continuous learning complete — no retraining needed")
    except Exception as e:
        logger.exception(f"SCHEDULER: Continuous learning failed: {e}")
        raise


def job_weekly_retrain() -> None:
    """Weekly full model retraining."""
    logger.info("SCHEDULER: Starting weekly model retrain")
    try:
        from training.trainer import ModelTrainer
        settings = Settings()
        trainer = ModelTrainer(settings)
        result = trainer.train_all()
        logger.info(f"SCHEDULER: Weekly retrain complete — {result}")
    except Exception as e:
        logger.exception(f"SCHEDULER: Weekly retrain failed: {e}")
        raise


def job_monthly_strategy_discovery() -> None:
    """Monthly strategy discovery and full retrain."""
    logger.info("SCHEDULER: Starting monthly strategy discovery")
    try:
        from data.pipeline import DataPipeline
        from features.feature_pipeline import FeaturePipeline
        from utils.strategy_discovery import StrategyDiscovery
        from training.trainer import ModelTrainer

        settings = Settings()
        dp = DataPipeline(settings)
        fp = FeaturePipeline(settings)

        ohlcv = dp.load_ohlcv("BTC/USDT", timeframe="1d")
        if not ohlcv.empty:
            features = fp.transform(ohlcv)
            discovery = StrategyDiscovery(settings)
            strategies = discovery.discover_strategies(ohlcv, features=features)
            logger.info(f"SCHEDULER: Discovered {len(strategies)} strategies")

        trainer = ModelTrainer(settings)
        result = trainer.train_all()
        logger.info(f"SCHEDULER: Monthly retrain complete — {result}")
    except Exception as e:
        logger.exception(f"SCHEDULER: Monthly job failed: {e}")
        raise


def job_performance_report() -> None:
    """Generate weekly performance analysis report."""
    logger.info("SCHEDULER: Generating performance report")
    try:
        from training.continuous_learning import ContinuousLearner
        settings = Settings()
        learner = ContinuousLearner(settings)
        report = learner.analyze_performance(days_back=7)
        bias = learner.detect_systematic_bias(days_back=30)
        logger.info(f"SCHEDULER: Performance report — accuracy={report.get('overall_accuracy', 'N/A'):.3f}")
        logger.info(f"SCHEDULER: Bias analysis — direction_bias={bias.get('direction_bias_up', 'N/A'):.2f}")
    except Exception as e:
        logger.exception(f"SCHEDULER: Performance report failed: {e}")


def on_job_error(event) -> None:
    """Alert on scheduler job failure."""
    logger.error(f"SCHEDULER ERROR: Job {event.job_id} failed — {event.exception}")


def on_job_executed(event) -> None:
    """Log successful job completion."""
    logger.info(f"SCHEDULER: Job {event.job_id} completed in {event.retval or 'N/A'}")


def main() -> None:
    setup_logger(log_level=os.getenv("LOG_LEVEL", "INFO"))
    settings = Settings()

    logger.info("Starting Autonomous Quant AI Scheduler")
    logger.info(f"Environment: {settings.environment}")

    scheduler = BlockingScheduler(timezone="UTC")

    # Register error/success handlers
    scheduler.add_listener(on_job_error, EVENT_JOB_ERROR)
    scheduler.add_listener(on_job_executed, EVENT_JOB_EXECUTED)

    # Daily jobs
    scheduler.add_job(
        job_data_collection,
        trigger="cron",
        hour=6, minute=0,
        id="data_collection",
        name="Daily Data Collection",
        max_instances=1,
        misfire_grace_time=3600,
    )

    scheduler.add_job(
        job_signal_generation,
        trigger="cron",
        hour=7, minute=0,
        id="signal_generation",
        name="Daily Signal Generation",
        max_instances=1,
        misfire_grace_time=3600,
    )

    scheduler.add_job(
        job_continuous_learning,
        trigger="cron",
        hour=8, minute=0,
        id="continuous_learning",
        name="Daily Continuous Learning",
        max_instances=1,
        misfire_grace_time=3600,
    )

    # Weekly retrain (Sunday at 02:00 UTC)
    scheduler.add_job(
        job_weekly_retrain,
        trigger="cron",
        day_of_week="sun",
        hour=2, minute=0,
        id="weekly_retrain",
        name="Weekly Model Retrain",
        max_instances=1,
        misfire_grace_time=7200,
    )

    # Weekly performance report (Friday at 18:00 UTC)
    scheduler.add_job(
        job_performance_report,
        trigger="cron",
        day_of_week="fri",
        hour=18, minute=0,
        id="performance_report",
        name="Weekly Performance Report",
        max_instances=1,
    )

    # Monthly strategy discovery (1st of month at 03:00 UTC)
    scheduler.add_job(
        job_monthly_strategy_discovery,
        trigger="cron",
        day=1,
        hour=3, minute=0,
        id="strategy_discovery",
        name="Monthly Strategy Discovery",
        max_instances=1,
        misfire_grace_time=7200,
    )

    logger.info("Scheduler configured:")
    for job in scheduler.get_jobs():
        logger.info(f"  {job.id}: {job.name} | next: {job.next_run_time}")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
