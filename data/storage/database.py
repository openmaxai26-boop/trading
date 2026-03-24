"""
Database layer — PostgreSQL via SQLAlchemy + Parquet data lake.
"""

import os
from datetime import datetime
from typing import Optional

import pandas as pd

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Optional SQLAlchemy — system works in Parquet-only mode if not installed
try:
    from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
    from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
    SQLALCHEMY_AVAILABLE = True

    class Base(DeclarativeBase):
        pass

except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy not installed — running in Parquet-only mode (no SQL persistence)")

    class Base:  # type: ignore[no-redef]
        pass


if SQLALCHEMY_AVAILABLE:
    class PredictionRecord(Base):
        """Stores every prediction for continuous learning."""
        __tablename__ = "predictions"
        id = Column(Integer, primary_key=True, autoincrement=True)
        asset = Column(String(50), nullable=False, index=True)
        asset_class = Column(String(20))
        horizon = Column(String(10))
        prob_up = Column(Float)
        prob_down = Column(Float)
        confidence = Column(Float)
        whale_score = Column(Float)
        explosion_score = Column(Float)
        market_regime = Column(String(20))
        recommended_strategy = Column(String(100))
        position_size_pct = Column(Float)
        predicted_at = Column(DateTime(timezone=True), nullable=False, index=True)
        actual_return = Column(Float, nullable=True)
        correct_direction = Column(Integer, nullable=True)

    class StrategyRecord(Base):
        __tablename__ = "strategies"
        id = Column(Integer, primary_key=True, autoincrement=True)
        name = Column(String(200), nullable=False)
        params_json = Column(Text)
        sharpe_ratio = Column(Float)
        max_drawdown = Column(Float)
        profit_factor = Column(Float)
        win_rate = Column(Float)
        total_trades = Column(Integer)
        backtest_start = Column(DateTime)
        backtest_end = Column(DateTime)
        created_at = Column(DateTime(timezone=True))
        is_active = Column(Integer, default=1)

    class ModelMetadata(Base):
        __tablename__ = "model_metadata"
        id = Column(Integer, primary_key=True, autoincrement=True)
        model_name = Column(String(100), nullable=False, index=True)
        version = Column(String(50))
        checkpoint_path = Column(String(500))
        train_accuracy = Column(Float)
        val_accuracy = Column(Float)
        features_json = Column(Text)
        hyperparams_json = Column(Text)
        trained_at = Column(DateTime(timezone=True))
        is_active = Column(Integer, default=1)
else:
    PredictionRecord = None  # type: ignore[assignment,misc]
    StrategyRecord = None    # type: ignore[assignment,misc]
    ModelMetadata = None     # type: ignore[assignment,misc]


class Database:
    """Manages PostgreSQL connections and Parquet data lake."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.lake_path = self.settings.data.data_lake_path
        os.makedirs(self.lake_path, exist_ok=True)

        if SQLALCHEMY_AVAILABLE:
            self.engine = create_engine(
                self.settings.database.url,
                pool_size=self.settings.database.pool_size,
                max_overflow=self.settings.database.max_overflow,
                echo=self.settings.database.echo,
            )
            self.SessionLocal = sessionmaker(bind=self.engine)
        else:
            self.engine = None
            self.SessionLocal = None

    def get_session(self):
        if not SQLALCHEMY_AVAILABLE or self.SessionLocal is None:
            return None
        return self.SessionLocal()

    # ------------------------------------------------------------------ #
    #  Parquet data lake                                                   #
    # ------------------------------------------------------------------ #

    def _ohlcv_path(self, symbol: str, timeframe: str) -> str:
        safe = symbol.replace("/", "_")
        return os.path.join(self.lake_path, "ohlcv", timeframe, f"{safe}.parquet")

    def save_ohlcv(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Save/append OHLCV data to Parquet."""
        path = self._ohlcv_path(symbol, timeframe)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(path):
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
        else:
            combined = df.sort_index()

        combined.to_parquet(path, engine="pyarrow", compression="snappy")
        logger.debug(f"Saved OHLCV: {symbol} {timeframe} → {len(combined)} bars")

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from Parquet."""
        path = self._ohlcv_path(symbol, timeframe)
        if not os.path.exists(path):
            logger.debug(f"No Parquet found for {symbol} {timeframe}")
            return pd.DataFrame()

        df = pd.read_parquet(path)
        if start:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        return df

    # ------------------------------------------------------------------ #
    #  Predictions CRUD                                                    #
    # ------------------------------------------------------------------ #

    def save_prediction(self, record: dict) -> None:
        """Insert a prediction record into the database (no-op if SQLAlchemy unavailable)."""
        if not SQLALCHEMY_AVAILABLE or self.SessionLocal is None:
            return
        session = self.SessionLocal()
        try:
            pred = PredictionRecord(**record)
            session.add(pred)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.debug(f"save_prediction failed: {e}")
        finally:
            session.close()

    def update_prediction_outcome(
        self, prediction_id: int, actual_return: float, correct_direction: int
    ) -> None:
        if not SQLALCHEMY_AVAILABLE or self.SessionLocal is None:
            return
        session = self.SessionLocal()
        try:
            pred = session.get(PredictionRecord, prediction_id)
            if pred:
                pred.actual_return = actual_return
                pred.correct_direction = correct_direction
                session.commit()
        except Exception as e:
            session.rollback()
            logger.debug(f"update_prediction_outcome failed: {e}")
        finally:
            session.close()

    def load_predictions_for_evaluation(
        self, days_back: int = 30, min_confidence: float = 0.0
    ) -> pd.DataFrame:
        """Load recent predictions for accuracy evaluation."""
        if not SQLALCHEMY_AVAILABLE or self.SessionLocal is None:
            return pd.DataFrame()
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
        session = self.SessionLocal()
        try:
            rows = (
                session.query(PredictionRecord)
                .filter(PredictionRecord.predicted_at >= cutoff)
                .filter(PredictionRecord.confidence >= min_confidence)
                .filter(PredictionRecord.actual_return.isnot(None))
                .all()
            )
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame([r.__dict__ for r in rows]).drop("_sa_instance_state", axis=1)
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    #  Strategies                                                          #
    # ------------------------------------------------------------------ #

    def save_strategy(self, record: dict) -> None:
        if not SQLALCHEMY_AVAILABLE or self.SessionLocal is None:
            return
        session = self.SessionLocal()
        try:
            strat = StrategyRecord(**record)
            session.add(strat)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.debug(f"save_strategy failed: {e}")
        finally:
            session.close()

    def load_active_strategies(self) -> list[dict]:
        if not SQLALCHEMY_AVAILABLE or self.SessionLocal is None:
            return []
        session = self.SessionLocal()
        try:
            rows = session.query(StrategyRecord).filter(StrategyRecord.is_active == 1).all()
            return [
                {k: v for k, v in r.__dict__.items() if not k.startswith("_")}
                for r in rows
            ]
        finally:
            session.close()


def init_db(settings: Optional[Settings] = None) -> Database:
    """Initialize database, creating tables if they don't exist."""
    db = Database(settings)
    if SQLALCHEMY_AVAILABLE and db.engine is not None:
        try:
            Base.metadata.create_all(db.engine)
            logger.info("Database initialized — tables created")
        except Exception as e:
            logger.warning(f"DB init failed (running in Parquet-only mode): {e}")
    else:
        logger.info("Database running in Parquet-only mode")
    return db
