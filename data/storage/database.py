"""
Database layer — PostgreSQL via SQLAlchemy + Parquet data lake.
"""

import os
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


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
    actual_return = Column(Float, nullable=True)   # Filled in after the fact
    correct_direction = Column(Integer, nullable=True)  # 1=correct, 0=wrong


class StrategyRecord(Base):
    """Stores discovered and validated strategies."""
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
    """Tracks model versions and their performance."""
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


class Database:
    """Manages PostgreSQL connections and Parquet data lake."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.engine = create_engine(
            self.settings.database.url,
            pool_size=self.settings.database.pool_size,
            max_overflow=self.settings.database.max_overflow,
            echo=self.settings.database.echo,
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.lake_path = self.settings.data.data_lake_path
        os.makedirs(self.lake_path, exist_ok=True)

    def get_session(self) -> Session:
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
        """Insert a prediction record into the database."""
        with self.get_session() as session:
            pred = PredictionRecord(**record)
            session.add(pred)
            session.commit()

    def update_prediction_outcome(
        self, prediction_id: int, actual_return: float, correct_direction: int
    ) -> None:
        """Update a prediction with its realized outcome."""
        with self.get_session() as session:
            pred = session.get(PredictionRecord, prediction_id)
            if pred:
                pred.actual_return = actual_return
                pred.correct_direction = correct_direction
                session.commit()

    def load_predictions_for_evaluation(
        self, days_back: int = 30, min_confidence: float = 0.0
    ) -> pd.DataFrame:
        """Load recent predictions for accuracy evaluation."""
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
        with self.get_session() as session:
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

    # ------------------------------------------------------------------ #
    #  Strategies                                                          #
    # ------------------------------------------------------------------ #

    def save_strategy(self, record: dict) -> None:
        with self.get_session() as session:
            strat = StrategyRecord(**record)
            session.add(strat)
            session.commit()

    def load_active_strategies(self) -> list[dict]:
        with self.get_session() as session:
            rows = session.query(StrategyRecord).filter(StrategyRecord.is_active == 1).all()
        return [
            {k: v for k, v in r.__dict__.items() if not k.startswith("_")}
            for r in rows
        ]


def init_db(settings: Optional[Settings] = None) -> Database:
    """Initialize database, creating tables if they don't exist."""
    db = Database(settings)
    Base.metadata.create_all(db.engine)
    logger.info("Database initialized — tables created")
    return db
