"""
Global configuration for the Autonomous Quant AI Trading System.
All parameters can be overridden via environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseConfig:
    url: str = os.getenv("DATABASE_URL", "postgresql://localhost:5432/trading")
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


@dataclass
class RedisConfig:
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    ttl_seconds: int = 3600          # 1h default cache TTL
    realtime_ttl: int = 60           # 1 min for real-time data


@dataclass
class DataConfig:
    # Storage paths
    data_lake_path: str = os.getenv("DATA_LAKE_PATH", "./data/lake")
    models_path: str = os.getenv("MODELS_PATH", "./models/checkpoints")
    predictions_path: str = os.getenv("PREDICTIONS_PATH", "./predictions/history")
    reports_path: str = os.getenv("REPORTS_PATH", "./predictions/reports")

    # Asset universe limits
    max_assets_per_day: int = 50_000
    crypto_symbols_limit: int = 500
    equity_symbols_limit: int = 5000

    # Timeframes to collect
    timeframes: list = field(default_factory=lambda: ["1m", "5m", "1h", "4h", "1d"])
    primary_timeframe: str = "1d"

    # Historical data window
    lookback_days: int = 730        # 2 years default
    min_bars_required: int = 252    # 1 year minimum for backtesting

    # API Keys
    news_api_key: str = os.getenv("NEWS_API_KEY", "")
    coingecko_api_key: str = os.getenv("COINGECKO_API_KEY", "")
    fred_api_key: str = os.getenv("FRED_API_KEY", "")
    glassnode_api_key: str = os.getenv("GLASSNODE_API_KEY", "")
    alpha_vantage_key: str = os.getenv("ALPHA_VANTAGE_KEY", "")

    # Exchange credentials
    exchange_id: str = os.getenv("CCXT_EXCHANGE_ID", "binance")
    exchange_api_key: str = os.getenv("CCXT_EXCHANGE_API_KEY", "")
    exchange_api_secret: str = os.getenv("CCXT_EXCHANGE_API_SECRET", "")


@dataclass
class FeatureConfig:
    # RSI periods
    rsi_periods: list = field(default_factory=lambda: [7, 14, 21])

    # Moving average periods
    ma_periods: list = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR period
    atr_period: int = 14

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Temporal embedding dimensions
    embedding_dim: int = 64
    encoder_hidden_dim: int = 128

    # Lookback window for sequence models
    sequence_length: int = 60      # 60 bars lookback


@dataclass
class ModelConfig:
    # LSTM
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 3
    lstm_dropout: float = 0.2

    # Transformer
    transformer_d_model: int = 256
    transformer_nhead: int = 8
    transformer_num_layers: int = 4
    transformer_dropout: float = 0.1
    transformer_max_seq_len: int = 512

    # CNN
    cnn_channels: list = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_size: int = 3

    # GNN
    gnn_hidden_channels: int = 128
    gnn_num_layers: int = 3

    # Ensemble
    ensemble_method: str = "stacking"   # stacking | averaging | voting

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 15              # Early stopping patience
    validation_split: float = 0.2
    test_split: float = 0.1

    # Device
    device: str = os.getenv("TORCH_DEVICE", "auto")   # auto | cpu | cuda | mps

    # Prediction horizons
    horizons: list = field(default_factory=lambda: ["1h", "4h", "1d", "1w"])


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_rate: float = 0.001      # 0.1% per trade
    slippage_rate: float = 0.0005       # 0.05% slippage

    # Strategy acceptance thresholds
    min_sharpe_ratio: float = 1.5
    max_drawdown_pct: float = 0.20      # 20%
    min_profit_factor: float = 1.5
    min_win_rate: float = 0.45

    # Walk-forward parameters
    walk_forward_folds: int = 5
    walk_forward_window_days: int = 365

    # Risk management
    max_position_size_pct: float = 0.05  # 5% max per position
    max_portfolio_risk_pct: float = 0.02  # 2% max portfolio risk per trade
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0


@dataclass
class SignalConfig:
    # Confidence thresholds
    min_confidence_to_trade: float = 0.60
    high_confidence_threshold: float = 0.80

    # Score thresholds
    whale_score_threshold: float = 65.0     # Above this = smart money active
    explosion_score_threshold: float = 70.0  # Above this = potential breakout

    # Output
    max_signals_per_run: int = 100
    output_formats: list = field(default_factory=lambda: ["json", "csv"])


@dataclass
class ContinuousLearningConfig:
    # Retraining triggers
    accuracy_degradation_threshold: float = 0.05   # 5% drop triggers retrain
    min_predictions_for_eval: int = 100
    eval_lookback_days: int = 30

    # Retraining schedule
    retrain_frequency_days: int = 7      # Weekly retraining
    full_retrain_frequency_days: int = 30 # Monthly full retrain

    # Data management
    max_prediction_history_days: int = 730


@dataclass
class StrategyDiscoveryConfig:
    # RL parameters
    rl_algorithm: str = "PPO"           # PPO | DQN | A2C
    rl_total_timesteps: int = 1_000_000
    rl_n_envs: int = 8

    # Genetic algorithm
    ga_population_size: int = 100
    ga_generations: int = 50
    ga_mutation_rate: float = 0.1
    ga_crossover_rate: float = 0.7

    # Strategy parameters to evolve
    param_bounds: dict = field(default_factory=lambda: {
        "rsi_period": (5, 50),
        "rsi_oversold": (20, 40),
        "rsi_overbought": (60, 80),
        "ma_fast": (5, 30),
        "ma_slow": (20, 200),
        "bb_period": (10, 30),
        "bb_std": (1.5, 3.0),
        "atr_multiplier": (1.0, 5.0),
    })


@dataclass
class Settings:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    continuous_learning: ContinuousLearningConfig = field(default_factory=ContinuousLearningConfig)
    strategy_discovery: StrategyDiscoveryConfig = field(default_factory=StrategyDiscoveryConfig)

    # Global
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    environment: str = os.getenv("ENVIRONMENT", "development")
    random_seed: int = 42
    dry_run: bool = os.getenv("DRY_RUN", "false").lower() == "true"
