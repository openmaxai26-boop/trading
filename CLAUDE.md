# CLAUDE.md — Autonomous Quant AI Trading System

## Project Overview

This repository implements a **fully autonomous AI-powered quantitative trading system** capable of:
- Analyzing up to 50,000 assets per day (equities, crypto, commodities, forex)
- Generating probabilistic predictions with explainability
- Auto-discovering and backtesting trading strategies
- Detecting whale/smart money activity
- Detecting market cycles and price explosion probability
- Continuous self-improvement via online learning

**Core principle**: Zero hallucination — all conclusions based on real, verifiable data.

---

## Repository Structure

```
trading/
├── CLAUDE.md                  # This file
├── requirements.txt           # Python dependencies
├── main.py                    # Main entry point (full pipeline run)
├── scheduler.py               # Automated daily scheduler
│
├── config/
│   ├── __init__.py
│   └── settings.py            # All configuration constants and env vars
│
├── data/
│   ├── __init__.py
│   ├── pipeline.py            # Orchestrates full data ingestion
│   └── collectors/
│       ├── __init__.py
│       ├── market_data.py     # OHLCV, tick data via ccxt/yfinance
│       ├── fundamental_data.py # Financial ratios, macro indices
│       ├── sentiment_data.py  # News, social sentiment
│       └── onchain_data.py    # On-chain crypto metrics
│   └── storage/
│       ├── __init__.py
│       ├── database.py        # PostgreSQL + Parquet data lake
│       └── cache.py           # Redis cache layer
│
├── features/
│   ├── __init__.py
│   ├── technical_indicators.py # RSI, MACD, Bollinger, ATR, etc.
│   ├── statistical_features.py # Autocorrelation, skewness, kurtosis
│   ├── temporal_embeddings.py  # Autoencoder temporal embeddings
│   └── feature_pipeline.py    # Orchestrates feature creation
│
├── models/
│   ├── __init__.py
│   ├── lstm_model.py          # LSTM / GRU time-series models
│   ├── transformer_model.py   # Temporal Transformer
│   ├── cnn_model.py           # 1D-CNN for pattern recognition
│   ├── gnn_model.py           # Graph Neural Network (asset relations)
│   └── ensemble.py            # Stacking / model averaging ensemble
│
├── training/
│   ├── __init__.py
│   ├── trainer.py             # Training loop, validation, checkpointing
│   └── continuous_learning.py # Error analysis + auto-retraining
│
├── backtesting/
│   ├── __init__.py
│   ├── engine.py              # Walk-forward backtesting engine
│   ├── metrics.py             # Sharpe, Sortino, drawdown, profit factor
│   └── risk_manager.py        # Position sizing, stop-loss, diversification
│
├── predictions/
│   ├── __init__.py
│   ├── signal_generator.py    # Produces trading signals per asset
│   └── output_formatter.py    # Formats final structured output
│
└── utils/
    ├── __init__.py
    ├── whale_detector.py      # Smart money / whale activity detection
    ├── cycle_detector.py      # Market cycle + price explosion detection
    ├── strategy_discovery.py  # RL + genetic algorithm strategy search
    └── logger.py              # Structured logging
```

---

## Architecture Overview

```
Raw Data Sources
      │
      ▼
Data Pipeline (data/pipeline.py)
  ├── MarketDataCollector   → OHLCV, order book
  ├── FundamentalCollector  → Financials, macro
  ├── SentimentCollector    → News, social
  └── OnChainCollector      → Crypto on-chain
      │
      ▼
Feature Engineering (features/feature_pipeline.py)
  ├── Technical Indicators
  ├── Statistical Features
  └── Temporal Embeddings
      │
      ▼
Model Ensemble (models/ensemble.py)
  ├── LSTM/GRU
  ├── Transformer
  ├── CNN
  └── GNN
      │         ┌──────────────────────┐
      ▼         │  Parallel Modules    │
Signal Generator├─► Whale Detector     │
(predictions/)  ├─► Cycle Detector     │
                └─► Strategy Discovery │
                          │
                          ▼
               Backtesting + Risk Mgmt
                          │
                          ▼
              Structured Output (per asset)
              {asset, prediction, confidence,
               whale_score, explosion_score,
               strategy, position_size}
                          │
                          ▼
              Continuous Learning Loop
```

---

## Key Conventions

### Python Style
- Python 3.10+
- Type hints on all public functions
- Docstrings on all classes and public methods
- Use `dataclasses` for structured data (signals, configs)
- Use `logging` (via `utils/logger.py`) — never `print()`

### Data Conventions
- All timestamps in UTC, stored as `datetime` or `pd.Timestamp`
- OHLCV columns: `open, high, low, close, volume` (lowercase)
- Asset symbols follow exchange conventions (e.g., `BTC/USDT` for crypto)
- Parquet files partitioned by `year/month/symbol`

### Model Conventions
- Models extend `BaseModel` abstract class
- All models expose `.predict(features: pd.DataFrame) -> np.ndarray`
- Models saved as `.pt` (PyTorch) or `.pkl` (sklearn) in `models/checkpoints/`
- Feature importance available via `.get_feature_importance()`

### Signal Conventions
Signals are `TradingSignal` dataclass instances:
```python
@dataclass
class TradingSignal:
    asset: str
    asset_class: str           # equity | crypto | commodity | forex
    horizon: str               # 1h | 4h | 1d | 1w
    prob_up: float             # 0.0 - 1.0
    prob_down: float           # 0.0 - 1.0
    confidence: float          # 0.0 - 1.0
    whale_score: float         # 0.0 - 100.0
    explosion_score: float     # 0.0 - 100.0
    market_regime: str         # bull | bear | range | high_vol | low_vol
    detected_cycles: list[str]
    recommended_strategy: str
    position_size_pct: float   # % of portfolio
    stop_loss_pct: float
    take_profit_pct: float
    timestamp: datetime
    explanation: dict          # Human-readable factor breakdown
```

### Configuration
All configuration lives in `config/settings.py`. Environment variables override defaults. Required env vars:
- `DATABASE_URL` — PostgreSQL connection string
- `REDIS_URL` — Redis connection string
- `CCXT_EXCHANGE_API_KEY` / `CCXT_EXCHANGE_API_SECRET` — exchange credentials
- `NEWS_API_KEY` — for sentiment data
- `COINGECKO_API_KEY` — for crypto on-chain metrics

### Backtesting Requirements
A strategy is only accepted if:
- Sharpe ratio > 1.5
- Max drawdown < 20%
- Profit factor > 1.5
- Minimum 252 trading days of backtest data
- Walk-forward validation passes

---

## Development Workflow

### Setup
```bash
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Initialize database
python -c "from data.storage.database import init_db; init_db()"
```

### Running the Full Pipeline
```bash
# Single full pipeline run
python main.py

# Run with specific asset universe
python main.py --assets crypto --limit 500

# Scheduler (runs daily at market open)
python scheduler.py
```

### Running Individual Modules
```bash
# Data collection only
python -c "from data.pipeline import DataPipeline; DataPipeline().run()"

# Feature engineering on cached data
python -c "from features.feature_pipeline import FeaturePipeline; FeaturePipeline().run('BTC/USDT')"

# Train all models
python -c "from training.trainer import ModelTrainer; ModelTrainer().train_all()"

# Generate signals
python -c "from predictions.signal_generator import SignalGenerator; SignalGenerator().run()"

# Run backtesting
python -c "from backtesting.engine import BacktestEngine; BacktestEngine().run_all()"
```

### Continuous Learning
```bash
# Analyze prediction errors and retrain if needed
python -c "from training.continuous_learning import ContinuousLearner; ContinuousLearner().analyze_and_retrain()"
```

---

## Data Flow

1. **Ingest**: `DataPipeline.run()` collects OHLCV + fundamentals + sentiment + on-chain for all assets
2. **Store**: Raw data → Parquet (historical) + Redis (real-time cache)
3. **Features**: `FeaturePipeline.transform()` generates 100+ features per asset
4. **Predict**: Ensemble model produces `(prob_up, prob_down, confidence)`
5. **Score**: WhaleDetector + CycleDetector produce scores 0-100
6. **Signal**: SignalGenerator merges predictions + scores → TradingSignal
7. **Backtest**: BacktestEngine validates strategies against historical data
8. **Output**: OutputFormatter produces JSON/CSV reports
9. **Learn**: ContinuousLearner compares predictions vs actuals, retrains underperformers

---

## Module Responsibilities

### `data/collectors/market_data.py`
- Connects to exchanges via `ccxt` and `yfinance`
- Fetches OHLCV at multiple timeframes (1m, 5m, 1h, 4h, 1d)
- Handles rate limiting, retries, and data gaps

### `data/collectors/fundamental_data.py`
- Fetches financial ratios (P/E, P/B, debt ratios) via financial APIs
- Macro indicators (VIX, DXY, yield curves) via FRED API

### `data/collectors/sentiment_data.py`
- News sentiment via NewsAPI + FinBERT NLP scoring
- Social sentiment via aggregated fear/greed indices

### `data/collectors/onchain_data.py`
- On-chain metrics via CoinGecko / Glassnode
- Exchange inflows/outflows, whale transactions, active addresses

### `features/technical_indicators.py`
- Uses `ta` library + custom implementations
- 50+ indicators: RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, etc.

### `models/ensemble.py`
- Combines LSTM + Transformer + CNN + GNN predictions
- Weighted averaging based on recent validation performance
- Confidence calibration via Platt scaling

### `utils/whale_detector.py`
- Detects abnormal volume, order book imbalances
- On-chain whale wallet monitoring
- Output: `whale_score` 0-100

### `utils/cycle_detector.py`
- Fourier + Wavelet analysis for cycle detection
- Hidden Markov Models for regime classification
- Volatility compression/expansion detection
- Output: `explosion_score` 0-100, `market_regime`

### `utils/strategy_discovery.py`
- PPO/DQN reinforcement learning for strategy optimization
- Genetic algorithm for strategy parameter search
- Auto-backtest and selection pipeline

### `training/continuous_learning.py`
- Stores all predictions with timestamps
- Compares predictions vs realized outcomes
- Identifies systematic biases and underperforming assets
- Triggers automatic retraining when accuracy drops below threshold

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Assets analyzed/day | 50,000+ |
| Signal generation latency | < 5 min per batch |
| Model prediction accuracy | > 60% directional |
| Strategy Sharpe ratio | > 1.5 |
| Max portfolio drawdown | < 20% |
| Retraining frequency | Weekly (or on degradation) |

---

## Testing

```bash
# Unit tests
pytest tests/ -v

# Integration test (small universe)
python main.py --assets crypto --limit 10 --dry-run

# Backtest validation
python -c "from backtesting.engine import BacktestEngine; BacktestEngine().validate()"
```

---

## Important Notes for AI Assistants

1. **Never fabricate data** — all data must come from real API calls or stored files
2. **Always validate signals** — no signal should be emitted without backtesting evidence
3. **Handle API failures gracefully** — use cached data when live data is unavailable
4. **Log everything** — use `utils/logger.py`, never bare `print()`
5. **Respect rate limits** — all collectors implement exponential backoff
6. **Configuration over hardcoding** — all parameters in `config/settings.py`
7. **Type safety** — all functions must have type hints; use `mypy` to validate
8. **Data quality** — validate and log data gaps, outliers, and anomalies before feature engineering
9. **Model versioning** — every trained model is saved with metadata (timestamp, features, metrics)
10. **Reproducibility** — set random seeds; store model configs alongside checkpoints
