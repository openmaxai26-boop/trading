"""
Walk-forward backtesting engine.
Tests trading strategies on historical data with proper out-of-sample validation.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtesting.metrics import PerformanceMetrics, BacktestResult
from backtesting.risk_manager import RiskManager
from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

StrategyFn = Callable[[pd.DataFrame, dict], pd.Series]  # ohlcv, params → signal series (1=buy, -1=sell, 0=hold)


@dataclass
class StrategyConfig:
    name: str
    params: dict
    asset_class: str = "crypto"
    timeframe: str = "1d"


class BacktestEngine:
    """
    Event-driven backtesting engine with walk-forward validation.

    Features:
    - Commission and slippage simulation
    - Walk-forward folds
    - Position sizing via RiskManager
    - Full metrics computation
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.cfg = self.settings.backtest
        self.metrics = PerformanceMetrics(
            min_sharpe=self.cfg.min_sharpe_ratio,
            max_drawdown=self.cfg.max_drawdown_pct,
            min_profit_factor=self.cfg.min_profit_factor,
        )
        self.risk = RiskManager(settings)
        self.results_dir = os.path.join(self.settings.data.predictions_path, "backtest_results")
        os.makedirs(self.results_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Single backtest run                                                 #
    # ------------------------------------------------------------------ #

    def run_single(
        self,
        ohlcv: pd.DataFrame,
        strategy_fn: StrategyFn,
        params: dict,
        min_bars: int = 50,
    ) -> BacktestResult:
        """
        Run a single backtest of a strategy on OHLCV data.

        Args:
            ohlcv:        OHLCV DataFrame
            strategy_fn:  Function(ohlcv, params) → pd.Series of signals
            params:       Strategy parameters dict
            min_bars:     Minimum bars required (50 for walk-forward folds)

        Returns:
            BacktestResult with all metrics.
        """
        if len(ohlcv) < min_bars:
            raise ValueError(f"Insufficient data: {len(ohlcv)} < {min_bars}")

        # Generate signals
        signals = strategy_fn(ohlcv, params)

        # Simulate trades
        equity_curve, trade_returns = self._simulate(ohlcv, signals)

        return self.metrics.compute(equity_curve, trade_returns)

    # ------------------------------------------------------------------ #
    #  Walk-forward validation                                             #
    # ------------------------------------------------------------------ #

    def walk_forward(
        self,
        ohlcv: pd.DataFrame,
        strategy_fn: StrategyFn,
        params: dict,
        n_folds: int | None = None,
    ) -> dict:
        """
        Walk-forward backtesting: train on in-sample, test on out-of-sample.
        Returns aggregated metrics across all folds.
        """
        n_folds = n_folds or self.cfg.walk_forward_folds
        n = len(ohlcv)
        fold_size = n // (n_folds + 1)

        fold_results: list[BacktestResult] = []
        for fold in range(n_folds):
            test_start = fold_size * (fold + 1)
            test_end = min(test_start + fold_size, n)
            test_data = ohlcv.iloc[test_start:test_end]

            if len(test_data) < 50:
                continue

            try:
                result = self.run_single(test_data, strategy_fn, params, min_bars=50)
                fold_results.append(result)
            except Exception as e:
                logger.debug(f"Walk-forward fold {fold+1} failed: {e}")

        if not fold_results:
            return {"error": "no_valid_folds"}

        # Aggregate
        sharpes = [r.sharpe_ratio for r in fold_results]
        drawdowns = [r.max_drawdown for r in fold_results]
        pf_list = [r.profit_factor for r in fold_results]

        agg = {
            "n_folds": len(fold_results),
            "mean_sharpe": float(np.mean(sharpes)),
            "std_sharpe": float(np.std(sharpes)),
            "min_sharpe": float(np.min(sharpes)),
            "mean_max_drawdown": float(np.mean(drawdowns)),
            "mean_profit_factor": float(np.mean(pf_list)),
            "passes_all_folds": all(r.passes_criteria for r in fold_results),
            "passes_fraction": sum(r.passes_criteria for r in fold_results) / len(fold_results),
        }
        return agg

    # ------------------------------------------------------------------ #
    #  Built-in strategy library                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def rsi_strategy(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
        """Simple RSI mean-reversion strategy."""
        period = params.get("rsi_period", 14)
        oversold = params.get("rsi_oversold", 30)
        overbought = params.get("rsi_overbought", 70)

        delta = ohlcv["close"].diff()
        gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        signals = pd.Series(0, index=ohlcv.index)
        signals[rsi < oversold] = 1    # Buy
        signals[rsi > overbought] = -1  # Sell
        return signals

    @staticmethod
    def ma_crossover_strategy(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
        """Moving average crossover strategy."""
        fast = params.get("ma_fast", 10)
        slow = params.get("ma_slow", 50)
        close = ohlcv["close"]
        ma_fast = close.ewm(span=fast, adjust=False).mean()
        ma_slow = close.ewm(span=slow, adjust=False).mean()

        signals = pd.Series(0, index=ohlcv.index)
        signals[ma_fast > ma_slow] = 1
        signals[ma_fast < ma_slow] = -1
        return signals

    @staticmethod
    def bollinger_breakout_strategy(ohlcv: pd.DataFrame, params: dict) -> pd.Series:
        """Bollinger Band breakout strategy."""
        period = params.get("bb_period", 20)
        std_mult = params.get("bb_std", 2.0)
        close = ohlcv["close"]
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = sma + std_mult * std
        lower = sma - std_mult * std

        signals = pd.Series(0, index=ohlcv.index)
        signals[close > upper] = 1    # Breakout long
        signals[close < lower] = -1   # Breakout short
        return signals

    # ------------------------------------------------------------------ #
    #  Simulation                                                          #
    # ------------------------------------------------------------------ #

    def _simulate(
        self,
        ohlcv: pd.DataFrame,
        signals: pd.Series,
    ) -> tuple[pd.Series, list[float]]:
        """
        Simulate trading based on signals.
        Returns (equity_curve, trade_returns).
        """
        capital = self.cfg.initial_capital
        position = 0.0       # Current position (units)
        entry_price = 0.0
        equity = []
        trade_returns = []
        position_value = 0.0

        for i, (ts, row) in enumerate(ohlcv.iterrows()):
            price = row["close"]
            signal = signals.get(ts, 0) if ts in signals.index else 0

            # Exit existing position
            if position != 0 and (
                (position > 0 and signal == -1) or
                (position < 0 and signal == 1) or
                signal == 0
            ):
                exit_price = price * (1 - self.cfg.slippage_rate if position > 0 else 1 + self.cfg.slippage_rate)
                commission = abs(position) * exit_price * self.cfg.commission_rate
                pnl = position * (exit_price - entry_price) - commission
                trade_return = pnl / (abs(position) * entry_price) if entry_price > 0 else 0.0
                trade_returns.append(float(trade_return))
                capital += pnl
                position = 0.0

            # Enter new position
            if signal != 0 and position == 0:
                position_size = capital * self.cfg.max_position_size_pct
                entry_price = price * (1 + self.cfg.slippage_rate if signal > 0 else 1 - self.cfg.slippage_rate)
                commission = position_size * self.cfg.commission_rate
                capital -= commission
                position = (position_size / entry_price) * signal

            # Mark to market
            position_value = position * price
            equity.append(capital + position_value)

        equity_series = pd.Series(equity, index=ohlcv.index)
        return equity_series, trade_returns

    # ------------------------------------------------------------------ #
    #  Batch validation                                                    #
    # ------------------------------------------------------------------ #

    def validate_strategy(
        self,
        ohlcv: pd.DataFrame,
        strategy_fn: StrategyFn,
        params: dict,
        strategy_name: str = "unknown",
    ) -> tuple[bool, dict]:
        """
        Validate a strategy with walk-forward. Returns (is_valid, summary).
        """
        wf_result = self.walk_forward(ohlcv, strategy_fn, params)
        if "error" in wf_result:
            return False, wf_result

        is_valid = (
            wf_result["mean_sharpe"] >= self.cfg.min_sharpe_ratio
            and abs(wf_result["mean_max_drawdown"]) <= self.cfg.max_drawdown_pct
            and wf_result["mean_profit_factor"] >= self.cfg.min_profit_factor
            and wf_result["passes_fraction"] >= 0.6   # Passes in at least 60% of folds
        )

        summary = {
            "strategy_name": strategy_name,
            "params": params,
            "is_valid": is_valid,
            **wf_result,
        }

        if is_valid:
            self._save_result(summary)
            logger.info(f"Strategy ACCEPTED: {strategy_name} | Sharpe={wf_result['mean_sharpe']:.2f}")
        else:
            logger.debug(f"Strategy REJECTED: {strategy_name} | Sharpe={wf_result['mean_sharpe']:.2f}")

        return is_valid, summary

    def _save_result(self, result: dict) -> None:
        name = result.get("strategy_name", "unknown").replace(" ", "_")
        path = os.path.join(self.results_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)
