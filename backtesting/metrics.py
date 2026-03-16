"""
Performance metrics for backtesting and strategy evaluation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestResult:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    profit_factor: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    volatility: float
    equity_curve: pd.Series
    trade_returns: list[float]
    passes_criteria: bool   # Meets min Sharpe, drawdown, profit factor thresholds


class PerformanceMetrics:
    """Computes comprehensive performance metrics for a backtest."""

    def __init__(
        self,
        min_sharpe: float = 1.5,
        max_drawdown: float = 0.20,
        min_profit_factor: float = 1.5,
        risk_free_rate: float = 0.05,
        trading_days: int = 252,
    ) -> None:
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.min_profit_factor = min_profit_factor
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def compute(
        self,
        equity_curve: pd.Series,
        trade_returns: list[float] | None = None,
    ) -> BacktestResult:
        """
        Compute full metrics from an equity curve.

        Args:
            equity_curve:   Indexed by date, values = portfolio equity
            trade_returns:  List of per-trade returns (optional)
        """
        returns = equity_curve.pct_change().dropna()
        trade_returns = trade_returns or list(returns[returns != 0])

        total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
        n_days = len(returns)
        n_years = max(n_days / self.trading_days, 1e-6)
        annualized_return = (1 + total_return) ** (1 / n_years) - 1

        volatility = float(returns.std() * np.sqrt(self.trading_days))

        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate / self.trading_days
        sharpe = float(excess_returns.mean() / (excess_returns.std() + 1e-10) * np.sqrt(self.trading_days))

        # Sortino ratio (only downside deviation)
        downside = returns[returns < 0]
        sortino_denom = float(downside.std() * np.sqrt(self.trading_days) + 1e-10)
        sortino = float((annualized_return - self.risk_free_rate) / sortino_denom)

        # Max drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_dd = float(drawdown.min())

        # Calmar ratio
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0.0

        # Trade-level metrics
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]

        win_rate = len(wins) / len(trade_returns) if trade_returns else 0.0
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_trade = float(np.mean(trade_returns)) if trade_returns else 0.0

        total_profit = sum(wins)
        total_loss = abs(sum(losses))
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        passes = (
            sharpe >= self.min_sharpe
            and abs(max_dd) <= self.max_drawdown
            and profit_factor >= self.min_profit_factor
            and win_rate >= 0.45
        )

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            profit_factor=profit_factor,
            win_rate=win_rate,
            total_trades=len(trade_returns),
            avg_trade_return=avg_trade,
            avg_win=avg_win,
            avg_loss=avg_loss,
            volatility=volatility,
            equity_curve=equity_curve,
            trade_returns=trade_returns,
            passes_criteria=passes,
        )

    @staticmethod
    def compute_drawdown_series(equity: pd.Series) -> pd.Series:
        rolling_max = equity.cummax()
        return (equity - rolling_max) / rolling_max

    @staticmethod
    def compute_underwater_days(equity: pd.Series) -> int:
        """Count number of days below previous peak."""
        rolling_max = equity.cummax()
        underwater = (equity < rolling_max)
        return int(underwater.sum())
