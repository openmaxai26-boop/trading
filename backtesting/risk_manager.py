"""
Risk management — position sizing, stop-loss, portfolio constraints.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PositionSizing:
    symbol: str
    position_size_pct: float        # Fraction of portfolio to allocate
    stop_loss_price: float
    take_profit_price: float
    risk_amount: float              # Dollar risk (stop × size)
    expected_return: float


class RiskManager:
    """
    Implements position sizing and risk controls.

    Methods:
    - Kelly criterion (fractional)
    - ATR-based stop-loss
    - Portfolio-level constraints
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.cfg = self.settings.backtest

    def compute_position_size(
        self,
        symbol: str,
        current_price: float,
        prob_up: float,
        prob_down: float,
        atr: float,
        portfolio_value: float,
        current_positions: dict[str, float],
    ) -> PositionSizing:
        """
        Compute optimal position size using fractional Kelly criterion
        combined with ATR-based stop-loss.

        Args:
            symbol:               Asset symbol
            current_price:        Current market price
            prob_up:              P(price goes up)
            prob_down:            P(price goes down)
            atr:                  Average True Range
            portfolio_value:      Current portfolio value
            current_positions:    Existing positions {symbol: pct_allocation}
        """
        # Kelly fraction: f = p - q/b
        # where p = win prob, q = loss prob, b = win/loss ratio (approx ATR-based)
        stop_distance = atr * self.cfg.stop_loss_atr_multiplier
        tp_distance = atr * self.cfg.take_profit_atr_multiplier
        b = tp_distance / stop_distance if stop_distance > 0 else 1.0

        if prob_up > prob_down:
            kelly_fraction = prob_up - prob_down / b
        else:
            kelly_fraction = 0.0

        # Fractional Kelly (use 25% of Kelly to reduce risk)
        kelly_fraction = max(0.0, kelly_fraction * 0.25)

        # Cap at max position size
        position_size_pct = min(kelly_fraction, self.cfg.max_position_size_pct)

        # Portfolio-level risk check
        total_allocated = sum(current_positions.values())
        if total_allocated + position_size_pct > 0.8:  # Max 80% deployed
            available = max(0, 0.8 - total_allocated)
            position_size_pct = min(position_size_pct, available)

        # Concentration limit: no single position > 5%
        position_size_pct = min(position_size_pct, self.cfg.max_position_size_pct)

        # Stop-loss price
        if prob_up > prob_down:
            stop_loss_price = current_price - stop_distance
            take_profit_price = current_price + tp_distance
        else:
            stop_loss_price = current_price + stop_distance
            take_profit_price = current_price - tp_distance

        risk_amount = portfolio_value * position_size_pct * (stop_distance / current_price)
        expected_return = portfolio_value * position_size_pct * (
            prob_up * (tp_distance / current_price) - prob_down * (stop_distance / current_price)
        )

        return PositionSizing(
            symbol=symbol,
            position_size_pct=round(position_size_pct, 4),
            stop_loss_price=round(stop_loss_price, 8),
            take_profit_price=round(take_profit_price, 8),
            risk_amount=round(risk_amount, 2),
            expected_return=round(expected_return, 2),
        )

    def apply_portfolio_constraints(
        self,
        signals: list[dict],
        portfolio_value: float,
        max_positions: int = 20,
        max_sector_pct: float = 0.30,
    ) -> list[dict]:
        """
        Filter and rank signals to respect portfolio-level constraints.

        Args:
            signals:        List of signal dicts with 'confidence', 'asset', 'asset_class'
            portfolio_value: Current portfolio value
            max_positions:   Max number of open positions
            max_sector_pct:  Max allocation per asset class

        Returns:
            Filtered and sized list of signals.
        """
        if not signals:
            return []

        # Sort by confidence (highest first)
        signals = sorted(signals, key=lambda s: s.get("confidence", 0), reverse=True)

        # Apply top N cap
        signals = signals[:max_positions]

        # Asset class concentration limit
        class_allocations: dict[str, float] = {}
        final_signals = []
        for sig in signals:
            asset_class = sig.get("asset_class", "unknown")
            current_class_alloc = class_allocations.get(asset_class, 0.0)
            sig_size = sig.get("position_size_pct", 0.01)

            if current_class_alloc + sig_size <= max_sector_pct:
                class_allocations[asset_class] = current_class_alloc + sig_size
                final_signals.append(sig)

        logger.debug(
            f"Portfolio constraints: {len(signals)} → {len(final_signals)} signals, "
            f"class allocations: {class_allocations}"
        )
        return final_signals

    def compute_portfolio_var(
        self,
        returns_matrix: pd.DataFrame,
        weights: np.ndarray,
        confidence: float = 0.95,
        horizon: int = 1,
    ) -> float:
        """
        Compute portfolio Value-at-Risk (parametric).

        Args:
            returns_matrix:  DataFrame of asset returns (n_days × n_assets)
            weights:         Portfolio weights array
            confidence:      VaR confidence level (e.g., 0.95)
            horizon:         VaR horizon in days

        Returns:
            VaR as a positive fraction (e.g., 0.05 = 5% loss)
        """
        cov_matrix = returns_matrix.cov().values
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_std = np.sqrt(portfolio_variance * horizon)

        from scipy import stats
        z = stats.norm.ppf(1 - confidence)
        portfolio_mean = returns_matrix.mean().values @ weights * horizon
        var = -(portfolio_mean + z * portfolio_std)
        return float(max(var, 0))

    def dynamic_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        atr: float,
        initial_stop: float,
        atr_multiplier: float = 2.0,
    ) -> float:
        """
        Trailing stop-loss: tightens as price moves in favor.
        Returns updated stop-loss price.
        """
        # ATR-based trailing stop
        new_stop = current_price - atr_multiplier * atr
        # Only move stop upward (for long positions)
        return max(new_stop, initial_stop)
