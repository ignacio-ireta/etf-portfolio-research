"""Rebalancing helpers for walk-forward backtests.

Three rebalance modes are supported:

* `full_rebalance` — trade to the optimizer target every period; assumes
  selling is allowed.
* `tolerance_band` — only trade when realized drift exceeds the per-ticker
  band; trades bring weights back to the band edge, not the optimizer target,
  to minimize turnover.
* `contribution_only` — never sell; route external `contribution_amount` cash
  to under-weight tickers via water-fill. Optional fallback selling can be
  enabled when drift breaches a configured threshold.

The shared API is `apply_rebalance_mode` which returns a `RebalanceDecision`
describing the new applied weights, dollar trades, and the new portfolio
value after the rebalance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from etf_portfolio.backtesting.contributions import (
    CASH_CONSERVATION_TOLERANCE,
    allocate_contribution,
)
from etf_portfolio.config import RebalanceToleranceBandsConfig

RebalanceMode = Literal["full_rebalance", "tolerance_band", "contribution_only"]


@dataclass(frozen=True)
class RebalanceDecision:
    """Outcome of one rebalance step."""

    mode: RebalanceMode
    applied_weights: pd.Series
    trades_dollars: pd.Series
    portfolio_value_after: float
    rebalanced: bool


def normalize_rebalance_dates(
    returns_index: pd.Index,
    rebalance_dates: Sequence[pd.Timestamp] | None = None,
) -> pd.DatetimeIndex:
    """Return sorted rebalance dates that exist in the return index."""

    if not isinstance(returns_index, pd.DatetimeIndex):
        raise ValueError("Return index must be a DatetimeIndex.")

    if rebalance_dates is None:
        return pd.DatetimeIndex(returns_index)

    normalized = pd.DatetimeIndex(pd.to_datetime(list(rebalance_dates))).sort_values().unique()
    valid_dates = normalized.intersection(returns_index)
    if valid_dates.empty:
        raise ValueError("No rebalance dates overlap with the return index.")

    return pd.DatetimeIndex(valid_dates)


def apply_rebalance_mode(
    *,
    mode: RebalanceMode,
    previous_weights: pd.Series,
    target_weights: pd.Series,
    portfolio_value: float,
    contribution_amount: float = 0.0,
    tolerance_bands: RebalanceToleranceBandsConfig | None = None,
    asset_classes: pd.Series | None = None,
    fallback_sell_allowed: bool = False,
    fallback_drift_threshold: float | None = None,
    force_sell_rebalance: bool = False,
) -> RebalanceDecision:
    """Apply the configured rebalance mode and return the decision.

    `previous_weights` are the weights observed at this rebalance date AFTER
    market drift since the last rebalance (so they represent the actual
    pre-trade portfolio mix). `target_weights` are what the optimizer wants.
    """

    target = _validate_weights(target_weights, "target_weights")
    previous = previous_weights.reindex(target.index).fillna(0.0).astype(float)
    portfolio_value = float(portfolio_value)

    if portfolio_value < 0.0:
        raise ValueError("portfolio_value must be non-negative.")

    if mode == "full_rebalance":
        return _apply_full_rebalance(
            previous=previous,
            target=target,
            portfolio_value=portfolio_value,
            contribution_amount=contribution_amount,
        )

    if mode == "tolerance_band":
        if tolerance_bands is None:
            raise ValueError(
                "tolerance_bands must be provided when mode='tolerance_band'.",
            )
        return _apply_tolerance_band(
            previous=previous,
            target=target,
            portfolio_value=portfolio_value,
            contribution_amount=contribution_amount,
            tolerance_bands=tolerance_bands,
            asset_classes=asset_classes,
        )

    if mode == "contribution_only":
        return _apply_contribution_only(
            previous=previous,
            target=target,
            portfolio_value=portfolio_value,
            contribution_amount=contribution_amount,
            fallback_sell_allowed=fallback_sell_allowed,
            fallback_drift_threshold=fallback_drift_threshold,
            force_sell_rebalance=force_sell_rebalance,
        )

    raise ValueError(f"Unsupported rebalance mode: {mode!r}.")


def _apply_full_rebalance(
    *,
    previous: pd.Series,
    target: pd.Series,
    portfolio_value: float,
    contribution_amount: float,
) -> RebalanceDecision:
    new_total = portfolio_value + contribution_amount
    new_holdings = target * new_total
    previous_holdings = previous * portfolio_value
    trades = new_holdings - previous_holdings
    rebalanced = bool(np.any(np.abs(trades.to_numpy()) > 1e-9))
    return RebalanceDecision(
        mode="full_rebalance",
        applied_weights=target.copy(),
        trades_dollars=trades,
        portfolio_value_after=new_total,
        rebalanced=rebalanced,
    )


def _apply_tolerance_band(
    *,
    previous: pd.Series,
    target: pd.Series,
    portfolio_value: float,
    contribution_amount: float,
    tolerance_bands: RebalanceToleranceBandsConfig,
    asset_classes: pd.Series | None,
) -> RebalanceDecision:
    drift_per_ticker = previous - target
    ticker_band = float(tolerance_bands.per_ticker_abs_drift)
    ticker_breach = drift_per_ticker.abs() > ticker_band

    asset_class_breach = False
    if asset_classes is not None:
        ac_band = float(tolerance_bands.per_asset_class_abs_drift)
        aligned_classes = asset_classes.reindex(target.index)
        if not aligned_classes.isna().all():
            current_class_weights = previous.groupby(aligned_classes).sum()
            target_class_weights = target.groupby(aligned_classes).sum()
            class_drift = (current_class_weights - target_class_weights).abs()
            asset_class_breach = bool((class_drift > ac_band).any())

    if not ticker_breach.any() and not asset_class_breach:
        new_total = portfolio_value + contribution_amount
        if contribution_amount > 0.0:
            return _apply_contribution_only(
                previous=previous,
                target=target,
                portfolio_value=portfolio_value,
                contribution_amount=contribution_amount,
                fallback_sell_allowed=False,
                fallback_drift_threshold=None,
            )
        return RebalanceDecision(
            mode="tolerance_band",
            applied_weights=previous.copy(),
            trades_dollars=pd.Series(0.0, index=target.index, dtype=float),
            portfolio_value_after=new_total,
            rebalanced=False,
        )

    band_high = (target + ticker_band).clip(upper=1.0)
    band_low = (target - ticker_band).clip(lower=0.0)
    new_weights = previous.clip(lower=band_low, upper=band_high)
    weight_sum = float(new_weights.sum())
    if weight_sum <= 0.0:
        new_weights = target.copy()
    else:
        new_weights = new_weights / weight_sum

    new_total = portfolio_value + contribution_amount
    previous_holdings = previous * portfolio_value
    new_holdings = new_weights * new_total
    trades = new_holdings - previous_holdings
    return RebalanceDecision(
        mode="tolerance_band",
        applied_weights=new_weights,
        trades_dollars=trades,
        portfolio_value_after=new_total,
        rebalanced=True,
    )


def _apply_contribution_only(
    *,
    previous: pd.Series,
    target: pd.Series,
    portfolio_value: float,
    contribution_amount: float,
    fallback_sell_allowed: bool,
    fallback_drift_threshold: float | None,
    force_sell_rebalance: bool,
) -> RebalanceDecision:
    new_total = portfolio_value + contribution_amount

    drift = (previous - target).abs()
    threshold_triggered = (
        fallback_drift_threshold is not None
        and bool((drift > fallback_drift_threshold).any())
        and fallback_sell_allowed
    )
    fallback_triggered = force_sell_rebalance or threshold_triggered

    if fallback_triggered:
        previous_holdings = previous * portfolio_value
        new_holdings = target * new_total
        trades = new_holdings - previous_holdings
        return RebalanceDecision(
            mode="contribution_only",
            applied_weights=target.copy(),
            trades_dollars=trades,
            portfolio_value_after=new_total,
            rebalanced=True,
        )

    if portfolio_value <= 0.0:
        applied_weights = target.copy()
        if contribution_amount > 0.0:
            trades = contribution_amount * target
        else:
            trades = pd.Series(0.0, index=target.index, dtype=float)
        return RebalanceDecision(
            mode="contribution_only",
            applied_weights=applied_weights,
            trades_dollars=trades,
            portfolio_value_after=new_total,
            rebalanced=contribution_amount > 0.0,
        )

    trades = allocate_contribution(
        previous_weights=previous,
        target_weights=target,
        cash_contribution=contribution_amount,
        portfolio_value=portfolio_value,
    )

    previous_holdings = previous * portfolio_value
    new_holdings = previous_holdings + trades
    if new_total > 0.0:
        applied_weights = new_holdings / new_total
    else:
        applied_weights = target.copy()

    cash_drift = float(trades.sum() - contribution_amount)
    if abs(cash_drift) > CASH_CONSERVATION_TOLERANCE * 100:
        raise RuntimeError(
            "Contribution allocation violated cash conservation: "
            f"sum(trades)={float(trades.sum()):.8f}, "
            f"contribution={contribution_amount:.8f}.",
        )

    return RebalanceDecision(
        mode="contribution_only",
        applied_weights=applied_weights,
        trades_dollars=trades,
        portfolio_value_after=new_total,
        rebalanced=contribution_amount > 0.0,
    )


def _validate_weights(weights: pd.Series, name: str) -> pd.Series:
    if weights.empty:
        raise ValueError(f"{name} must not be empty.")
    series = weights.astype(float).copy()
    if not np.isclose(series.sum(), 1.0, atol=1e-6):
        raise ValueError(f"{name} must sum to 1.0, got {float(series.sum()):.6f}.")
    return series


__all__ = [
    "RebalanceDecision",
    "RebalanceMode",
    "apply_rebalance_mode",
    "normalize_rebalance_dates",
]
