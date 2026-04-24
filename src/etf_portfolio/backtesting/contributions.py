"""Contribution-aware allocation routines.

Contribution-only rebalancing is the production mode for long-horizon, taxable
accumulation portfolios: instead of selling over-weight positions, every
rebalance directs new external cash to under-weight positions. This module
exposes one function, `allocate_contribution`, that converts a target weight
vector into a non-negative dollar trade vector under that constraint.

Properties guaranteed:
    1. All trades are >= 0 (no selling).
    2. Over-weight tickers receive zero new capital.
    3. Trades sum to the cash contribution exactly (within numerical tolerance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CASH_CONSERVATION_TOLERANCE = 1e-8


def allocate_contribution(
    previous_weights: pd.Series,
    target_weights: pd.Series,
    cash_contribution: float,
    portfolio_value: float,
) -> pd.Series:
    """Allocate `cash_contribution` to under-weight tickers via water-fill.

    Algorithm:
        1. Compute current dollar holdings from `previous_weights * portfolio_value`.
        2. Compute target dollar holdings post-contribution:
           `target_weights * (portfolio_value + cash_contribution)`.
        3. The "gap" per ticker is `max(target_holdings - current_holdings, 0)`.
        4. If the total gap fits within the contribution, fully fill each gap
           and distribute the remainder proportionally to `target_weights`.
        5. Otherwise, distribute the contribution proportionally to gap size.

    Returns:
        A pandas Series indexed by ticker (aligned to `target_weights`) with
        dollar amounts to BUY. All entries are >= 0.
    """

    if cash_contribution < 0.0:
        raise ValueError("cash_contribution must be non-negative.")
    if portfolio_value < 0.0:
        raise ValueError("portfolio_value must be non-negative.")

    target = target_weights.astype(float).copy()
    if target.empty:
        raise ValueError("target_weights must not be empty.")

    if not np.isclose(target.sum(), 1.0, atol=1e-6):
        raise ValueError(f"target_weights must sum to 1.0, got {target.sum():.6f}.")

    previous = previous_weights.reindex(target.index).fillna(0.0).astype(float)

    if cash_contribution == 0.0:
        return pd.Series(0.0, index=target.index, dtype=float)

    current_holdings = previous * portfolio_value
    new_total_value = portfolio_value + cash_contribution
    target_holdings = target * new_total_value
    gaps = (target_holdings - current_holdings).clip(lower=0.0)

    total_gap = float(gaps.sum())
    if total_gap <= 0.0:
        return pd.Series(
            (cash_contribution * target).to_numpy(),
            index=target.index,
            dtype=float,
        )

    if total_gap >= cash_contribution - CASH_CONSERVATION_TOLERANCE:
        trades = gaps * (cash_contribution / total_gap)
    else:
        leftover = cash_contribution - total_gap
        trades = gaps + (leftover * target)

    trades = trades.clip(lower=0.0)
    delta = cash_contribution - float(trades.sum())
    if abs(delta) > CASH_CONSERVATION_TOLERANCE:
        trades = trades + (delta * target)
        trades = trades.clip(lower=0.0)
        delta = cash_contribution - float(trades.sum())
        if abs(delta) > CASH_CONSERVATION_TOLERANCE:
            trades.iloc[int(target.values.argmax())] += delta

    return trades


__all__ = ["CASH_CONSERVATION_TOLERANCE", "allocate_contribution"]
