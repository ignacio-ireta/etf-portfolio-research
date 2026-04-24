"""Transaction cost helpers for backtesting."""

from __future__ import annotations

import pandas as pd


def transaction_costs(
    trades_dollars: pd.Series,
    *,
    cost_rate: float,
) -> float:
    """Compute transaction costs from the absolute dollar value traded."""

    if cost_rate < 0:
        raise ValueError("cost_rate must be non-negative.")

    gross_traded_dollars = float(trades_dollars.abs().sum())
    return gross_traded_dollars * cost_rate
