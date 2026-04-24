"""Return and risk attribution for portfolios with time-varying weights.

Two attribution flavors are supported:

* `return_attribution`: per-period contribution to the realized portfolio
  return, computed as `weight_t * asset_return_t`. Sums (across assets) equal
  the realized portfolio return for that period to within float precision.
* `risk_attribution`: classical Euler decomposition of portfolio volatility,
  `RC_i = w_i * (Sigma w)_i / sigma_p`, which sums to `sigma_p`.

Both functions are pure (no IO, no plotting, no logging) so they can be
re-used by reports, tests, and notebooks alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def return_attribution(
    weights_history: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Per-period contribution to portfolio return for each asset.

    The portfolio is assumed to hold ``weights_history.loc[rebalance_date]``
    over the interval ``[rebalance_date, next_rebalance_date)``. Within each
    interval, the contribution of asset ``i`` on day ``t`` is
    ``w_i(rebalance_date) * r_i(t)`` (no intra-period drift adjustment, in
    line with the rest of the backtest engine).

    Sum across columns gives the gross (pre-cost) portfolio return per period:
    `attribution.sum(axis=1) == (weights_active * asset_returns).sum(axis=1)`.
    """

    if weights_history.empty:
        raise ValueError("weights_history must not be empty.")
    if asset_returns.empty:
        raise ValueError("asset_returns must not be empty.")
    if not isinstance(weights_history.index, pd.DatetimeIndex):
        raise ValueError("weights_history index must be a DatetimeIndex.")
    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        raise ValueError("asset_returns index must be a DatetimeIndex.")

    columns = weights_history.columns.intersection(asset_returns.columns)
    if columns.empty:
        raise ValueError("weights_history and asset_returns share no overlapping asset columns.")

    aligned_returns = asset_returns.loc[:, columns].astype(float)
    rebalance_dates = weights_history.index
    first_rebalance = rebalance_dates.min()
    realized = aligned_returns.loc[aligned_returns.index >= first_rebalance].copy()

    weights_active = (
        weights_history.loc[:, columns].astype(float).reindex(realized.index, method="ffill")
    )
    attribution = weights_active.mul(realized, axis=0)
    attribution.index.name = realized.index.name or "date"
    return attribution


def risk_attribution(
    weights: pd.Series,
    covariance_matrix: pd.DataFrame,
) -> pd.Series:
    """Euler-decomposed risk contributions per asset.

    Returns a series indexed by asset whose entries sum to the portfolio's
    standard deviation `sqrt(w' Sigma w)`. Defined as
    ``RC_i = w_i * (Sigma w)_i / sigma_p``.

    For zero-volatility portfolios (e.g., all weights in cash) returns a
    series of zeros.
    """

    if weights.empty:
        raise ValueError("weights must not be empty.")
    if covariance_matrix.empty:
        raise ValueError("covariance_matrix must not be empty.")
    if not covariance_matrix.index.equals(covariance_matrix.columns):
        raise ValueError("covariance_matrix index and columns must match.")

    aligned_weights = weights.reindex(covariance_matrix.index).astype(float)
    if aligned_weights.isna().any():
        missing = aligned_weights.index[aligned_weights.isna()].tolist()
        raise ValueError(f"weights are missing entries for assets: {', '.join(missing)}.")

    w = aligned_weights.to_numpy(dtype=float)
    sigma = covariance_matrix.to_numpy(dtype=float)
    portfolio_variance = max(float(w @ sigma @ w), 0.0)
    portfolio_vol = float(np.sqrt(portfolio_variance))

    if portfolio_vol == 0.0:
        return pd.Series(0.0, index=aligned_weights.index, name="risk_contribution")

    marginal = sigma @ w
    contributions = w * marginal / portfolio_vol
    return pd.Series(
        contributions,
        index=aligned_weights.index,
        name="risk_contribution",
    )


def asset_class_return_attribution(
    attribution: pd.DataFrame,
    asset_classes: pd.Series,
    *,
    missing_label: str = "Unclassified",
) -> pd.DataFrame:
    """Aggregate per-asset return contributions into per-class contributions."""

    if attribution.empty:
        raise ValueError("attribution must not be empty.")

    aligned_classes = asset_classes.reindex(attribution.columns).fillna(missing_label).astype(str)
    grouped = attribution.T.groupby(aligned_classes).sum().T
    grouped.columns.name = "asset_class"
    return grouped


def asset_class_risk_attribution(
    risk_contributions: pd.Series,
    asset_classes: pd.Series,
    *,
    missing_label: str = "Unclassified",
) -> pd.Series:
    """Aggregate per-asset risk contributions into per-class contributions."""

    if risk_contributions.empty:
        raise ValueError("risk_contributions must not be empty.")

    aligned_classes = (
        asset_classes.reindex(risk_contributions.index).fillna(missing_label).astype(str)
    )
    grouped = risk_contributions.groupby(aligned_classes).sum()
    grouped.name = "risk_contribution"
    grouped.index.name = "asset_class"
    return grouped


__all__ = [
    "asset_class_return_attribution",
    "asset_class_risk_attribution",
    "return_attribution",
    "risk_attribution",
]
