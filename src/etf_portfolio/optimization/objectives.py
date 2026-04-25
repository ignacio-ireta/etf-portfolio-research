"""Objective functions for the portfolio optimizer.

Each builder returns a callable `(weights: np.ndarray) -> float` that the
SLSQP solver can minimize. The orchestrator in `optimization.optimizer`
selects the right builder per `OptimizationMethod`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from etf_portfolio.optimization.optimizer import OptimizationMethod

ObjectiveFn = Callable[[np.ndarray], float]


def build_objective_function(
    *,
    method: OptimizationMethod,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> ObjectiveFn:
    """Return the SLSQP objective function for `method`."""

    mu = expected_returns.to_numpy(dtype=float)
    sigma = covariance_matrix.to_numpy(dtype=float)

    if method == "equal_weight":
        return _equal_weight_objective(asset_count=len(mu))

    if method == "inverse_volatility":
        return _inverse_volatility_objective(sigma=sigma)

    if method == "min_variance":
        return _variance_objective(sigma=sigma)

    if method == "max_sharpe":
        return _negative_sharpe_objective(
            mu=mu,
            sigma=sigma,
            risk_free_rate=risk_free_rate,
        )

    if method == "target_volatility":
        return _negative_return_objective(mu=mu)

    if method == "target_return":
        return _variance_objective(sigma=sigma)

    if method == "risk_parity":
        return _risk_parity_objective(sigma=sigma)

    raise ValueError(f"Unsupported optimization method: {method}.")


def _equal_weight_objective(*, asset_count: int) -> ObjectiveFn:
    target = np.full(asset_count, 1.0 / asset_count, dtype=float)
    return lambda weights: float(np.square(weights - target).sum())


def _inverse_volatility_objective(*, sigma: np.ndarray) -> ObjectiveFn:
    inverse_vol = 1.0 / np.sqrt(np.diag(sigma))
    target = inverse_vol / inverse_vol.sum()
    return lambda weights: float(np.square(weights - target).sum())


def _variance_objective(*, sigma: np.ndarray) -> ObjectiveFn:
    return lambda weights: float(weights.T.dot(sigma).dot(weights))


def _negative_return_objective(*, mu: np.ndarray) -> ObjectiveFn:
    return lambda weights: -float(weights.dot(mu))


def _negative_sharpe_objective(
    *,
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_free_rate: float,
) -> ObjectiveFn:
    def negative_sharpe(weights: np.ndarray) -> float:
        portfolio_variance = float(weights.T.dot(sigma).dot(weights))
        # Use a small epsilon to keep the objective differentiable and avoid division by zero.
        portfolio_vol = np.sqrt(max(portfolio_variance, 1e-12))
        portfolio_return = float(weights.dot(mu))
        return -float((portfolio_return - risk_free_rate) / portfolio_vol)

    return negative_sharpe


def _risk_parity_objective(*, sigma: np.ndarray) -> ObjectiveFn:
    def risk_parity_objective(weights: np.ndarray) -> float:
        portfolio_variance = max(float(weights.T.dot(sigma).dot(weights)), 0.0)
        portfolio_vol = float(np.sqrt(portfolio_variance))
        if np.isclose(portfolio_vol, 0.0):
            return 1e9
        marginal = sigma.dot(weights)
        contributions = weights * marginal / portfolio_vol
        target = np.full(len(weights), portfolio_vol / len(weights), dtype=float)
        return float(np.square(contributions - target).sum())

    return risk_parity_objective


__all__ = ["ObjectiveFn", "build_objective_function"]
