"""Efficient frontier utilities."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from etf_portfolio.backtesting.metrics import (
    calculate_portfolio_return,
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
)
from etf_portfolio.logging_config import get_logger, log_event
from etf_portfolio.optimization.optimizer import optimize_portfolio

LOGGER = get_logger(__name__)


def build_efficient_frontier(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    *,
    num_points: int = 25,
    long_only: bool = True,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    weight_sum: float = 1.0,
    risk_free_rate: float = 0.0,
    asset_classes: pd.Series | None = None,
    asset_class_bounds: Mapping[str, tuple[float, float]] | None = None,
    ticker_bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    bond_assets: Sequence[str] | None = None,
    min_bond_exposure: float | None = None,
    expense_ratios: pd.Series | None = None,
    max_expense_ratio: float | None = None,
) -> pd.DataFrame:
    """Build target-return efficient frontier points under the configured constraints."""

    if expected_returns.empty:
        raise ValueError("expected_returns must not be empty.")
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")

    optimization_kwargs = {
        "long_only": long_only,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "weight_sum": weight_sum,
        "risk_free_rate": risk_free_rate,
        "asset_classes": asset_classes,
        "asset_class_bounds": asset_class_bounds,
        "ticker_bounds": ticker_bounds,
        "bond_assets": bond_assets,
        "min_bond_exposure": min_bond_exposure,
        "expense_ratios": expense_ratios,
        "max_expense_ratio": max_expense_ratio,
    }
    min_vol_weights = optimize_portfolio(
        expected_returns,
        covariance_matrix,
        method="min_variance",
        **optimization_kwargs,
    )
    max_return_weights = optimize_portfolio(
        expected_returns,
        covariance_matrix,
        method="target_volatility",
        target_volatility=1e6,
        **optimization_kwargs,
    )
    lower_target = calculate_portfolio_return(min_vol_weights, expected_returns)
    upper_target = calculate_portfolio_return(max_return_weights, expected_returns)
    targets = np.linspace(float(lower_target), float(upper_target), num_points)
    rows: list[dict[str, float]] = []

    for target in targets:
        try:
            weights = optimize_portfolio(
                expected_returns,
                covariance_matrix,
                method="target_return",
                **optimization_kwargs,
                target_return=float(target),
            )
        except ValueError as exc:
            log_event(
                LOGGER,
                logging.WARNING,
                "efficient_frontier_point_failed",
                optimizer_status="failed",
                solver_used="SLSQP",
                target_return=float(target),
                reason=str(exc),
            )
            continue

        portfolio_return = calculate_portfolio_return(weights, expected_returns)
        portfolio_volatility = calculate_portfolio_volatility(weights, covariance_matrix)
        rows.append(
            {
                "target_return": float(target),
                "portfolio_return": portfolio_return,
                "portfolio_volatility": portfolio_volatility,
                "sharpe_ratio": calculate_sharpe_ratio(
                    portfolio_return,
                    portfolio_volatility,
                    risk_free_rate,
                ),
            }
        )
        log_event(
            LOGGER,
            logging.INFO,
            "efficient_frontier_point_completed",
            optimizer_status="success",
            solver_used="SLSQP",
            target_return=float(target),
            portfolio_return=portfolio_return,
            portfolio_volatility=portfolio_volatility,
            portfolio_sharpe=calculate_sharpe_ratio(
                portfolio_return,
                portfolio_volatility,
                risk_free_rate,
            ),
        )

    frontier = pd.DataFrame(rows)
    if frontier.empty:
        log_event(
            LOGGER,
            logging.ERROR,
            "efficient_frontier_failed",
            optimizer_status="failed",
            solver_used="SLSQP",
            reason="No feasible efficient frontier points were found for the supplied constraints.",
        )
        raise ValueError(
            "No feasible efficient frontier points were found for the supplied constraints."
        )
    log_event(
        LOGGER,
        logging.INFO,
        "efficient_frontier_completed",
        optimizer_status="success",
        solver_used="SLSQP",
        frontier_points=len(frontier),
        requested_points=num_points,
    )
    return frontier.sort_values("portfolio_volatility").reset_index(drop=True)


__all__ = ["build_efficient_frontier"]
