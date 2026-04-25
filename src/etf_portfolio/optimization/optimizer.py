"""Constraint-aware portfolio optimizer.

This module is the orchestrator: it validates inputs, composes constraint
builders from `optimization.constraints`, picks the right objective from
`optimization.objectives`, and delegates the actual numerical optimization
to scipy's SLSQP solver. Any new constraint or objective should be added in
those files, not here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from etf_portfolio.backtesting.metrics import (
    calculate_portfolio_return,
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
)
from etf_portfolio.logging_config import get_logger, log_event
from etf_portfolio.optimization.constraints import (
    build_bounds,
    build_linear_constraints,
    build_nonlinear_constraints,
    check_linear_feasibility,
    validate_linear_feasibility,
)
from etf_portfolio.optimization.objectives import build_objective_function

OptimizationMethod = Literal[
    "equal_weight",
    "inverse_volatility",
    "min_variance",
    "min_volatility",
    "max_sharpe",
    "target_volatility",
    "target_return",
    "efficient_return",
    "risk_parity",
]

LOGGER = get_logger(__name__)
DEFAULT_SOLVER = "SLSQP"


def optimize_portfolio(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    *,
    method: OptimizationMethod,
    long_only: bool = True,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    weight_sum: float = 1.0,
    risk_free_rate: float = 0.0,
    target_return: float | None = None,
    target_volatility: float | None = None,
    asset_classes: pd.Series | None = None,
    asset_class_bounds: Mapping[str, tuple[float, float]] | None = None,
    ticker_bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    bond_assets: Sequence[str] | None = None,
    min_bond_exposure: float | None = None,
    expense_ratios: pd.Series | None = None,
    max_expense_ratio: float | None = None,
    initial_guess: pd.Series | None = None,
    previous_weights: pd.Series | None = None,
    max_turnover: float | None = None,
) -> pd.Series:
    """Optimize portfolio weights under investment constraints."""

    validated_method = _normalize_method(method)
    log_event(
        LOGGER,
        logging.INFO,
        "optimizer_started",
        method=validated_method,
        solver_used=DEFAULT_SOLVER,
        optimizer_status="started",
        asset_count=len(expected_returns),
        min_weight=min_weight,
        max_weight=max_weight,
        weight_sum=weight_sum,
        long_only=long_only,
        target_return=target_return,
        target_volatility=target_volatility,
        default_max_weight=max_weight,
        ticker_bound_count=len(ticker_bounds or {}),
        tightest_ticker_cap=_tightest_ticker_cap(ticker_bounds),
    )
    assets, mu, sigma = _validate_inputs(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        method=validated_method,
        long_only=long_only,
        min_weight=min_weight,
        max_weight=max_weight,
        weight_sum=weight_sum,
        target_return=target_return,
        target_volatility=target_volatility,
    )

    bounds = build_bounds(
        assets=assets,
        long_only=long_only,
        min_weight=min_weight,
        max_weight=max_weight,
        ticker_bounds=ticker_bounds,
    )
    _validate_bound_feasibility(
        lower_bounds=bounds.lb,
        upper_bounds=bounds.ub,
        weight_sum=weight_sum,
        long_only=long_only,
    )
    linear_constraints = build_linear_constraints(
        assets=assets,
        weight_sum=weight_sum,
        asset_classes=asset_classes,
        asset_class_bounds=asset_class_bounds,
        bond_assets=bond_assets,
        min_bond_exposure=min_bond_exposure,
        expense_ratios=expense_ratios,
        max_expense_ratio=max_expense_ratio,
    )
    validate_linear_feasibility(bounds=bounds, constraints=linear_constraints)
    initial_weights = _initial_weights(
        assets=assets,
        covariance_matrix=sigma,
        method=validated_method,
        weight_sum=weight_sum,
        lower_bounds=bounds.lb,
        upper_bounds=bounds.ub,
    )
    nonlinear_constraints = build_nonlinear_constraints(
        covariance_matrix=sigma,
        previous_weights=previous_weights,
        max_turnover=max_turnover,
        target_return=target_return,
        target_volatility=target_volatility,
        expected_returns=mu,
        method=validated_method,
    )

    if (
        validated_method == "equal_weight"
        and not nonlinear_constraints
        and _weights_within_bounds(initial_weights, bounds)
        and check_linear_feasibility(initial_weights, linear_constraints)
    ):
        weights = pd.Series(initial_weights, index=assets, dtype=float, name="weight")
        _log_optimizer_completion(
            weights,
            expected_returns=mu,
            covariance_matrix=sigma,
            risk_free_rate=risk_free_rate,
            method=validated_method,
            solver_used="direct_equal_weight",
            default_max_weight=max_weight,
            ticker_bounds=ticker_bounds,
        )
        return weights

    objective = build_objective_function(
        method=validated_method,
        expected_returns=mu,
        covariance_matrix=sigma,
        risk_free_rate=risk_free_rate,
    )
    candidate_initial_weights = _build_candidate_initial_weights(
        initial_weights=initial_weights,
        initial_guess=initial_guess.reindex(assets).to_numpy(dtype=float)
        if initial_guess is not None
        else None,
        expected_returns=mu,
        covariance_matrix=sigma,
        lower_bounds=bounds.lb,
        upper_bounds=bounds.ub,
        weight_sum=weight_sum,
        linear_constraints=linear_constraints,
    )
    result = _solve_with_retries(
        objective=objective,
        candidate_initial_weights=candidate_initial_weights,
        bounds=bounds,
        constraints=[*linear_constraints, *nonlinear_constraints],
    )

    if not result.success:
        log_event(
            LOGGER,
            logging.ERROR,
            "optimizer_failed",
            method=validated_method,
            solver_used=DEFAULT_SOLVER,
            optimizer_status="failed",
            reason=str(result.message),
            status_code=getattr(result, "status", None),
            target_return=target_return,
            target_volatility=target_volatility,
            default_max_weight=max_weight,
            ticker_bound_count=len(ticker_bounds or {}),
            tightest_ticker_cap=_tightest_ticker_cap(ticker_bounds),
        )
        raise ValueError(f"Optimization failed: {result.message}")

    weights = pd.Series(result.x, index=assets, dtype=float, name="weight")
    weights[np.abs(weights) < 1e-10] = 0.0
    _log_optimizer_completion(
        weights,
        expected_returns=mu,
        covariance_matrix=sigma,
        risk_free_rate=risk_free_rate,
        method=validated_method,
        solver_used=DEFAULT_SOLVER,
        default_max_weight=max_weight,
        ticker_bounds=ticker_bounds,
    )
    return weights


def _normalize_method(method: OptimizationMethod) -> OptimizationMethod:
    aliases = {
        "min_volatility": "min_variance",
        "efficient_return": "target_return",
    }
    return aliases.get(method, method)  # type: ignore[return-value]


def _validate_inputs(
    *,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    method: OptimizationMethod,
    long_only: bool,
    min_weight: float,
    max_weight: float,
    weight_sum: float,
    target_return: float | None,
    target_volatility: float | None,
) -> tuple[pd.Index, pd.Series, pd.DataFrame]:
    if expected_returns.empty:
        raise ValueError("expected_returns must not be empty.")
    if covariance_matrix.empty:
        raise ValueError("covariance_matrix must not be empty.")
    if not covariance_matrix.index.equals(covariance_matrix.columns):
        raise ValueError("covariance_matrix index and columns must match.")
    if not expected_returns.index.equals(covariance_matrix.index):
        raise ValueError("expected_returns and covariance_matrix must align on the same assets.")
    if weight_sum <= 0.0:
        raise ValueError("weight_sum must be positive.")
    if max_weight <= 0.0 or max_weight > 1.0:
        raise ValueError("max_weight must be in the interval (0, 1].")
    if min_weight < 0.0 and long_only:
        raise ValueError("min_weight must be non-negative for long-only portfolios.")
    if min_weight > max_weight:
        raise ValueError("min_weight must be less than or equal to max_weight.")
    if method == "target_return" and target_return is None:
        raise ValueError("target_return is required when method='target_return'.")
    if method == "target_volatility" and target_volatility is None:
        raise ValueError("target_volatility is required when method='target_volatility'.")
    if target_volatility is not None and target_volatility <= 0.0:
        raise ValueError("target_volatility must be positive.")

    return (
        expected_returns.index,
        expected_returns.astype(float),
        covariance_matrix.astype(float).loc[expected_returns.index, expected_returns.index],
    )


def _initial_weights(
    *,
    assets: pd.Index,
    covariance_matrix: pd.DataFrame,
    method: OptimizationMethod,
    weight_sum: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    if method == "inverse_volatility":
        inverse_vol = 1.0 / np.sqrt(np.diag(covariance_matrix))
        raw = inverse_vol / inverse_vol.sum()
    else:
        raw = np.full(len(assets), 1.0 / len(assets), dtype=float)

    clipped = np.clip(raw * weight_sum, lower_bounds, upper_bounds)
    if np.isclose(clipped.sum(), weight_sum):
        return clipped

    adjusted = np.full(len(assets), weight_sum / len(assets), dtype=float)
    return np.clip(adjusted, lower_bounds, upper_bounds)


def _build_candidate_initial_weights(
    *,
    initial_weights: np.ndarray,
    initial_guess: np.ndarray | None,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    weight_sum: float,
    linear_constraints: Sequence[dict[str, Any]],
) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []

    def add_candidate(raw: np.ndarray) -> None:
        repaired = _repair_initial_weights(
            raw,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            weight_sum=weight_sum,
        )
        if repaired is None:
            return
        if not check_linear_feasibility(repaired, linear_constraints):
            return
        if any(np.allclose(repaired, existing, atol=1e-8) for existing in candidates):
            return
        candidates.append(repaired)

    if initial_guess is not None:
        add_candidate(initial_guess)
    add_candidate(initial_weights)

    inverse_vol = 1.0 / np.sqrt(np.maximum(np.diag(covariance_matrix), 1e-12))
    add_candidate(inverse_vol / inverse_vol.sum())

    excess_returns = expected_returns.to_numpy(dtype=float)
    vol = np.sqrt(np.maximum(np.diag(covariance_matrix), 1e-12))
    add_candidate(excess_returns / np.maximum(vol, 1e-12))
    add_candidate(np.maximum(excess_returns, 0.0))

    return candidates or [initial_weights]


def _repair_initial_weights(
    raw: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    weight_sum: float,
) -> np.ndarray | None:
    if float(lower_bounds.sum()) - 1e-8 > weight_sum:
        return None
    if float(upper_bounds.sum()) + 1e-8 < weight_sum:
        return None

    sanitized = np.nan_to_num(np.asarray(raw, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if np.allclose(sanitized.sum(), 0.0):
        sanitized = np.ones_like(sanitized, dtype=float)
    preference = np.argsort(-sanitized)

    weights = lower_bounds.astype(float).copy()
    remaining = float(weight_sum - weights.sum())
    if remaining < -1e-8:
        return None

    for index in preference:
        room = float(upper_bounds[index] - weights[index])
        if room <= 0.0:
            continue
        allocation = min(room, remaining)
        weights[index] += allocation
        remaining -= allocation
        if remaining <= 1e-10:
            break

    if remaining > 1e-8:
        return None
    return weights


def _solve_with_retries(
    *,
    objective: Any,
    candidate_initial_weights: Sequence[np.ndarray],
    bounds: Any,
    constraints: Sequence[dict[str, Any]],
) -> Any:
    last_result: Any = None
    for candidate in candidate_initial_weights:
        result = minimize(
            objective,
            x0=candidate,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        last_result = result
        if result.success:
            return result
    return last_result


def _validate_bound_feasibility(
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    weight_sum: float,
    long_only: bool,
) -> None:
    if np.any(lower_bounds > upper_bounds + 1e-8):
        raise ValueError(
            "Constraints are infeasible: at least one ticker has min bound > max bound."
        )
    if long_only and np.any(lower_bounds < -1e-8):
        raise ValueError("ticker_bounds minimums must be non-negative for long-only portfolios.")
    if float(lower_bounds.sum()) - 1e-8 > weight_sum:
        raise ValueError(
            "Constraints are infeasible: sum(min_bounds) exceeds the requested weight_sum."
        )
    if float(upper_bounds.sum()) + 1e-8 < weight_sum:
        raise ValueError(
            "Constraints are infeasible: sum(max_bounds) is below the requested weight_sum."
        )


def _weights_within_bounds(weights: np.ndarray, bounds: Any) -> bool:
    return bool(
        np.all(weights >= np.asarray(bounds.lb, dtype=float) - 1e-8)
        and np.all(weights <= np.asarray(bounds.ub, dtype=float) + 1e-8)
    )


def _tightest_ticker_cap(
    ticker_bounds: Mapping[str, tuple[float | None, float | None]] | None,
) -> float | None:
    if not ticker_bounds:
        return None
    caps = [float(bounds[1]) for bounds in ticker_bounds.values() if bounds[1] is not None]
    return min(caps) if caps else None


def summarize_constraints(
    weights: pd.Series,
    *,
    asset_classes: pd.Series | None = None,
    expense_ratios: pd.Series | None = None,
    previous_weights: pd.Series | None = None,
) -> dict[str, float]:
    """Summarize realized exposure constraints for an optimized portfolio."""

    summary = {
        "weight_sum": float(weights.sum()),
        "min_weight": float(weights.min()),
        "max_weight": float(weights.max()),
    }

    if asset_classes is not None:
        aligned_classes = asset_classes.reindex(weights.index)
        if aligned_classes.isna().any():
            raise ValueError("asset_classes must contain entries for every asset.")
        for asset_class, exposure in weights.groupby(aligned_classes).sum().items():
            summary[f"asset_class::{asset_class}"] = float(exposure)

    if expense_ratios is not None:
        aligned_expense = expense_ratios.reindex(weights.index)
        if aligned_expense.isna().any():
            raise ValueError("expense_ratios must contain entries for every asset.")
        summary["weighted_expense_ratio"] = float(weights.dot(aligned_expense))

    if previous_weights is not None:
        aligned_previous = previous_weights.reindex(weights.index).fillna(0.0)
        summary["turnover"] = float((weights - aligned_previous).abs().sum())

    return summary


def _log_optimizer_completion(
    weights: pd.Series,
    *,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float,
    method: OptimizationMethod,
    solver_used: str,
    default_max_weight: float,
    ticker_bounds: Mapping[str, tuple[float | None, float | None]] | None,
) -> None:
    portfolio_return = calculate_portfolio_return(weights, expected_returns)
    portfolio_volatility = calculate_portfolio_volatility(weights, covariance_matrix)
    log_event(
        LOGGER,
        logging.INFO,
        "optimizer_completed",
        method=method,
        solver_used=solver_used,
        optimizer_status="success",
        portfolio_return=portfolio_return,
        portfolio_volatility=portfolio_volatility,
        portfolio_sharpe=calculate_sharpe_ratio(
            portfolio_return,
            portfolio_volatility,
            risk_free_rate,
        ),
        default_max_weight=default_max_weight,
        ticker_bound_count=len(ticker_bounds or {}),
        tightest_ticker_cap=_tightest_ticker_cap(ticker_bounds),
        realized_largest_weight=float(weights.max()),
        max_weight=float(weights.max()),
        weight_sum=float(weights.sum()),
    )


__all__ = [
    "OptimizationMethod",
    "optimize_portfolio",
    "summarize_constraints",
]
