"""Constraint builders for the portfolio optimizer.

Kept narrow on purpose: each builder returns a list of scipy constraint
objects (or a `Bounds` instance) and validates its own inputs. The
orchestrator in `optimization.optimizer` composes them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog

if TYPE_CHECKING:
    from etf_portfolio.optimization.optimizer import OptimizationMethod


def build_bounds(
    *,
    assets: pd.Index,
    long_only: bool,
    min_weight: float,
    max_weight: float,
    ticker_bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> Bounds:
    """Per-asset bounds, optionally allowing shorts up to `max_weight`."""

    lower = np.full(len(assets), min_weight, dtype=float)
    if not long_only:
        lower = np.maximum(lower, -max_weight)
    upper = np.full(len(assets), max_weight, dtype=float)
    if ticker_bounds:
        normalized_bounds = {ticker.upper(): bounds for ticker, bounds in ticker_bounds.items()}
        unknown_tickers = sorted(
            normalized_bounds.keys() - {str(asset).upper() for asset in assets}
        )
        if unknown_tickers:
            raise ValueError(
                f"ticker_bounds contains unknown tickers not present in assets: {unknown_tickers}."
            )
        for idx, asset in enumerate(assets):
            ticker_min, ticker_max = normalized_bounds.get(str(asset).upper(), (None, None))
            if ticker_min is not None:
                lower[idx] = float(ticker_min)
            if ticker_max is not None:
                upper[idx] = float(ticker_max)
    return Bounds(lower, upper)


def build_linear_constraints(
    *,
    assets: pd.Index,
    weight_sum: float,
    asset_classes: pd.Series | None,
    asset_class_bounds: Mapping[str, tuple[float, float]] | None,
    bond_assets: Sequence[str] | None,
    min_bond_exposure: float | None,
    expense_ratios: pd.Series | None,
    max_expense_ratio: float | None,
) -> list[LinearConstraint]:
    """Compose the linear constraints used by the optimizer.

    Always includes the equality `sum(weights) == weight_sum`. Optionally
    includes asset-class bounds, a bond-floor, and a weighted expense-ratio cap.
    """

    constraints: list[LinearConstraint] = [
        LinearConstraint(np.ones((1, len(assets))), [weight_sum], [weight_sum]),
    ]
    constraints.extend(
        _asset_class_linear_constraints(
            assets=assets,
            asset_classes=asset_classes,
            asset_class_bounds=asset_class_bounds,
        )
    )
    bond_constraint = _bond_floor_linear_constraint(
        assets=assets,
        bond_assets=bond_assets,
        min_bond_exposure=min_bond_exposure,
    )
    if bond_constraint is not None:
        constraints.append(bond_constraint)

    expense_constraint = _expense_ratio_linear_constraint(
        assets=assets,
        expense_ratios=expense_ratios,
        max_expense_ratio=max_expense_ratio,
    )
    if expense_constraint is not None:
        constraints.append(expense_constraint)

    return constraints


def build_nonlinear_constraints(
    *,
    covariance_matrix: pd.DataFrame,
    previous_weights: pd.Series | None,
    max_turnover: float | None,
    target_return: float | None,
    target_volatility: float | None,
    expected_returns: pd.Series,
    method: OptimizationMethod,
) -> list[dict[str, Any]]:
    """Compose the nonlinear (SLSQP-style) constraints required by the method."""

    constraints: list[dict[str, Any]] = []

    turnover_constraint = _turnover_constraint(
        previous_weights=previous_weights,
        max_turnover=max_turnover,
        expected_returns=expected_returns,
    )
    if turnover_constraint is not None:
        constraints.append(turnover_constraint)

    if method == "target_return" and target_return is not None:
        constraints.append(
            _target_return_constraint(
                expected_returns=expected_returns,
                target_return=target_return,
            )
        )

    if method == "target_volatility" and target_volatility is not None:
        constraints.append(
            _target_volatility_constraint(
                covariance_matrix=covariance_matrix,
                target_volatility=target_volatility,
            )
        )

    return constraints


def check_linear_feasibility(
    weights: np.ndarray,
    constraints: Sequence[LinearConstraint],
) -> bool:
    """Return True iff `weights` satisfy all linear constraints within tolerance."""

    for constraint in constraints:
        value = constraint.A @ weights
        if np.any(value < constraint.lb - 1e-8) or np.any(value > constraint.ub + 1e-8):
            return False
    return True


def validate_linear_feasibility(
    *,
    bounds: Bounds,
    constraints: Sequence[LinearConstraint],
) -> None:
    """Fail early if no vector can satisfy bounds plus all linear constraints."""

    variable_count = len(bounds.lb)
    objective = np.zeros(variable_count, dtype=float)
    variable_bounds = [
        (float(lower), float(upper)) for lower, upper in zip(bounds.lb, bounds.ub, strict=True)
    ]
    eq_rows: list[np.ndarray] = []
    eq_bounds: list[float] = []
    ub_rows: list[np.ndarray] = []
    ub_bounds: list[float] = []

    for constraint in constraints:
        matrix = np.asarray(constraint.A, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        lower_bounds = np.broadcast_to(np.asarray(constraint.lb, dtype=float), matrix.shape[0])
        upper_bounds = np.broadcast_to(np.asarray(constraint.ub, dtype=float), matrix.shape[0])

        for row, lower, upper in zip(matrix, lower_bounds, upper_bounds, strict=True):
            lower_is_finite = np.isfinite(lower)
            upper_is_finite = np.isfinite(upper)
            if lower_is_finite and upper_is_finite and np.isclose(lower, upper, atol=1e-10):
                eq_rows.append(row)
                eq_bounds.append(float(lower))
                continue
            if upper_is_finite:
                ub_rows.append(row)
                ub_bounds.append(float(upper))
            if lower_is_finite:
                ub_rows.append(-row)
                ub_bounds.append(float(-lower))

    result = linprog(
        objective,
        A_ub=np.vstack(ub_rows) if ub_rows else None,
        b_ub=np.asarray(ub_bounds, dtype=float) if ub_bounds else None,
        A_eq=np.vstack(eq_rows) if eq_rows else None,
        b_eq=np.asarray(eq_bounds, dtype=float) if eq_bounds else None,
        bounds=variable_bounds,
        method="highs",
    )
    if not result.success:
        raise ValueError(
            "Constraints are infeasible before optimization: no portfolio can satisfy "
            "the combined linear constraints (weight sum, ticker bounds, asset-class "
            "bounds, bond exposure, and expense cap)."
        )


def _asset_class_linear_constraints(
    *,
    assets: pd.Index,
    asset_classes: pd.Series | None,
    asset_class_bounds: Mapping[str, tuple[float, float]] | None,
) -> list[LinearConstraint]:
    if asset_classes is None or not asset_class_bounds:
        return []
    class_map = asset_classes.reindex(assets)
    if class_map.isna().any():
        raise ValueError("asset_classes must contain entries for every asset.")
    available_asset_classes = set(class_map.dropna().astype(str))
    unknown_asset_classes = sorted(set(asset_class_bounds) - available_asset_classes)
    if unknown_asset_classes:
        raise ValueError(
            "asset_class_bounds contains unknown asset classes not present in asset_classes: "
            f"{unknown_asset_classes}. available={sorted(available_asset_classes)}."
        )
    constraints: list[LinearConstraint] = []
    for asset_class, bounds in asset_class_bounds.items():
        lower_bound, upper_bound = bounds
        mask = (class_map == asset_class).astype(float).to_numpy()
        constraints.append(LinearConstraint(mask[None, :], [lower_bound], [upper_bound]))
    return constraints


def _bond_floor_linear_constraint(
    *,
    assets: pd.Index,
    bond_assets: Sequence[str] | None,
    min_bond_exposure: float | None,
) -> LinearConstraint | None:
    if min_bond_exposure is None:
        return None
    if bond_assets is None:
        raise ValueError("bond_assets is required when min_bond_exposure is set.")
    bond_set = {asset.upper() for asset in bond_assets}
    mask = np.array(
        [1.0 if asset.upper() in bond_set else 0.0 for asset in assets],
        dtype=float,
    )
    return LinearConstraint(mask[None, :], [min_bond_exposure], [np.inf])


def _expense_ratio_linear_constraint(
    *,
    assets: pd.Index,
    expense_ratios: pd.Series | None,
    max_expense_ratio: float | None,
) -> LinearConstraint | None:
    if max_expense_ratio is None:
        return None
    if expense_ratios is None:
        raise ValueError("expense_ratios is required when max_expense_ratio is set.")
    aligned_expense = expense_ratios.reindex(assets)
    if aligned_expense.isna().any():
        raise ValueError("expense_ratios must contain entries for every asset.")
    return LinearConstraint(
        aligned_expense.to_numpy(dtype=float)[None, :],
        [-np.inf],
        [max_expense_ratio],
    )


def _turnover_constraint(
    *,
    previous_weights: pd.Series | None,
    max_turnover: float | None,
    expected_returns: pd.Series,
) -> dict[str, Any] | None:
    if max_turnover is None:
        return None
    if previous_weights is None:
        raise ValueError("previous_weights is required when max_turnover is set.")
    aligned_previous = previous_weights.reindex(expected_returns.index)
    if aligned_previous.isna().any():
        raise ValueError("previous_weights must contain entries for every asset.")
    previous = aligned_previous.to_numpy(dtype=float)
    return {
        "type": "ineq",
        "fun": lambda weights: float(max_turnover - np.abs(weights - previous).sum()),
    }


def _target_return_constraint(
    *,
    expected_returns: pd.Series,
    target_return: float,
) -> dict[str, Any]:
    mu = expected_returns.to_numpy(dtype=float)
    return {
        "type": "ineq",
        "fun": lambda weights: float(weights.dot(mu) - target_return),
    }


def _target_volatility_constraint(
    *,
    covariance_matrix: pd.DataFrame,
    target_volatility: float,
) -> dict[str, Any]:
    sigma = covariance_matrix.to_numpy(dtype=float)
    return {
        "type": "ineq",
        "fun": lambda weights: float(
            target_volatility - np.sqrt(max(weights.T.dot(sigma).dot(weights), 0.0))
        ),
    }


__all__ = [
    "build_bounds",
    "build_linear_constraints",
    "build_nonlinear_constraints",
    "check_linear_feasibility",
    "validate_linear_feasibility",
]
