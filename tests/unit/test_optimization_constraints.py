"""Per-builder tests for optimizer constraint and bound builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import LinearConstraint

from etf_portfolio.optimization.constraints import (
    build_bounds,
    build_linear_constraints,
    build_nonlinear_constraints,
    check_linear_feasibility,
)


def _assets() -> pd.Index:
    return pd.Index(["VTI", "VEA", "BND", "IAU"], name="ticker")


def test_build_bounds_long_only_uses_min_weight_floor() -> None:
    assets = _assets()

    bounds = build_bounds(assets=assets, long_only=True, min_weight=0.0, max_weight=0.4)

    np.testing.assert_array_equal(bounds.lb, np.zeros(len(assets)))
    np.testing.assert_array_equal(bounds.ub, np.full(len(assets), 0.4))


def test_build_bounds_long_short_widens_lower_bound() -> None:
    assets = _assets()

    bounds = build_bounds(assets=assets, long_only=False, min_weight=-1.0, max_weight=0.5)

    np.testing.assert_array_equal(bounds.lb, np.full(len(assets), -0.5))
    np.testing.assert_array_equal(bounds.ub, np.full(len(assets), 0.5))


def test_build_bounds_long_short_respects_explicit_min_weight_floor() -> None:
    """Long/short bounds never go below `-max_weight`, but `min_weight` clips above."""

    assets = _assets()
    bounds = build_bounds(assets=assets, long_only=False, min_weight=-0.1, max_weight=0.5)

    np.testing.assert_array_equal(bounds.lb, np.full(len(assets), -0.1))
    np.testing.assert_array_equal(bounds.ub, np.full(len(assets), 0.5))


def test_linear_constraints_always_includes_weight_sum_equality() -> None:
    assets = _assets()

    constraints = build_linear_constraints(
        assets=assets,
        weight_sum=1.0,
        asset_classes=None,
        asset_class_bounds=None,
        bond_assets=None,
        min_bond_exposure=None,
        expense_ratios=None,
        max_expense_ratio=None,
    )

    assert len(constraints) == 1
    np.testing.assert_array_equal(constraints[0].A, np.ones((1, len(assets))))
    np.testing.assert_array_equal(constraints[0].lb, [1.0])
    np.testing.assert_array_equal(constraints[0].ub, [1.0])


def test_linear_constraints_emit_one_row_per_asset_class_bound() -> None:
    assets = _assets()
    asset_classes = pd.Series(
        {"VTI": "equity", "VEA": "equity", "BND": "fixed_income", "IAU": "commodity"}
    )

    constraints = build_linear_constraints(
        assets=assets,
        weight_sum=1.0,
        asset_classes=asset_classes,
        asset_class_bounds={"equity": (0.4, 0.7), "commodity": (0.0, 0.1)},
        bond_assets=None,
        min_bond_exposure=None,
        expense_ratios=None,
        max_expense_ratio=None,
    )

    assert len(constraints) == 3
    bounds_by_lo = sorted((c.lb[0], c.ub[0]) for c in constraints[1:])
    assert (0.0, 0.1) in bounds_by_lo
    assert (0.4, 0.7) in bounds_by_lo


def test_linear_constraints_reject_missing_asset_class_metadata() -> None:
    assets = _assets()
    asset_classes = pd.Series({"VTI": "equity", "VEA": "equity"})

    with pytest.raises(ValueError, match="asset_classes must contain entries"):
        build_linear_constraints(
            assets=assets,
            weight_sum=1.0,
            asset_classes=asset_classes,
            asset_class_bounds={"equity": (0.0, 1.0)},
            bond_assets=None,
            min_bond_exposure=None,
            expense_ratios=None,
            max_expense_ratio=None,
        )


def test_linear_constraints_bond_floor_requires_bond_assets() -> None:
    assets = _assets()

    with pytest.raises(ValueError, match="bond_assets is required"):
        build_linear_constraints(
            assets=assets,
            weight_sum=1.0,
            asset_classes=None,
            asset_class_bounds=None,
            bond_assets=None,
            min_bond_exposure=0.2,
            expense_ratios=None,
            max_expense_ratio=None,
        )


def test_linear_constraints_bond_floor_targets_only_bond_assets() -> None:
    assets = _assets()

    constraints = build_linear_constraints(
        assets=assets,
        weight_sum=1.0,
        asset_classes=None,
        asset_class_bounds=None,
        bond_assets=["BND"],
        min_bond_exposure=0.2,
        expense_ratios=None,
        max_expense_ratio=None,
    )

    bond_constraint = constraints[-1]
    np.testing.assert_array_equal(bond_constraint.A, np.array([[0.0, 0.0, 1.0, 0.0]]))
    assert bond_constraint.lb[0] == 0.2
    assert np.isinf(bond_constraint.ub[0])


def test_linear_constraints_expense_cap_uses_full_expense_ratio_vector() -> None:
    assets = _assets()
    expense = pd.Series({"VTI": 0.0003, "VEA": 0.0005, "BND": 0.0003, "IAU": 0.0025})

    constraints = build_linear_constraints(
        assets=assets,
        weight_sum=1.0,
        asset_classes=None,
        asset_class_bounds=None,
        bond_assets=None,
        min_bond_exposure=None,
        expense_ratios=expense,
        max_expense_ratio=0.0010,
    )

    expense_constraint = constraints[-1]
    np.testing.assert_allclose(expense_constraint.A, expense.reindex(assets).to_numpy()[None, :])
    assert np.isneginf(expense_constraint.lb[0])
    assert expense_constraint.ub[0] == pytest.approx(0.0010)


def test_linear_constraints_expense_cap_requires_expense_data() -> None:
    assets = _assets()

    with pytest.raises(ValueError, match="expense_ratios is required"):
        build_linear_constraints(
            assets=assets,
            weight_sum=1.0,
            asset_classes=None,
            asset_class_bounds=None,
            bond_assets=None,
            min_bond_exposure=None,
            expense_ratios=None,
            max_expense_ratio=0.001,
        )


def test_nonlinear_constraints_turnover_evaluates_to_remaining_budget() -> None:
    expected_returns = pd.Series([0.0, 0.0, 0.0], index=["AAA", "BBB", "CCC"])
    covariance = pd.DataFrame(
        np.eye(3),
        index=expected_returns.index,
        columns=expected_returns.index,
    )
    previous = pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})

    constraints = build_nonlinear_constraints(
        covariance_matrix=covariance,
        previous_weights=previous,
        max_turnover=0.4,
        target_return=None,
        target_volatility=None,
        expected_returns=expected_returns,
        method="min_variance",
    )

    assert len(constraints) == 1
    fn = constraints[0]["fun"]
    new_weights = np.array([0.55, 0.25, 0.20])
    expected_remaining = 0.4 - float(np.abs(new_weights - previous.to_numpy()).sum())
    assert fn(new_weights) == pytest.approx(expected_remaining)


def test_nonlinear_constraints_target_return_active_only_for_method() -> None:
    expected_returns = pd.Series([0.10, 0.05], index=["AAA", "BBB"])
    covariance = pd.DataFrame(
        np.eye(2),
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    none_method = build_nonlinear_constraints(
        covariance_matrix=covariance,
        previous_weights=None,
        max_turnover=None,
        target_return=0.08,
        target_volatility=None,
        expected_returns=expected_returns,
        method="min_variance",
    )
    assert none_method == []

    constraints = build_nonlinear_constraints(
        covariance_matrix=covariance,
        previous_weights=None,
        max_turnover=None,
        target_return=0.08,
        target_volatility=None,
        expected_returns=expected_returns,
        method="target_return",
    )
    assert len(constraints) == 1
    fn = constraints[0]["fun"]
    assert fn(np.array([1.0, 0.0])) == pytest.approx(0.10 - 0.08)
    assert fn(np.array([0.0, 1.0])) == pytest.approx(0.05 - 0.08)


def test_nonlinear_constraints_target_volatility_uses_psd_quadratic() -> None:
    expected_returns = pd.Series([0.10, 0.05], index=["AAA", "BBB"])
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.01]],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    constraints = build_nonlinear_constraints(
        covariance_matrix=covariance,
        previous_weights=None,
        max_turnover=None,
        target_return=None,
        target_volatility=0.15,
        expected_returns=expected_returns,
        method="target_volatility",
    )
    assert len(constraints) == 1

    fn = constraints[0]["fun"]
    weights = np.array([0.5, 0.5])
    realized_vol = np.sqrt(0.25 * 0.04 + 0.25 * 0.01)
    assert fn(weights) == pytest.approx(0.15 - realized_vol)


def test_check_linear_feasibility_accepts_satisfying_weights() -> None:
    constraint = LinearConstraint(np.ones((1, 3)), [1.0], [1.0])
    feasible = np.array([0.4, 0.4, 0.2])
    infeasible = np.array([0.4, 0.4, 0.4])

    assert check_linear_feasibility(feasible, [constraint])
    assert not check_linear_feasibility(infeasible, [constraint])
