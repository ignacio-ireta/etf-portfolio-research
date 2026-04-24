"""Per-objective tests for the optimizer's objective builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_portfolio.optimization.objectives import build_objective_function


def _expected_returns() -> pd.Series:
    return pd.Series({"AAA": 0.10, "BBB": 0.04, "CCC": 0.06})


def _covariance() -> pd.DataFrame:
    diag = np.array([0.04, 0.01, 0.02])
    return pd.DataFrame(
        np.diag(diag),
        index=_expected_returns().index,
        columns=_expected_returns().index,
    )


def test_min_variance_objective_returns_quadratic_form() -> None:
    fn = build_objective_function(
        method="min_variance",
        expected_returns=_expected_returns(),
        covariance_matrix=_covariance(),
        risk_free_rate=0.0,
    )
    weights = np.array([0.5, 0.3, 0.2])
    expected = float(weights @ _covariance().to_numpy() @ weights)
    assert fn(weights) == pytest.approx(expected)


def test_max_sharpe_objective_returns_negative_sharpe() -> None:
    fn = build_objective_function(
        method="max_sharpe",
        expected_returns=_expected_returns(),
        covariance_matrix=_covariance(),
        risk_free_rate=0.02,
    )
    weights = np.array([0.5, 0.3, 0.2])
    portfolio_return = float(weights @ _expected_returns().to_numpy())
    portfolio_vol = float(np.sqrt(weights @ _covariance().to_numpy() @ weights))
    expected = -(portfolio_return - 0.02) / portfolio_vol
    assert fn(weights) == pytest.approx(expected)


def test_max_sharpe_responds_to_risk_free_rate_change() -> None:
    weights = np.array([0.5, 0.3, 0.2])
    fn_low_rf = build_objective_function(
        method="max_sharpe",
        expected_returns=_expected_returns(),
        covariance_matrix=_covariance(),
        risk_free_rate=0.0,
    )
    fn_high_rf = build_objective_function(
        method="max_sharpe",
        expected_returns=_expected_returns(),
        covariance_matrix=_covariance(),
        risk_free_rate=0.05,
    )
    assert fn_low_rf(weights) != pytest.approx(fn_high_rf(weights))


def test_max_sharpe_objective_handles_zero_volatility_safely() -> None:
    expected = pd.Series({"AAA": 0.10, "BBB": 0.04})
    sigma = pd.DataFrame(np.zeros((2, 2)), index=expected.index, columns=expected.index)
    fn = build_objective_function(
        method="max_sharpe",
        expected_returns=expected,
        covariance_matrix=sigma,
        risk_free_rate=0.0,
    )
    assert fn(np.array([0.5, 0.5])) == pytest.approx(1e9)


def test_target_volatility_objective_maximizes_return() -> None:
    fn = build_objective_function(
        method="target_volatility",
        expected_returns=_expected_returns(),
        covariance_matrix=_covariance(),
        risk_free_rate=0.0,
    )
    weights_a = np.array([0.5, 0.3, 0.2])
    weights_b = np.array([0.2, 0.3, 0.5])
    assert fn(weights_a) == pytest.approx(-float(weights_a @ _expected_returns().to_numpy()))
    assert fn(weights_a) < fn(weights_b), (
        "Higher expected return should map to a smaller (more negative) objective."
    )


def test_target_return_objective_minimizes_variance() -> None:
    fn = build_objective_function(
        method="target_return",
        expected_returns=_expected_returns(),
        covariance_matrix=_covariance(),
        risk_free_rate=0.0,
    )
    weights = np.array([0.5, 0.3, 0.2])
    expected = float(weights @ _covariance().to_numpy() @ weights)
    assert fn(weights) == pytest.approx(expected)


def test_equal_weight_objective_is_zero_at_equal_weights() -> None:
    fn = build_objective_function(
        method="equal_weight",
        expected_returns=_expected_returns(),
        covariance_matrix=_covariance(),
        risk_free_rate=0.0,
    )
    target = np.full(3, 1.0 / 3)
    assert fn(target) == pytest.approx(0.0)
    assert fn(np.array([1.0, 0.0, 0.0])) > 0.0


def test_inverse_volatility_objective_zero_at_inverse_vol_weights() -> None:
    sigma = _covariance().to_numpy()
    inverse_vol = 1.0 / np.sqrt(np.diag(sigma))
    target = inverse_vol / inverse_vol.sum()

    fn = build_objective_function(
        method="inverse_volatility",
        expected_returns=_expected_returns(),
        covariance_matrix=_covariance(),
        risk_free_rate=0.0,
    )
    assert fn(target) == pytest.approx(0.0)
    assert fn(np.full(3, 1.0 / 3)) > 0.0


def test_risk_parity_objective_zero_when_risk_contributions_equalize() -> None:
    sigma = pd.DataFrame(
        np.diag([0.04, 0.04, 0.04]),
        index=_expected_returns().index,
        columns=_expected_returns().index,
    )
    fn = build_objective_function(
        method="risk_parity",
        expected_returns=_expected_returns(),
        covariance_matrix=sigma,
        risk_free_rate=0.0,
    )
    equal = np.full(3, 1.0 / 3)
    assert fn(equal) == pytest.approx(0.0, abs=1e-12)


def test_unsupported_method_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported optimization method"):
        build_objective_function(
            method="not_a_real_method",  # type: ignore[arg-type]
            expected_returns=_expected_returns(),
            covariance_matrix=_covariance(),
            risk_free_rate=0.0,
        )
