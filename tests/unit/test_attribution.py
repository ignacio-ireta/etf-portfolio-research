"""Tests for return and risk attribution decompositions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_portfolio.risk.attribution import (
    asset_class_return_attribution,
    asset_class_risk_attribution,
    return_attribution,
    risk_attribution,
)


def _make_returns() -> pd.DataFrame:
    index = pd.date_range("2024-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "AAA": [0.010, 0.020, -0.005, 0.015, 0.000, 0.025],
            "BBB": [0.001, 0.002, 0.003, 0.000, -0.002, 0.004],
            "CCC": [0.020, -0.010, 0.005, 0.010, 0.012, -0.003],
        },
        index=index,
    )


def _make_weights_history(returns: pd.DataFrame) -> pd.DataFrame:
    rebalance_dates = returns.index[[0, 3]]
    return pd.DataFrame(
        [[0.5, 0.3, 0.2], [0.6, 0.2, 0.2]],
        index=rebalance_dates,
        columns=returns.columns,
    )


def test_return_attribution_per_period_sum_equals_portfolio_return() -> None:
    returns = _make_returns()
    weights_history = _make_weights_history(returns)

    attribution = return_attribution(weights_history, returns)
    weights_active = weights_history.reindex(attribution.index, method="ffill")
    expected_portfolio_returns = (weights_active * returns.loc[attribution.index]).sum(axis=1)

    np.testing.assert_allclose(
        attribution.sum(axis=1).to_numpy(),
        expected_portfolio_returns.to_numpy(),
        atol=1e-12,
    )


def test_return_attribution_total_equals_sum_of_period_portfolio_returns() -> None:
    returns = _make_returns()
    weights_history = _make_weights_history(returns)

    attribution = return_attribution(weights_history, returns)
    total_attribution = attribution.sum(axis=1).sum()
    weights_active = weights_history.reindex(attribution.index, method="ffill")
    portfolio_total = (weights_active * returns.loc[attribution.index]).sum(axis=1).sum()

    assert total_attribution == pytest.approx(portfolio_total, abs=1e-8)


def test_return_attribution_uses_only_assets_present_in_both_inputs() -> None:
    returns = _make_returns()
    weights_history = _make_weights_history(returns).drop(columns=["CCC"])

    attribution = return_attribution(weights_history, returns)

    assert list(attribution.columns) == ["AAA", "BBB"]
    weights_active = weights_history.reindex(attribution.index, method="ffill")
    expected = (weights_active * returns.loc[attribution.index, ["AAA", "BBB"]]).sum(axis=1)
    np.testing.assert_allclose(
        attribution.sum(axis=1).to_numpy(),
        expected.to_numpy(),
        atol=1e-12,
    )


def test_risk_attribution_sums_to_portfolio_volatility() -> None:
    weights = pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
    covariance = pd.DataFrame(
        [
            [0.04, 0.01, 0.005],
            [0.01, 0.02, 0.001],
            [0.005, 0.001, 0.03],
        ],
        index=weights.index,
        columns=weights.index,
    )

    contributions = risk_attribution(weights, covariance)

    portfolio_vol = float(np.sqrt(weights.to_numpy() @ covariance.to_numpy() @ weights.to_numpy()))
    assert contributions.sum() == pytest.approx(portfolio_vol, abs=1e-12)


def test_risk_attribution_is_zero_for_zero_variance_portfolio() -> None:
    weights = pd.Series({"AAA": 1.0, "BBB": 0.0})
    covariance = pd.DataFrame(
        np.zeros((2, 2)),
        index=weights.index,
        columns=weights.index,
    )

    contributions = risk_attribution(weights, covariance)

    np.testing.assert_array_equal(contributions.to_numpy(), np.zeros(len(weights)))


def test_risk_attribution_rejects_missing_assets() -> None:
    weights = pd.Series({"AAA": 1.0})
    covariance = pd.DataFrame(
        np.eye(2),
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
    )

    with pytest.raises(ValueError, match="missing entries for assets"):
        risk_attribution(weights, covariance)


def test_asset_class_return_attribution_aggregates_columns() -> None:
    returns = _make_returns()
    weights_history = _make_weights_history(returns)
    attribution = return_attribution(weights_history, returns)
    classes = pd.Series(
        {"AAA": "equity", "BBB": "fixed_income", "CCC": "equity"},
    )

    grouped = asset_class_return_attribution(attribution, classes)

    assert set(grouped.columns) == {"equity", "fixed_income"}
    np.testing.assert_allclose(
        grouped.sum(axis=1).to_numpy(),
        attribution.sum(axis=1).to_numpy(),
        atol=1e-12,
    )


def test_asset_class_risk_attribution_sums_to_portfolio_volatility() -> None:
    weights = pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
    covariance = pd.DataFrame(
        [
            [0.04, 0.01, 0.005],
            [0.01, 0.02, 0.001],
            [0.005, 0.001, 0.03],
        ],
        index=weights.index,
        columns=weights.index,
    )
    classes = pd.Series(
        {"AAA": "equity", "BBB": "fixed_income", "CCC": "equity"},
    )

    contributions = risk_attribution(weights, covariance)
    grouped = asset_class_risk_attribution(contributions, classes)

    portfolio_vol = float(np.sqrt(weights.to_numpy() @ covariance.to_numpy() @ weights.to_numpy()))
    assert grouped.sum() == pytest.approx(portfolio_vol, abs=1e-12)
