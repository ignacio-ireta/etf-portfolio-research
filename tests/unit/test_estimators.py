from __future__ import annotations

import pandas as pd
import pytest

from etf_portfolio.features.estimators import (
    calculate_covariance_matrix,
    estimate_expected_returns,
)


def make_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "VTI": [0.01, 0.015, -0.005, 0.012],
            "BND": [0.002, 0.001, 0.003, 0.002],
            "IAU": [0.005, -0.002, 0.004, 0.006],
        }
    )


def test_estimate_expected_returns_historical_mean() -> None:
    result = estimate_expected_returns(make_returns(), periods_per_year=12)
    assert list(result.index) == ["VTI", "BND", "IAU"]
    assert (result > -1.0).all()


def test_calculate_covariance_matrix_sample() -> None:
    returns = make_returns()
    result = calculate_covariance_matrix(returns, method="sample", periods_per_year=12)
    expected = returns.cov() * 12
    pd.testing.assert_frame_equal(result, expected)


def test_calculate_covariance_matrix_ledoit_wolf_shape() -> None:
    result = calculate_covariance_matrix(make_returns(), method="ledoit_wolf", periods_per_year=12)
    assert result.shape == (3, 3)
    assert result.index.tolist() == ["VTI", "BND", "IAU"]
    assert result.columns.tolist() == ["VTI", "BND", "IAU"]


def test_calculate_covariance_matrix_rejects_missing_values() -> None:
    returns = make_returns()
    returns.loc[0, "VTI"] = None
    with pytest.raises(ValueError, match="missing values"):
        calculate_covariance_matrix(returns)
