"""Statistical estimators for portfolio construction."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.covariance import LedoitWolf

from etf_portfolio.features.returns import annualize_return

CovarianceMethod = Literal["sample", "ledoit_wolf"]
ExpectedReturnMethod = Literal["historical_mean"]


def estimate_expected_returns(
    returns: pd.DataFrame,
    *,
    method: ExpectedReturnMethod = "historical_mean",
    periods_per_year: int = 252,
) -> pd.Series:
    """Estimate annualized expected returns from a historical return matrix."""

    _validate_returns_frame(returns)

    if method != "historical_mean":
        raise ValueError(f"Unsupported expected return estimator: {method}.")

    return pd.Series(
        annualize_return(returns, periods_per_year=periods_per_year),
        index=returns.columns,
        dtype=float,
        name="expected_return",
    )


def calculate_covariance_matrix(
    returns: pd.DataFrame,
    *,
    method: CovarianceMethod = "sample",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Estimate an annualized covariance matrix from asset returns."""

    _validate_returns_frame(returns)

    if method == "sample":
        covariance = returns.cov()
    elif method == "ledoit_wolf":
        estimator = LedoitWolf().fit(returns.to_numpy())
        covariance = pd.DataFrame(
            estimator.covariance_,
            index=returns.columns,
            columns=returns.columns,
            dtype=float,
        )
    else:
        raise ValueError(f"Unsupported covariance estimator: {method}.")

    return covariance * periods_per_year


def _validate_returns_frame(returns: pd.DataFrame) -> None:
    if returns.empty:
        raise ValueError("returns must not be empty.")
    if returns.isna().any().any():
        raise ValueError("returns must not contain missing values.")


__all__ = [
    "calculate_covariance_matrix",
    "estimate_expected_returns",
]
