"""Portfolio backtesting metrics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from etf_portfolio.features.returns import annualized_return, max_drawdown


def calculate_beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Compute beta of an asset or portfolio series relative to a benchmark."""

    aligned_returns, aligned_benchmark = _align_series(asset_returns, benchmark_returns)
    benchmark_variance = aligned_benchmark.var(ddof=1)
    if np.isclose(benchmark_variance, 0.0):
        raise ValueError("Benchmark variance must be non-zero to calculate beta.")
    return float(aligned_returns.cov(aligned_benchmark) / benchmark_variance)


def calculate_portfolio_return(
    weights: pd.Series,
    expected_returns: pd.Series,
) -> float:
    """Compute expected portfolio return from weights and asset expected returns."""

    aligned_weights = _align_vector(weights, expected_returns, value_name="expected_returns")
    return float(aligned_weights.dot(expected_returns))


def calculate_portfolio_volatility(
    weights: pd.Series,
    covariance_matrix: pd.DataFrame,
) -> float:
    """Compute portfolio volatility from weights and a covariance matrix."""

    if covariance_matrix.empty:
        raise ValueError("covariance_matrix must not be empty.")
    if not covariance_matrix.index.equals(covariance_matrix.columns):
        raise ValueError("covariance_matrix must have matching index and columns.")

    aligned_weights = _align_vector(
        weights,
        covariance_matrix.index.to_series(),
        value_name="covariance_matrix",
    )
    covariance = covariance_matrix.loc[aligned_weights.index, aligned_weights.index]
    variance = float(aligned_weights.T.dot(covariance).dot(aligned_weights))
    if variance < -1e-12:
        raise ValueError("Portfolio variance must not be negative.")
    return float(np.sqrt(max(variance, 0.0)))


def calculate_sharpe_ratio(
    portfolio_return: float,
    portfolio_volatility: float,
    risk_free_rate: float,
) -> float:
    """Compute the Sharpe ratio from annualized return and volatility inputs."""

    if portfolio_volatility <= 0.0:
        raise ValueError("portfolio_volatility must be positive.")
    return float((portfolio_return - risk_free_rate) / portfolio_volatility)


def alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Jensen's alpha relative to a benchmark."""

    aligned_returns, aligned_benchmark = _align_series(returns, benchmark_returns)
    portfolio_ann = cagr(aligned_returns, periods_per_year=periods_per_year)
    benchmark_ann = cagr(aligned_benchmark, periods_per_year=periods_per_year)
    portfolio_beta = calculate_beta(aligned_returns, aligned_benchmark)
    return float(
        portfolio_ann - (risk_free_rate + portfolio_beta * (benchmark_ann - risk_free_rate))
    )


def portfolio_return(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Compute periodic portfolio returns from asset returns and weights."""

    if asset_returns.empty:
        raise ValueError("Asset returns must not be empty.")

    missing_weights = asset_returns.columns.difference(weights.index)
    if not missing_weights.empty:
        raise ValueError(
            f"Weights are missing entries for asset columns: {', '.join(missing_weights)}."
        )

    aligned_weights = weights.reindex(asset_returns.columns)
    if aligned_weights.isna().any():
        raise ValueError("Weights must align to all asset return columns.")

    return asset_returns.mul(aligned_weights, axis=1).sum(axis=1)


def portfolio_volatility(returns: pd.Series, *, periods_per_year: int) -> float:
    """Compute annualized portfolio volatility from periodic returns."""

    _validate_non_empty(returns)
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe ratio from periodic returns."""

    _validate_non_empty(returns)

    excess_returns = returns - (risk_free_rate / periods_per_year)
    volatility = returns.std(ddof=1)
    return float(excess_returns.mean() / volatility * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sortino ratio from periodic returns."""

    _validate_non_empty(returns)

    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = np.minimum(excess_returns, 0.0)
    downside_deviation = np.sqrt(np.mean(np.square(downside_returns))) * np.sqrt(periods_per_year)
    return float(excess_returns.mean() * periods_per_year / downside_deviation)


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Compute portfolio beta relative to a benchmark return series."""

    return calculate_beta(returns, benchmark_returns)


def tracking_error(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    periods_per_year: int,
) -> float:
    """Compute annualized tracking error from active returns."""

    aligned_returns, aligned_benchmark = _align_series(returns, benchmark_returns)
    active_returns = aligned_returns - aligned_benchmark
    return float(active_returns.std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    periods_per_year: int,
) -> float:
    """Compute annualized information ratio from active returns."""

    aligned_returns, aligned_benchmark = _align_series(returns, benchmark_returns)
    active_returns = aligned_returns - aligned_benchmark
    active_volatility = active_returns.std(ddof=1)
    if np.isclose(active_volatility, 0.0):
        return 0.0
    return float(active_returns.mean() / active_volatility * np.sqrt(periods_per_year))


def cagr(returns: pd.Series, *, periods_per_year: int) -> float:
    """Compute compound annual growth rate from periodic returns."""

    return float(annualized_return(returns, periods_per_year=periods_per_year))


def calmar_ratio(returns: pd.Series, *, periods_per_year: int) -> float:
    """Compute the Calmar ratio from periodic returns."""

    growth_rate = cagr(returns, periods_per_year=periods_per_year)
    drawdown = float(max_drawdown(returns))
    return float(growth_rate / abs(drawdown))


def turnover(weights: pd.DataFrame) -> float:
    """Compute average one-way turnover across rebalance dates."""

    if weights.empty:
        raise ValueError("weights must not be empty.")
    if len(weights.index) == 1:
        return float(weights.iloc[0].abs().sum())

    changes = weights.fillna(0.0).diff().abs().sum(axis=1).dropna()
    return float(changes.mean())


def average_number_of_holdings(weights: pd.DataFrame, *, tolerance: float = 1e-8) -> float:
    """Compute the average number of active holdings across rebalance dates."""

    if weights.empty:
        raise ValueError("weights must not be empty.")
    return float((weights.abs() > tolerance).sum(axis=1).mean())


def largest_position(weights: pd.DataFrame) -> float:
    """Compute the largest realized portfolio position across all rebalances."""

    if weights.empty:
        raise ValueError("weights must not be empty.")
    return float(weights.max(axis=1).max())


def herfindahl_concentration_index(weights: pd.DataFrame) -> float:
    """Compute the average Herfindahl concentration index over rebalances."""

    if weights.empty:
        raise ValueError("weights must not be empty.")
    return float(weights.pow(2).sum(axis=1).mean())


def worst_month(returns: pd.Series) -> float:
    """Compute the worst compounded monthly return."""

    monthly = _aggregate_periodic_returns(returns, "ME")
    return float(monthly.min())


def best_month(returns: pd.Series) -> float:
    """Compute the best compounded monthly return."""

    monthly = _aggregate_periodic_returns(returns, "ME")
    return float(monthly.max())


def worst_quarter(returns: pd.Series) -> float:
    """Compute the worst compounded quarterly return."""

    quarterly = _aggregate_periodic_returns(returns, "QE")
    return float(quarterly.min())


def summarize_backtest_metrics(
    portfolio_returns: pd.Series,
    *,
    weights: pd.DataFrame,
    periods_per_year: int,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Build a comprehensive performance summary for one backtest series."""

    metrics: list[tuple[str, float]] = [
        ("CAGR", cagr(portfolio_returns, periods_per_year=periods_per_year)),
        (
            "Annualized Volatility",
            portfolio_volatility(portfolio_returns, periods_per_year=periods_per_year),
        ),
        (
            "Sharpe Ratio",
            sharpe_ratio(
                portfolio_returns,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            ),
        ),
        (
            "Sortino Ratio",
            sortino_ratio(
                portfolio_returns,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            ),
        ),
        ("Max Drawdown", float(max_drawdown(portfolio_returns))),
        ("Calmar Ratio", calmar_ratio(portfolio_returns, periods_per_year=periods_per_year)),
        ("Turnover", turnover(weights)),
        ("Average Number of Holdings", average_number_of_holdings(weights)),
        ("Largest Position", largest_position(weights)),
        ("Herfindahl Concentration Index", herfindahl_concentration_index(weights)),
        ("Worst Month", worst_month(portfolio_returns)),
        ("Worst Quarter", worst_quarter(portfolio_returns)),
        ("Best Month", best_month(portfolio_returns)),
    ]

    if benchmark_returns is not None:
        aligned_portfolio, aligned_benchmark = _align_series(
            portfolio_returns,
            benchmark_returns,
        )
        metrics.extend(
            [
                ("Beta", beta(aligned_portfolio, aligned_benchmark)),
                (
                    "Alpha",
                    alpha(
                        aligned_portfolio,
                        aligned_benchmark,
                        periods_per_year=periods_per_year,
                        risk_free_rate=risk_free_rate,
                    ),
                ),
                (
                    "Tracking Error",
                    tracking_error(
                        aligned_portfolio,
                        aligned_benchmark,
                        periods_per_year=periods_per_year,
                    ),
                ),
                (
                    "Information Ratio",
                    information_ratio(
                        aligned_portfolio,
                        aligned_benchmark,
                        periods_per_year=periods_per_year,
                    ),
                ),
            ]
        )

    summary = pd.Series(dict(metrics), dtype=float)
    summary.index.name = "Metric"
    return summary


def compare_against_benchmarks(
    portfolio_returns: pd.Series,
    *,
    weights: pd.DataFrame,
    benchmark_returns: Mapping[str, pd.Series],
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Summarize the portfolio and multiple benchmark strategies in one table."""

    rows: dict[str, pd.Series] = {
        "Optimized Strategy": summarize_backtest_metrics(
            portfolio_returns,
            weights=weights,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        )
    }

    for name, benchmark_series in benchmark_returns.items():
        rows[name] = summarize_backtest_metrics(
            benchmark_series,
            weights=_passive_benchmark_weights(benchmark_series, name=name),
            periods_per_year=periods_per_year,
            benchmark_returns=(
                portfolio_returns
                if name == "Previous Optimized Strategy"
                else benchmark_returns.get("Selected Benchmark ETF")
            ),
            risk_free_rate=risk_free_rate,
        )

    return pd.DataFrame(rows).T


def _align_series(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join="inner")
    _validate_non_empty(aligned_returns)
    _validate_non_empty(aligned_benchmark)
    return aligned_returns, aligned_benchmark


def _align_vector(
    weights: pd.Series,
    reference: pd.Series,
    *,
    value_name: str,
) -> pd.Series:
    if weights.empty:
        raise ValueError("weights must not be empty.")

    missing = reference.index.difference(weights.index)
    if not missing.empty:
        raise ValueError(
            f"weights are missing entries required by {value_name}: {', '.join(missing)}."
        )

    aligned_weights = weights.reindex(reference.index)
    if aligned_weights.isna().any():
        raise ValueError("weights must align exactly to the reference index.")
    return aligned_weights.astype(float)


def _validate_non_empty(data: pd.Series | pd.DataFrame) -> None:
    if data.empty:
        raise ValueError("Input data must not be empty.")


def _aggregate_periodic_returns(returns: pd.Series, frequency: str) -> pd.Series:
    _validate_non_empty(returns)
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns index must be a DatetimeIndex.")
    aggregated = (1.0 + returns).resample(frequency).prod() - 1.0
    aggregated = aggregated.dropna()
    _validate_non_empty(aggregated)
    return aggregated


def _passive_benchmark_weights(
    benchmark_series: pd.Series,
    *,
    name: str,
) -> pd.DataFrame:
    index = pd.Index(
        [
            benchmark_series.index.min(),
            benchmark_series.index.max(),
        ],
        name="rebalance_date",
    ).unique()
    return pd.DataFrame(
        1.0,
        index=index,
        columns=[name],
        dtype=float,
    )


__all__ = [
    "alpha",
    "average_number_of_holdings",
    "beta",
    "best_month",
    "cagr",
    "calculate_beta",
    "calculate_portfolio_return",
    "calculate_portfolio_volatility",
    "calculate_sharpe_ratio",
    "calmar_ratio",
    "compare_against_benchmarks",
    "herfindahl_concentration_index",
    "information_ratio",
    "largest_position",
    "portfolio_return",
    "portfolio_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "summarize_backtest_metrics",
    "tracking_error",
    "turnover",
    "worst_month",
    "worst_quarter",
]
