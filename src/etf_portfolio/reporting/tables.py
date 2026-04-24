"""Tabular report builders."""

from __future__ import annotations

import pandas as pd

from etf_portfolio.backtesting.metrics import (
    beta,
    compare_against_benchmarks,
    summarize_backtest_metrics,
)
from etf_portfolio.risk.exposure import (
    aggregate_group_exposure,
    latest_portfolio_weights,
    weighted_expense_ratio,
    weighted_expense_ratio_history,
)
from etf_portfolio.risk.stress import stress_period_returns


def build_metrics_table(
    portfolio_returns: pd.Series,
    *,
    weights: pd.DataFrame | None = None,
    periods_per_year: int,
    benchmark_returns: pd.Series | None = None,
    benchmark_suite: dict[str, pd.Series] | None = None,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Build a metrics table for the portfolio and optional benchmark suite."""

    effective_weights = weights
    if effective_weights is None:
        effective_weights = pd.DataFrame(
            1.0,
            index=portfolio_returns.index,
            columns=[portfolio_returns.name or "portfolio"],
        )

    if benchmark_suite:
        comparison = compare_against_benchmarks(
            portfolio_returns,
            weights=effective_weights,
            benchmark_returns=benchmark_suite,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        )
        comparison = comparison.round(6)
        comparison.index.name = "Strategy"
        return comparison.reset_index()

    summary = summarize_backtest_metrics(
        portfolio_returns,
        weights=effective_weights,
        periods_per_year=periods_per_year,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate,
    )
    metrics_table = summary.rename("Value").reset_index()
    metrics_table["Value"] = metrics_table["Value"].map(lambda value: round(value, 6))
    return metrics_table


def build_weights_table(weights: pd.DataFrame) -> pd.DataFrame:
    """Build a latest-weights table by ETF."""

    latest = latest_portfolio_weights(weights)
    return latest.rename_axis("ETF").rename("Weight").reset_index()


def build_group_exposure_table(
    weights: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    field: str,
    label: str,
) -> pd.DataFrame:
    """Build a latest-weights exposure table for a metadata grouping."""

    latest = latest_portfolio_weights(weights)
    metadata_by_ticker = _metadata_by_ticker(metadata)
    exposure = aggregate_group_exposure(latest, metadata_by_ticker.reindex(latest.index)[field])
    return exposure.rename_axis(label).rename("Weight").reset_index()


def build_portfolio_profile_table(
    portfolio_returns: pd.Series,
    *,
    weights: pd.DataFrame,
    metadata: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Build a compact table of portfolio profile diagnostics."""

    latest = latest_portfolio_weights(weights)
    metadata_by_ticker = _metadata_by_ticker(metadata)
    profile = {
        "Weighted Expense Ratio": weighted_expense_ratio(
            latest,
            metadata_by_ticker["expense_ratio"],
        ),
    }
    if benchmark_returns is not None:
        aligned_portfolio, aligned_benchmark = portfolio_returns.align(
            benchmark_returns,
            join="inner",
        )
        if not aligned_portfolio.empty:
            profile["Portfolio Beta"] = beta(aligned_portfolio, aligned_benchmark)

    return pd.Series(profile, dtype=float).rename("Value").rename_axis("Metric").reset_index()


def build_stress_period_table(
    portfolio_returns: pd.Series,
    *,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a stress-period returns table."""

    table = stress_period_returns(
        portfolio_returns,
        benchmark_returns=benchmark_returns,
    )
    if table.empty:
        return pd.DataFrame(columns=["Period", "Portfolio"])
    numeric_columns = table.columns.difference(["Period"])
    table.loc[:, numeric_columns] = table.loc[:, numeric_columns].round(6)
    return table


def build_weighted_expense_over_time_table(
    weights_history: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Build a per-rebalance-date table of weighted expense ratios."""

    metadata_by_ticker = _metadata_by_ticker(metadata)
    if "expense_ratio" not in metadata_by_ticker.columns:
        raise ValueError("metadata must contain an 'expense_ratio' column.")

    history = weighted_expense_ratio_history(
        weights_history,
        metadata_by_ticker["expense_ratio"],
    )
    table = history.rename("weighted_expense_ratio").rename_axis("rebalance_date").reset_index()
    table["weighted_expense_ratio"] = table["weighted_expense_ratio"].astype(float).round(6)
    return table


def build_etf_universe_summary_table(
    metadata: pd.DataFrame,
    *,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Build a compact ETF universe summary table from metadata."""

    if metadata.empty:
        raise ValueError("metadata must not be empty.")

    summary = metadata.copy()
    if "ticker" not in summary.columns:
        summary = summary.reset_index().rename(columns={"index": "ticker"})

    if tickers is not None:
        summary = summary.loc[summary["ticker"].isin(tickers)]

    preferred_columns = [
        "ticker",
        "name",
        "asset_class",
        "region",
        "sector",
        "currency",
        "expense_ratio",
        "benchmark_index",
        "inception_date",
        "notes",
    ]
    available_columns = [column for column in preferred_columns if column in summary.columns]
    return summary.loc[:, available_columns].sort_values("ticker").reset_index(drop=True)


def build_data_coverage_table(
    prices: pd.DataFrame,
    *,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a data-coverage table for the current price matrix."""

    if prices.empty:
        raise ValueError("prices must not be empty.")

    rows: list[dict[str, object]] = []
    metadata_by_ticker = None
    if metadata is not None:
        metadata_frame = metadata.copy()
        if "ticker" not in metadata_frame.columns:
            metadata_frame = metadata_frame.reset_index().rename(columns={"index": "ticker"})
        metadata_by_ticker = metadata_frame.set_index("ticker")

    for ticker in prices.columns:
        series = prices[ticker]
        non_null = series.dropna()
        row: dict[str, object] = {
            "ticker": ticker,
            "start_date": non_null.index.min().date().isoformat() if not non_null.empty else None,
            "end_date": non_null.index.max().date().isoformat() if not non_null.empty else None,
            "observations": int(non_null.shape[0]),
            "coverage_ratio": float(series.notna().mean()),
        }
        if metadata_by_ticker is not None and ticker in metadata_by_ticker.index:
            metadata_row = metadata_by_ticker.loc[ticker]
            for field in ("asset_class", "region", "currency", "inception_date"):
                if field in metadata_row.index:
                    value = metadata_row[field]
                    if hasattr(value, "date"):
                        value = value.date().isoformat()
                    row[field] = value
        rows.append(row)

    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def build_missing_data_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a missing-data diagnostics table for the price matrix."""

    if prices.empty:
        raise ValueError("prices must not be empty.")

    row_count = len(prices.index)
    rows = [
        {
            "ticker": ticker,
            "missing_count": int(prices[ticker].isna().sum()),
            "missing_fraction": float(prices[ticker].isna().mean()),
            "non_null_observations": int(prices[ticker].notna().sum()),
            "row_count": row_count,
        }
        for ticker in prices.columns
    ]
    table = pd.DataFrame(rows).sort_values(
        by=["missing_fraction", "ticker"],
        ascending=[False, True],
    )
    return table.reset_index(drop=True)


def _metadata_by_ticker(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata_frame = metadata.copy()
    if "ticker" in metadata_frame.columns:
        return metadata_frame.set_index("ticker")
    return metadata_frame
