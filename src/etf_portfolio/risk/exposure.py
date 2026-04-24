"""Exposure analysis helpers for ETF portfolios."""

from __future__ import annotations

import pandas as pd


def latest_portfolio_weights(weights: pd.DataFrame) -> pd.Series:
    """Return the latest rebalance weights sorted from largest to smallest."""

    if weights.empty:
        raise ValueError("weights must not be empty.")
    latest = weights.iloc[-1].astype(float)
    return latest.sort_values(ascending=False)


def aggregate_group_exposure(
    weights: pd.Series,
    classifications: pd.Series,
    *,
    missing_label: str = "Unclassified",
) -> pd.Series:
    """Aggregate portfolio weights by a metadata classification."""

    if weights.empty:
        raise ValueError("weights must not be empty.")

    aligned_classes = classifications.reindex(weights.index)
    grouped = (
        pd.DataFrame(
            {
                "weight": weights.astype(float),
                "classification": aligned_classes.fillna(missing_label).astype(str),
            }
        )
        .groupby("classification", sort=True)["weight"]
        .sum()
        .sort_values(ascending=False)
    )
    grouped.index.name = "classification"
    return grouped


def weighted_expense_ratio(
    weights: pd.Series,
    expense_ratios: pd.Series,
) -> float:
    """Compute the weighted-average expense ratio of a portfolio."""

    if weights.empty:
        raise ValueError("weights must not be empty.")

    aligned_expense = expense_ratios.reindex(weights.index)
    if aligned_expense.isna().any():
        missing = aligned_expense.index[aligned_expense.isna()].tolist()
        raise ValueError(f"Missing expense ratios for tickers: {', '.join(missing)}.")

    return float(weights.astype(float).dot(aligned_expense.astype(float)))


def weighted_expense_ratio_history(
    weights_history: pd.DataFrame,
    expense_ratios: pd.Series,
) -> pd.Series:
    """Compute the weighted-average expense ratio at each rebalance date."""

    if weights_history.empty:
        raise ValueError("weights_history must not be empty.")

    aligned_expense = expense_ratios.reindex(weights_history.columns).astype(float)
    if aligned_expense.isna().any():
        missing = aligned_expense.index[aligned_expense.isna()].tolist()
        raise ValueError(f"Missing expense ratios for tickers: {', '.join(missing)}.")

    series = weights_history.astype(float).dot(aligned_expense)
    series.name = "weighted_expense_ratio"
    return series


__all__ = [
    "aggregate_group_exposure",
    "latest_portfolio_weights",
    "weighted_expense_ratio",
    "weighted_expense_ratio_history",
]
