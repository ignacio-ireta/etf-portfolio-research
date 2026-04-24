"""Regression tests asserting the ML dataset never carries future information.

Mirrors the chronological invariants enforced by
`etf_portfolio.ml.governance.evaluate_leakage_checks`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_portfolio.config import MLConfig, MLValidationConfig
from etf_portfolio.ml.dataset import build_ml_dataset
from etf_portfolio.ml.evaluate import iter_walk_forward_splits
from etf_portfolio.ml.governance import evaluate_leakage_checks


def _make_returns(periods: int = 600, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=periods)
    return pd.DataFrame(
        {
            "VTI": rng.normal(loc=0.0006, scale=0.011, size=periods),
            "BND": rng.normal(loc=0.0002, scale=0.004, size=periods),
        },
        index=index,
    )


def _ml_config(horizon: int = 5) -> MLConfig:
    return MLConfig(
        task="regression",
        target="forward_return",
        horizon_periods=horizon,
        models=["historical_mean"],
    )


def test_feature_end_date_is_strictly_before_target_start_date() -> None:
    asset_returns = _make_returns()
    benchmark = asset_returns.mean(axis=1).rename("VT")
    dataset = build_ml_dataset(
        asset_returns,
        ml_config=_ml_config(),
        benchmark_returns=benchmark,
    )

    assert not dataset.frame.empty
    assert (dataset.frame["feature_end_date"] < dataset.frame["target_start_date"]).all()


def test_target_window_is_forward_only() -> None:
    asset_returns = _make_returns()
    benchmark = asset_returns.mean(axis=1).rename("VT")
    dataset = build_ml_dataset(
        asset_returns,
        ml_config=_ml_config(horizon=10),
        benchmark_returns=benchmark,
    )

    assert (dataset.frame["target_start_date"] <= dataset.frame["target_end_date"]).all()
    assert (dataset.frame["feature_end_date"] < dataset.frame["target_end_date"]).all()


def test_features_at_date_t_are_invariant_to_future_returns() -> None:
    """Replace asset returns from cutoff onward; features dated < cutoff must not change.

    This is the strongest leakage assertion for the ML dataset: any feature whose
    `feature_end_date < cutoff` was computed without ever observing returns
    at-or-after `cutoff`.
    """

    asset_returns = _make_returns()
    benchmark = asset_returns.mean(axis=1).rename("VT")
    cutoff = asset_returns.index[-100]

    baseline_dataset = build_ml_dataset(
        asset_returns,
        ml_config=_ml_config(),
        benchmark_returns=benchmark,
    )
    baseline_features = baseline_dataset.frame.loc[
        baseline_dataset.frame["feature_end_date"] < cutoff,
        baseline_dataset.feature_columns,
    ]

    poisoned_returns = asset_returns.copy()
    rng = np.random.default_rng(2026)
    mask = poisoned_returns.index >= cutoff
    poisoned_returns.loc[mask] = rng.normal(
        loc=-0.05,
        scale=0.30,
        size=(int(mask.sum()), poisoned_returns.shape[1]),
    )
    poisoned_benchmark = poisoned_returns.mean(axis=1).rename("VT")
    poisoned_dataset = build_ml_dataset(
        poisoned_returns,
        ml_config=_ml_config(),
        benchmark_returns=poisoned_benchmark,
    )
    poisoned_features = poisoned_dataset.frame.loc[
        poisoned_dataset.frame["feature_end_date"] < cutoff,
        poisoned_dataset.feature_columns,
    ]

    common_index = baseline_features.index.intersection(poisoned_features.index)
    assert len(common_index) > 0

    pd.testing.assert_frame_equal(
        baseline_features.loc[common_index].sort_index(),
        poisoned_features.loc[common_index].sort_index(),
        check_exact=True,
    )


def test_walk_forward_splits_are_chronological_and_pass_governance() -> None:
    asset_returns = _make_returns()
    benchmark = asset_returns.mean(axis=1).rename("VT")
    dataset = build_ml_dataset(
        asset_returns,
        ml_config=_ml_config(),
        benchmark_returns=benchmark,
    )

    validation = MLValidationConfig(
        train_window_periods=120,
        test_window_periods=21,
        step_periods=21,
        min_train_periods=120,
    )
    fold_records: list[dict[str, object]] = []
    for fold_number, (train_frame, test_frame) in enumerate(
        iter_walk_forward_splits(dataset.frame, validation=validation),
        start=1,
    ):
        train_dates = train_frame.index.get_level_values("date")
        test_dates = test_frame.index.get_level_values("date")
        fold_records.append(
            {
                "model": "historical_mean",
                "fold": fold_number,
                "train_end_date": train_dates.max(),
                "train_max_target_end_date": train_frame["target_end_date"].max(),
                "test_start_date": test_dates.min(),
                "test_end_date": test_dates.max(),
                "test_min_feature_end_date": test_frame["feature_end_date"].min(),
            }
        )
    fold_metrics = pd.DataFrame(fold_records)
    assert not fold_metrics.empty

    assert (fold_metrics["train_end_date"] < fold_metrics["test_start_date"]).all()
    assert (fold_metrics["test_start_date"] <= fold_metrics["test_end_date"]).all()
    assert (
        fold_metrics["train_max_target_end_date"] < fold_metrics["test_min_feature_end_date"]
    ).all()

    checks = evaluate_leakage_checks(dataset.frame, fold_metrics)
    assert all(checks.values()), checks


def test_walk_forward_purges_overlapping_target_windows() -> None:
    dates = pd.bdate_range("2024-01-01", periods=9)
    index = pd.MultiIndex.from_product([dates, ["VTI"]], names=["date", "ticker"])
    frame = pd.DataFrame(index=index)
    frame["feature_end_date"] = frame.index.get_level_values("date")
    frame["target_start_date"] = frame["feature_end_date"] + pd.tseries.offsets.BDay(1)
    frame["target_end_date"] = frame["feature_end_date"] + pd.tseries.offsets.BDay(2)

    validation = MLValidationConfig(
        train_window_periods=5,
        test_window_periods=2,
        step_periods=2,
        min_train_periods=5,
    )
    splits = list(iter_walk_forward_splits(frame, validation=validation))
    assert splits

    first_train, first_test = splits[0]
    test_start = first_test.index.get_level_values("date").min()

    naive_train_dates = dates[:5]
    assert len(naive_train_dates) == 5
    naive_rows = frame.loc[frame.index.get_level_values("date").isin(naive_train_dates)]
    overlapping_rows = naive_rows.loc[naive_rows["target_end_date"] >= test_start]
    assert not overlapping_rows.empty

    assert len(first_train) < len(naive_rows)
    assert (first_train["target_end_date"] < test_start).all()


def test_walk_forward_embargo_excludes_recent_training_rows() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    index = pd.MultiIndex.from_product([dates, ["VTI"]], names=["date", "ticker"])
    frame = pd.DataFrame(index=index)
    frame["feature_end_date"] = frame.index.get_level_values("date")
    frame["target_start_date"] = frame["feature_end_date"] + pd.tseries.offsets.BDay(1)
    frame["target_end_date"] = frame["feature_end_date"] + pd.tseries.offsets.BDay(1)

    no_embargo = MLValidationConfig(
        train_window_periods=6,
        test_window_periods=2,
        step_periods=2,
        min_train_periods=6,
        embargo_periods=0,
    )
    with_embargo = MLValidationConfig(
        train_window_periods=6,
        test_window_periods=2,
        step_periods=2,
        min_train_periods=6,
        embargo_periods=2,
    )

    train_no_embargo, test_frame = next(iter_walk_forward_splits(frame, validation=no_embargo))
    train_with_embargo, _ = next(iter_walk_forward_splits(frame, validation=with_embargo))
    test_feature_min = test_frame["feature_end_date"].min()

    assert len(train_with_embargo) < len(train_no_embargo)
    assert (
        train_with_embargo["feature_end_date"] < (test_feature_min - pd.tseries.offsets.BDay(2))
    ).all()
