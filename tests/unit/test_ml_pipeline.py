from __future__ import annotations

import numpy as np
import pandas as pd

from etf_portfolio.config import MLValidationConfig
from etf_portfolio.ml.dataset import build_ml_dataset
from etf_portfolio.ml.evaluate import chronological_train_test_split, walk_forward_evaluate
from etf_portfolio.ml.governance import evaluate_leakage_checks


def test_chronological_train_test_split_respects_date_order() -> None:
    dataset = _make_dataset()

    train_frame, test_frame = chronological_train_test_split(
        dataset.frame,
        test_window_periods=5,
    )

    assert (
        train_frame.index.get_level_values("date").max()
        < test_frame.index.get_level_values("date").min()
    )


def test_walk_forward_evaluate_returns_model_metrics() -> None:
    dataset = _make_dataset()
    validation = MLValidationConfig(
        train_window_periods=25,
        test_window_periods=5,
        step_periods=5,
        min_train_periods=25,
    )

    result = walk_forward_evaluate(
        dataset.frame,
        feature_columns=dataset.feature_columns,
        target_column=dataset.target_column,
        model_names=["historical_mean", "ridge"],
        task="regression",
        validation=validation,
    )

    assert set(result.summary["model"]) == {"historical_mean", "ridge"}
    assert {"rmse", "mae", "r2"}.issubset(result.summary.columns)
    assert {
        "train_end_date",
        "test_start_date",
        "test_end_date",
        "train_max_target_end_date",
        "test_min_feature_end_date",
    }.issubset(result.fold_metrics.columns)
    assert not result.predictions.empty
    assert result.fold_metrics["fold"].nunique() >= 1


def test_evaluate_leakage_checks_passes_for_chronological_dataset() -> None:
    dataset = _make_dataset()
    validation = MLValidationConfig(
        train_window_periods=25,
        test_window_periods=5,
        step_periods=5,
        min_train_periods=25,
    )

    result = walk_forward_evaluate(
        dataset.frame,
        feature_columns=dataset.feature_columns,
        target_column=dataset.target_column,
        model_names=["historical_mean"],
        task="regression",
        validation=validation,
    )
    leakage_checks = evaluate_leakage_checks(dataset.frame, result.fold_metrics)

    assert leakage_checks == {
        "features_end_before_target_start": True,
        "target_window_is_forward_only": True,
        "walk_forward_splits_are_chronological": True,
        "purged_train_target_windows_before_test_features": True,
    }


def _make_dataset():
    index = pd.bdate_range("2020-01-01", periods=80)
    asset_returns = pd.DataFrame(
        {
            "VTI": 0.001 + 0.0003 * np.sin(np.arange(len(index)) / 5),
            "BND": 0.0005 + 0.0002 * np.cos(np.arange(len(index)) / 7),
            "IAU": 0.0007 + 0.00025 * np.sin(np.arange(len(index)) / 9),
        },
        index=index,
    )
    benchmark_returns = asset_returns.mean(axis=1).rename("VT")
    from etf_portfolio.config import MLConfig

    return build_ml_dataset(
        asset_returns,
        ml_config=MLConfig(
            task="regression",
            target="forward_return",
            horizon_periods=5,
            features={
                "lag_periods": [1, 5],
                "momentum_periods": [5, 10],
                "volatility_windows": [5],
                "drawdown_windows": [5],
                "correlation_windows": [5],
                "moving_average_windows": [5],
            },
        ),
        benchmark_returns=benchmark_returns,
    )
