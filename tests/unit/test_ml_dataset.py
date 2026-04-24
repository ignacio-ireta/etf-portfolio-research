from __future__ import annotations

import numpy as np
import pandas as pd

from etf_portfolio.config import MLConfig
from etf_portfolio.ml.dataset import build_ml_dataset


def test_build_ml_dataset_prevents_feature_target_overlap() -> None:
    asset_returns = _make_asset_returns(periods=320)
    benchmark_returns = asset_returns.mean(axis=1).rename("VT")
    ml_config = MLConfig(
        task="regression",
        target="forward_return",
        horizon_periods=5,
    )

    dataset = build_ml_dataset(
        asset_returns,
        ml_config=ml_config,
        benchmark_returns=benchmark_returns,
    )

    assert not dataset.frame.empty
    assert (dataset.frame["feature_end_date"] < dataset.frame["target_start_date"]).all()
    assert (dataset.frame["target_start_date"] <= dataset.frame["target_end_date"]).all()
    assert "momentum_21" in dataset.feature_columns
    assert "benchmark_correlation_63" in dataset.feature_columns


def test_build_ml_dataset_aligns_forward_return_target() -> None:
    index = pd.bdate_range("2021-01-01", periods=30)
    asset_returns = pd.DataFrame({"VTI": np.linspace(0.001, 0.003, len(index))}, index=index)
    benchmark_returns = asset_returns["VTI"].rename("VT")
    ml_config = MLConfig(
        task="regression",
        target="forward_return",
        horizon_periods=3,
        features={
            "lag_periods": [1],
            "momentum_periods": [3],
            "volatility_windows": [3],
            "drawdown_windows": [3],
            "correlation_windows": [3],
            "moving_average_windows": [3],
        },
    )

    dataset = build_ml_dataset(
        asset_returns,
        ml_config=ml_config,
        benchmark_returns=benchmark_returns,
    )

    target_value = dataset.frame.loc[(index[5], "VTI"), "target"]
    expected = float(np.prod(1.0 + asset_returns["VTI"].iloc[6:9]) - 1.0)
    assert abs(target_value - expected) < 1e-12


def test_build_ml_dataset_supports_beat_benchmark_target() -> None:
    asset_returns = _make_asset_returns(periods=35)
    benchmark_returns = (asset_returns.mean(axis=1) - 0.0001).rename("VT")
    ml_config = MLConfig(
        task="classification",
        target="beat_benchmark",
        horizon_periods=5,
        models=["historical_mean"],
    )

    dataset = build_ml_dataset(
        asset_returns,
        ml_config=ml_config,
        benchmark_returns=benchmark_returns,
    )

    assert set(dataset.frame["target"].unique()) <= {0, 1}


def _make_asset_returns(periods: int) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=periods)
    series_one = 0.001 + 0.0004 * np.sin(np.arange(periods) / 4)
    series_two = 0.0008 + 0.0003 * np.cos(np.arange(periods) / 5)
    return pd.DataFrame({"VTI": series_one, "BND": series_two}, index=index)
