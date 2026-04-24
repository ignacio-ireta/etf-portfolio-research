"""Leakage-safe dataset construction for ETF ML research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from etf_portfolio.config import MLConfig
from etf_portfolio.features.returns import drawdown_series


@dataclass(frozen=True)
class MLDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str


def build_ml_dataset(
    asset_returns: pd.DataFrame,
    *,
    ml_config: MLConfig,
    benchmark_returns: pd.Series | None = None,
    volume_data: pd.DataFrame | None = None,
    periods_per_year: int = 252,
) -> MLDataset:
    """Build a feature/target dataset without future-data leakage."""

    _validate_asset_returns(asset_returns)
    horizon = ml_config.horizon_periods
    benchmark = _validate_benchmark(benchmark_returns, asset_returns.index)

    features = _build_feature_frame(
        asset_returns,
        benchmark_returns=benchmark,
        volume_data=volume_data,
        ml_config=ml_config,
        periods_per_year=periods_per_year,
    )
    target = _build_target_frame(
        asset_returns,
        target=ml_config.target,
        horizon_periods=horizon,
        benchmark_returns=benchmark,
        periods_per_year=periods_per_year,
    )
    metadata = _build_metadata_frame(
        asset_returns.index,
        asset_returns.columns,
        horizon_periods=horizon,
    )

    feature_panel = _stack_frame(features)
    target_panel = _stack_frame(target).rename(columns={"value": "target"})
    dataset = feature_panel.join(target_panel, how="inner").join(metadata, how="left")
    dataset = dataset.dropna(subset=[*feature_panel.columns, "target"])
    dataset["target"] = dataset["target"].astype(float)

    if ml_config.task == "classification":
        dataset["target"] = dataset["target"].astype(int)

    return MLDataset(
        frame=dataset.sort_index(),
        feature_columns=list(feature_panel.columns),
        target_column="target",
    )


def _build_feature_frame(
    asset_returns: pd.DataFrame,
    *,
    benchmark_returns: pd.Series | None,
    volume_data: pd.DataFrame | None,
    ml_config: MLConfig,
    periods_per_year: int,
) -> dict[str, pd.DataFrame]:
    feature_frames: dict[str, pd.DataFrame] = {}
    synthetic_prices = (1.0 + asset_returns).cumprod()

    for lag in ml_config.features.lag_periods:
        feature_frames[f"lag_return_{lag}"] = asset_returns.shift(lag)

    for window in ml_config.features.momentum_periods:
        feature_frames[f"momentum_{window}"] = (1.0 + asset_returns).rolling(window).apply(
            np.prod, raw=True
        ) - 1.0

    for window in ml_config.features.volatility_windows:
        feature_frames[f"rolling_volatility_{window}"] = asset_returns.rolling(window).std(
            ddof=1
        ) * np.sqrt(periods_per_year)

    for window in ml_config.features.drawdown_windows:
        feature_frames[f"rolling_drawdown_{window}"] = asset_returns.rolling(window).apply(
            _window_max_drawdown,
            raw=False,
        )

    for window in ml_config.features.moving_average_windows:
        moving_average = synthetic_prices.rolling(window).mean()
        feature_frames[f"ma_distance_{window}"] = synthetic_prices.div(moving_average) - 1.0

    if benchmark_returns is not None:
        for window in ml_config.features.correlation_windows:
            benchmark_aligned = benchmark_returns.reindex(asset_returns.index)
            frame = pd.DataFrame(
                index=asset_returns.index,
                columns=asset_returns.columns,
                dtype=float,
            )
            for ticker in asset_returns.columns:
                frame[ticker] = asset_returns[ticker].rolling(window).corr(benchmark_aligned)
            feature_frames[f"benchmark_correlation_{window}"] = frame

    if volume_data is not None:
        aligned_volume = volume_data.reindex(
            index=asset_returns.index,
            columns=asset_returns.columns,
        )
        log_volume = np.log(aligned_volume.where(aligned_volume > 0))
        for window in (21, 63):
            rolling_mean = log_volume.rolling(window).mean()
            rolling_std = log_volume.rolling(window).std(ddof=1)
            feature_frames[f"log_volume_zscore_{window}"] = (
                log_volume - rolling_mean
            ) / rolling_std

    return feature_frames


def _build_target_frame(
    asset_returns: pd.DataFrame,
    *,
    target: str,
    horizon_periods: int,
    benchmark_returns: pd.Series | None,
    periods_per_year: int,
) -> pd.DataFrame:
    forward_returns = _forward_compounded_returns(asset_returns, horizon_periods)

    if target == "forward_return":
        return forward_returns

    if target == "forward_volatility":
        return _forward_volatility(
            asset_returns,
            horizon_periods,
            periods_per_year=periods_per_year,
        )

    if target == "forward_drawdown":
        return _forward_drawdown(asset_returns, horizon_periods)

    if target == "beat_benchmark":
        if benchmark_returns is None:
            raise ValueError("benchmark_returns is required for the beat_benchmark target.")
        benchmark_forward = _forward_compounded_returns(
            benchmark_returns.to_frame("benchmark"),
            horizon_periods,
        )["benchmark"]
        comparison = forward_returns.gt(benchmark_forward, axis=0)
        return comparison.astype(int)

    raise ValueError(f"Unsupported ML target: {target}.")


def _build_metadata_frame(
    index: pd.DatetimeIndex,
    columns: pd.Index,
    *,
    horizon_periods: int,
) -> pd.DataFrame:
    feature_end_dates = pd.Series(index, index=index, name="feature_end_date")
    shifted_start = pd.DatetimeIndex(list(index[1:]) + [pd.NaT])
    shifted_end = pd.DatetimeIndex(list(index[horizon_periods:]) + [pd.NaT] * horizon_periods)
    target_start_dates = pd.Series(shifted_start, index=index, name="target_start_date")
    target_end_dates = pd.Series(shifted_end, index=index, name="target_end_date")
    dates = pd.DataFrame(
        {
            "feature_end_date": feature_end_dates,
            "target_start_date": target_start_dates,
            "target_end_date": target_end_dates,
        }
    )
    metadata = []
    for ticker in columns:
        ticker_dates = dates.copy()
        ticker_dates["ticker"] = ticker
        metadata.append(ticker_dates.rename_axis("date").reset_index())
    frame = pd.concat(metadata, ignore_index=True).set_index(["date", "ticker"]).sort_index()
    return frame


def _stack_frame(feature_frames: dict[str, pd.DataFrame] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(feature_frames, pd.DataFrame):
        stacked = feature_frames.stack().rename("value").to_frame()
        stacked.index = stacked.index.set_names(["date", "ticker"])
        return stacked

    panels = []
    for name, frame in feature_frames.items():
        stacked = frame.stack().rename(name)
        stacked.index = stacked.index.set_names(["date", "ticker"])
        panels.append(stacked)
    return pd.concat(panels, axis=1)


def _forward_compounded_returns(returns: pd.DataFrame, horizon_periods: int) -> pd.DataFrame:
    shifted = (1.0 + returns).shift(-1)
    return (
        shifted.rolling(horizon_periods).apply(np.prod, raw=True).shift(-(horizon_periods - 1))
        - 1.0
    )


def _forward_volatility(
    returns: pd.DataFrame,
    horizon_periods: int,
    *,
    periods_per_year: int,
) -> pd.DataFrame:
    shifted = returns.shift(-1)
    return shifted.rolling(horizon_periods).std(ddof=1).shift(-(horizon_periods - 1)) * np.sqrt(
        periods_per_year
    )


def _forward_drawdown(returns: pd.DataFrame, horizon_periods: int) -> pd.DataFrame:
    shifted = returns.shift(-1)
    return (
        shifted.rolling(horizon_periods)
        .apply(_window_max_drawdown, raw=False)
        .shift(-(horizon_periods - 1))
    )


def _window_max_drawdown(window: pd.Series) -> float:
    return float(drawdown_series(window).min())


def _validate_asset_returns(asset_returns: pd.DataFrame) -> None:
    if asset_returns.empty:
        raise ValueError("asset_returns must not be empty.")
    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        raise ValueError("asset_returns index must be a DatetimeIndex.")
    if asset_returns.isna().any().any():
        raise ValueError("asset_returns must not contain missing values.")


def _validate_benchmark(
    benchmark_returns: pd.Series | None,
    index: pd.DatetimeIndex,
) -> pd.Series | None:
    if benchmark_returns is None:
        return None

    aligned = benchmark_returns.reindex(index)
    if aligned.isna().any():
        raise ValueError("benchmark_returns must align to all asset return dates.")
    return aligned.astype(float)


__all__ = ["MLDataset", "build_ml_dataset"]
