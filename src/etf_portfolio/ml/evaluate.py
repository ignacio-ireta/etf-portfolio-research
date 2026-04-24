"""Chronological evaluation for ETF ML models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from etf_portfolio.config import MLTask, MLValidationConfig
from etf_portfolio.ml.train import fit_model


@dataclass(frozen=True)
class EvaluationResult:
    summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame


def chronological_train_test_split(
    dataset: pd.DataFrame,
    *,
    test_window_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a long-panel dataset into chronological train/test partitions."""

    unique_dates = _unique_dates(dataset)
    if test_window_periods <= 0 or test_window_periods >= len(unique_dates):
        raise ValueError("test_window_periods must be between 1 and len(unique_dates) - 1.")

    train_dates = unique_dates[:-test_window_periods]
    test_dates = unique_dates[-test_window_periods:]
    return _subset_by_dates(dataset, train_dates), _subset_by_dates(dataset, test_dates)


def walk_forward_evaluate(
    dataset: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    model_names: list[str],
    task: MLTask,
    validation: MLValidationConfig,
    random_state: int = 42,
) -> EvaluationResult:
    """Run grouped-by-date walk-forward validation over supported models."""

    unique_dates = _unique_dates(dataset)
    if len(unique_dates) < validation.min_train_periods + validation.test_window_periods:
        raise ValueError("Not enough dates are available for ML walk-forward validation.")

    fold_predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, float | str | int]] = []

    for model_name in model_names:
        for fold_number, (train_frame, test_frame) in enumerate(
            iter_walk_forward_splits(dataset, validation=validation),
            start=1,
        ):
            train_dates = train_frame.index.get_level_values("date")
            test_dates = test_frame.index.get_level_values("date")
            model = fit_model(
                train_frame,
                feature_columns=feature_columns,
                target_column=target_column,
                model_name=model_name,
                task=task,
                random_state=random_state,
            )
            X_test = test_frame.loc[:, feature_columns]
            y_test = test_frame.loc[:, target_column]
            prediction_frame = test_frame.loc[
                :,
                ["feature_end_date", "target_start_date", "target_end_date"],
            ].copy()
            prediction_frame["model"] = model_name
            prediction_frame["fold"] = fold_number
            prediction_frame["actual"] = y_test
            prediction_frame["prediction"] = model.predict(X_test)

            if task == "classification" and hasattr(model, "predict_proba"):
                prediction_frame["prediction_probability"] = model.predict_proba(X_test)[:, 1]

            fold_predictions.append(prediction_frame)
            metrics = _compute_metrics(prediction_frame, task=task)
            metrics.update(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "train_start_date": train_dates.min(),
                    "train_end_date": train_dates.max(),
                    "train_max_target_end_date": train_frame["target_end_date"].max(),
                    "test_start_date": test_dates.min(),
                    "test_end_date": test_dates.max(),
                    "test_min_feature_end_date": test_frame["feature_end_date"].min(),
                }
            )
            fold_metrics.append(metrics)

    predictions = pd.concat(fold_predictions).sort_index()
    metrics_frame = pd.DataFrame(fold_metrics)
    summary = metrics_frame.groupby("model", as_index=False).mean(numeric_only=True)
    return EvaluationResult(summary=summary, fold_metrics=metrics_frame, predictions=predictions)


def iter_walk_forward_splits(
    dataset: pd.DataFrame,
    *,
    validation: MLValidationConfig,
):
    """Yield chronological rolling train/test splits grouped by date."""

    unique_dates = _unique_dates(dataset)
    train_window = validation.train_window_periods
    test_window = validation.test_window_periods
    step = validation.step_periods
    embargo = validation.embargo_periods

    start = validation.min_train_periods
    stop = len(unique_dates) - test_window + 1
    for train_end in range(start, stop, step):
        train_start = max(0, train_end - train_window)
        train_dates = unique_dates[train_start:train_end]
        test_dates = unique_dates[train_end : train_end + test_window]
        train_frame = _subset_by_dates(dataset, train_dates)
        test_frame = _subset_by_dates(dataset, test_dates)
        if test_frame.empty:
            continue

        test_start_date = test_dates.min()
        test_feature_start = test_frame["feature_end_date"].min()

        # Purge train rows whose target window reaches into or beyond the test period.
        train_frame = train_frame.loc[train_frame["target_end_date"] < test_start_date]
        if embargo > 0:
            embargo_cutoff = test_feature_start - pd.tseries.offsets.BDay(embargo)
            train_frame = train_frame.loc[train_frame["feature_end_date"] < embargo_cutoff]

        if train_frame.empty:
            continue
        yield train_frame.copy(), test_frame


def _compute_metrics(predictions: pd.DataFrame, *, task: MLTask) -> dict[str, float]:
    if task == "classification":
        y_true = predictions["actual"].astype(int)
        y_pred = predictions["prediction"].astype(int)
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }
        if "prediction_probability" in predictions:
            y_score = predictions["prediction_probability"].astype(float)
            metrics["log_loss"] = float(log_loss(y_true, y_score, labels=[0, 1]))
            if y_true.nunique() > 1:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        return metrics

    y_true = predictions["actual"].astype(float)
    y_pred = predictions["prediction"].astype(float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _unique_dates(dataset: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(dataset.index, pd.MultiIndex) or dataset.index.names[:2] != [
        "date",
        "ticker",
    ]:
        raise ValueError("dataset must be indexed by ['date', 'ticker'].")
    dates = dataset.index.get_level_values("date").unique().sort_values()
    if len(dates) < 2:
        raise ValueError("dataset must contain at least two unique dates.")
    return pd.DatetimeIndex(dates)


def _subset_by_dates(dataset: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    mask = dataset.index.get_level_values("date").isin(dates)
    return dataset.loc[mask].copy()


__all__ = [
    "EvaluationResult",
    "chronological_train_test_split",
    "iter_walk_forward_splits",
    "walk_forward_evaluate",
]
