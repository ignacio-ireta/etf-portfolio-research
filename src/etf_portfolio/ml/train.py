"""Training and artifact persistence for ETF ML models."""

from __future__ import annotations

import json
import math
import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from etf_portfolio.config import AppConfig, MLTask
from etf_portfolio.ml.registry import build_model


def fit_model(
    train_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    model_name: str,
    task: MLTask,
    random_state: int = 42,
):
    """Fit a model on the provided training frame."""

    model = build_model(model_name, task=task, random_state=random_state)
    model.fit(train_frame.loc[:, feature_columns], train_frame.loc[:, target_column])
    return model


def save_model_bundle(
    model,
    *,
    output_path: Path,
    feature_columns: list[str],
    target_column: str,
    metadata: dict[str, Any],
) -> Path:
    """Persist a trained model and its metadata to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "metadata": metadata,
    }
    with output_path.open("wb") as file_handle:
        pickle.dump(payload, file_handle)
    return output_path


def log_mlflow_run(
    *,
    config: AppConfig,
    run_id: str,
    metrics: dict[str, Any],
    params: dict[str, Any],
    artifacts: dict[str, str],
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Log an ML run to MLflow when the dependency is installed."""

    tracking = config.ml.tracking
    if not tracking.enable_mlflow:
        return {"requested": False, "active": False}

    try:
        import mlflow
    except ModuleNotFoundError:
        return {"requested": True, "active": False, "reason": "mlflow_not_installed"}

    mlflow.set_experiment(tracking.experiment_name)
    with mlflow.start_run(run_name=run_id):
        mlflow.log_params(_flatten(params))
        mlflow.log_metrics(_flatten_numeric(metrics))
        mlflow_tags = {
            "run_id": run_id,
            "dataset_version": tracking.dataset_version,
            "feature_version": tracking.feature_version,
            "task": config.ml.task,
            "target": config.ml.target,
        }
        if tags:
            mlflow_tags.update(tags)
        mlflow.set_tags(mlflow_tags)
        for artifact_path in artifacts.values():
            mlflow.log_artifact(artifact_path)
        run = mlflow.active_run()
        run_id = run.info.run_id if run is not None else None

    return {"requested": True, "active": True, "run_id": run_id}


def write_metrics_json(metrics: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            sanitize_json_payload(metrics, allow_nan=False),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return output_path


def sanitize_json_payload(payload: Any, *, allow_nan: bool = False) -> Any:
    if isinstance(payload, dict):
        return {
            str(sanitize_json_payload(key, allow_nan=allow_nan)): sanitize_json_payload(
                value,
                allow_nan=allow_nan,
            )
            for key, value in payload.items()
        }
    if isinstance(payload, (list, tuple)):
        return [sanitize_json_payload(value, allow_nan=allow_nan) for value in payload]
    if isinstance(payload, (pd.Timestamp, datetime, date)):
        return payload.isoformat()
    if isinstance(payload, pd.Series):
        return {
            str(sanitize_json_payload(key, allow_nan=allow_nan)): sanitize_json_payload(
                value,
                allow_nan=allow_nan,
            )
            for key, value in payload.items()
        }
    if isinstance(payload, pd.DataFrame):
        return sanitize_json_payload(payload.to_dict(orient="records"), allow_nan=allow_nan)
    if isinstance(payload, pd.Index):
        return [sanitize_json_payload(value, allow_nan=allow_nan) for value in payload.tolist()]
    if isinstance(payload, np.ndarray):
        return [sanitize_json_payload(value, allow_nan=allow_nan) for value in payload.tolist()]
    if isinstance(payload, np.generic):
        return sanitize_json_payload(payload.item(), allow_nan=allow_nan)
    if isinstance(payload, float):
        return payload if allow_nan or math.isfinite(payload) else None
    return payload


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten(value, prefix=full_key))
        else:
            flattened[full_key] = str(value)
    return flattened


def _flatten_numeric(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_numeric(value, prefix=full_key))
        elif isinstance(value, (int, float)):
            numeric = float(value)
            if math.isfinite(numeric):
                flattened[full_key] = numeric
    return flattened


__all__ = [
    "fit_model",
    "log_mlflow_run",
    "sanitize_json_payload",
    "save_model_bundle",
    "write_metrics_json",
]
