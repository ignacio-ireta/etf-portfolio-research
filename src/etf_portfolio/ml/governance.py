"""Model governance artifacts and approval checks for ETF ML experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from etf_portfolio.config import AppConfig
from etf_portfolio.ml.dataset import MLDataset
from etf_portfolio.tracking import file_sha256


def evaluate_leakage_checks(dataset: pd.DataFrame, fold_metrics: pd.DataFrame) -> dict[str, bool]:
    """Evaluate the core chronological leakage checks for the ML dataset."""

    feature_before_target = bool((dataset["feature_end_date"] < dataset["target_start_date"]).all())
    target_window_ordered = bool((dataset["target_start_date"] <= dataset["target_end_date"]).all())

    split_columns = {
        "train_end_date",
        "test_start_date",
        "test_end_date",
    }
    purged_split_columns = {
        "train_max_target_end_date",
        "test_min_feature_end_date",
    }
    chronological_walk_forward = bool(
        not fold_metrics.empty
        and split_columns.issubset(fold_metrics.columns)
        and (fold_metrics["train_end_date"] < fold_metrics["test_start_date"]).all()
        and (fold_metrics["test_start_date"] <= fold_metrics["test_end_date"]).all()
    )
    purged_target_windows = bool(
        not fold_metrics.empty
        and purged_split_columns.issubset(fold_metrics.columns)
        and (
            fold_metrics["train_max_target_end_date"] < fold_metrics["test_min_feature_end_date"]
        ).all()
    )

    return {
        "features_end_before_target_start": feature_before_target,
        "target_window_is_forward_only": target_window_ordered,
        "walk_forward_splits_are_chronological": chronological_walk_forward,
        "purged_train_target_windows_before_test_features": purged_target_windows,
    }


def evaluate_model_governance(
    *,
    config: AppConfig,
    dataset: MLDataset,
    evaluation_summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    best_model_name: str,
    model_path: Path,
    leakage_checks: dict[str, bool],
) -> dict[str, Any]:
    """Determine whether a trained model is eligible for production use."""

    baseline_name = "historical_mean"
    fold_win_rate = _fold_win_rate(
        fold_metrics=fold_metrics,
        best_model_name=best_model_name,
        baseline_name=baseline_name,
        task=config.ml.task,
    )
    governance_config = config.ml.governance
    baseline_present = baseline_name in set(evaluation_summary["model"])
    beats_baseline = _beats_baseline(
        evaluation_summary=evaluation_summary,
        best_model_name=best_model_name,
        baseline_name=baseline_name,
        task=config.ml.task,
    )
    stable_across_windows = (
        fold_metrics["fold"].nunique() >= governance_config.minimum_folds_for_stability
        and fold_win_rate >= governance_config.minimum_fold_win_rate
    )
    leakage_passed = all(leakage_checks.values())
    reproducible_training = (
        file_sha256(model_path) is not None
        and bool(config.ml.tracking.dataset_version)
        and bool(config.ml.tracking.feature_version)
    )
    rollback_path_available = baseline_present
    documented_failure_modes = _failure_modes(config)
    eligible = all(
        [
            beats_baseline if governance_config.require_baseline_outperformance else True,
            leakage_passed if governance_config.require_leakage_checks else True,
            stable_across_windows,
            reproducible_training,
            rollback_path_available,
            bool(documented_failure_modes),
        ]
    )

    return {
        "approval_status": "eligible" if eligible else "research_only",
        "eligible_for_production": eligible,
        "checks": {
            "beats_baseline_out_of_sample": beats_baseline,
            "passes_leakage_checks": leakage_passed,
            "stable_across_multiple_time_windows": stable_across_windows,
            "reproducible_training": reproducible_training,
            "rollback_path_available": rollback_path_available,
            "documented_failure_modes": bool(documented_failure_modes),
        },
        "fold_win_rate_vs_baseline": fold_win_rate,
        "feature_set": dataset.feature_columns,
        "target_definition": {
            "task": config.ml.task,
            "target": config.ml.target,
            "horizon_periods": config.ml.horizon_periods,
        },
        "train_validation_test_windows": {
            "train_window_periods": config.ml.validation.train_window_periods,
            "test_window_periods": config.ml.validation.test_window_periods,
            "step_periods": config.ml.validation.step_periods,
            "min_train_periods": config.ml.validation.min_train_periods,
        },
        "leakage_checks": leakage_checks,
        "model_version": file_sha256(model_path),
        "failure_modes": documented_failure_modes,
    }


def write_model_card(
    *,
    output_path: Path,
    run_id: str,
    config: AppConfig,
    governance: dict[str, Any],
    summary: pd.DataFrame,
) -> Path:
    """Write a concise model card for governance review."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = summary.round(6).to_dict(orient="records")
    lines = [
        f"# Model Card: {run_id}",
        "",
        "## Overview",
        f"- Project: {config.project.name}",
        f"- Approval status: {governance['approval_status']}",
        f"- Task: {config.ml.task}",
        f"- Target: {config.ml.target}",
        f"- Horizon periods: {config.ml.horizon_periods}",
        f"- Models evaluated: {', '.join(config.ml.models)}",
        f"- Model artifact: {governance['model_artifact']}",
        f"- Training scope: {governance['model_training_scope']}",
        f"- Training scope detail: {governance['model_training_scope_description']}",
        f"- Training observations: {governance['training_observations']}",
        f"- Holdout observations: {governance['holdout_observations']}",
        "",
        "## Feature Set",
        f"- Features: {', '.join(governance['feature_set'])}",
        "",
        "## Validation Windows",
        f"- Train window: {config.ml.validation.train_window_periods}",
        f"- Test window: {config.ml.validation.test_window_periods}",
        f"- Step periods: {config.ml.validation.step_periods}",
        f"- Minimum train periods: {config.ml.validation.min_train_periods}",
        "",
        "## Leakage Checks",
    ]
    for name, passed in governance["leakage_checks"].items():
        lines.append(f"- {name}: {'pass' if passed else 'fail'}")
    lines.extend(
        [
            "",
            "## Approval Checks",
        ]
    )
    for name, passed in governance["checks"].items():
        lines.append(f"- {name}: {'pass' if passed else 'fail'}")
    lines.extend(
        [
            "",
            "## Failure Modes",
        ]
    )
    for failure_mode in governance["failure_modes"]:
        lines.append(f"- {failure_mode}")
    lines.extend(
        [
            "",
            "## Evaluation Summary",
            json.dumps(summary_rows, indent=2),
            "",
            "## Rollback Path",
            "- Revert to the historical_mean baseline and the prior approved model artifact.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _beats_baseline(
    *,
    evaluation_summary: pd.DataFrame,
    best_model_name: str,
    baseline_name: str,
    task: str,
) -> bool:
    if baseline_name not in set(evaluation_summary["model"]):
        return False

    summary = evaluation_summary.set_index("model")
    best = summary.loc[best_model_name]
    baseline = summary.loc[baseline_name]
    if task == "classification":
        return float(best["accuracy"]) > float(baseline["accuracy"])
    return float(best["rmse"]) < float(baseline["rmse"])


def _fold_win_rate(
    *,
    fold_metrics: pd.DataFrame,
    best_model_name: str,
    baseline_name: str,
    task: str,
) -> float:
    if baseline_name not in set(fold_metrics["model"]):
        return 0.0

    best = fold_metrics.loc[fold_metrics["model"] == best_model_name].set_index("fold")
    baseline = fold_metrics.loc[fold_metrics["model"] == baseline_name].set_index("fold")
    common_folds = best.index.intersection(baseline.index)
    if len(common_folds) == 0:
        return 0.0

    if task == "classification":
        wins = best.loc[common_folds, "accuracy"] > baseline.loc[common_folds, "accuracy"]
    else:
        wins = best.loc[common_folds, "rmse"] < baseline.loc[common_folds, "rmse"]
    return float(wins.mean())


def _failure_modes(config: AppConfig) -> list[str]:
    return [
        "Model performance can degrade under regime shifts not represented in the training window.",
        "Thin ETF history or benchmark instability can make target labels noisy.",
        (
            "Feature relationships estimated on daily data may not survive "
            "transaction costs or market structure changes."
        ),
        (
            "The model remains research-only when out-of-sample performance does not beat the "
            "historical_mean baseline consistently."
        ),
    ]


__all__ = [
    "evaluate_leakage_checks",
    "evaluate_model_governance",
    "write_model_card",
]
