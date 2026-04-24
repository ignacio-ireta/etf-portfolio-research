"""Experiment tracking utilities for reproducible research runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from etf_portfolio.config import AppConfig


def generate_run_id(stage: str) -> str:
    """Create a sortable run identifier for a pipeline stage."""

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stage}-{timestamp}-{uuid4().hex[:8]}"


def build_run_record(
    *,
    stage: str,
    run_id: str,
    config: AppConfig,
    project_root: Path,
    data_version_path: Path | None,
    output_artifacts: dict[str, Path | str],
    actual_start_date: str | None = None,
    actual_end_date: str | None = None,
    optimization_method: str | None = None,
    backtest_metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized run record for any research stage."""

    output_manifest = {
        name: _artifact_record(project_root, Path(path)) for name, path in output_artifacts.items()
    }
    record = {
        "run_id": run_id,
        "stage": stage,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit_hash": current_git_commit_hash(project_root),
        "config_hash": config_hash(config),
        "data_version": (
            _artifact_record(project_root, data_version_path)
            if data_version_path is not None
            else None
        ),
        "universe_id": universe_id(config.universe.tickers),
        "benchmark": config.benchmark.primary,
        "start_date": actual_start_date or config.data.start_date.isoformat(),
        "end_date": actual_end_date
        or (config.data.end_date.isoformat() if config.data.end_date is not None else None),
        "optimization_method": optimization_method,
        "risk_model": config.optimization.risk_model,
        "expected_return_estimator": config.optimization.expected_return_estimator,
        "constraints": config.constraints.model_dump(mode="json"),
        "output_artifacts": output_manifest,
    }
    if backtest_metrics is not None:
        record["backtest_metrics"] = backtest_metrics
    if extra:
        record.update(extra)
    return record


def write_run_record(record: dict[str, Any], *, artifact_dir: Path) -> Path:
    """Persist a run record and return the written path."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifact_dir / f"{record['stage']}_{record['run_id']}.json"
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def config_hash(config: AppConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def universe_id(tickers: list[str]) -> str:
    normalized = ",".join(sorted({ticker.strip().upper() for ticker in tickers}))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def current_git_commit_hash(project_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    commit_hash = completed.stdout.strip()
    return commit_hash or None


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(project_root: Path, path: Path) -> dict[str, Any]:
    resolved_path = path if path.is_absolute() else project_root / path
    return {
        "path": str(resolved_path),
        "exists": resolved_path.exists(),
        "sha256": file_sha256(resolved_path),
    }


__all__ = [
    "build_run_record",
    "config_hash",
    "current_git_commit_hash",
    "file_sha256",
    "generate_run_id",
    "universe_id",
    "write_run_record",
]
