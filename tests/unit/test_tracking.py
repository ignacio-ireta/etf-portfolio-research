from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from etf_portfolio.config import load_config
from etf_portfolio.tracking import build_run_record, relative_to_project_root, write_run_record


def test_build_run_record_requires_git_commit(tmp_path: Path) -> None:
    artifact_path = tmp_path / "reports/metrics.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="real git commit"):
        build_run_record(
            stage="backtest",
            run_id="backtest-test",
            config=load_config("configs/base.yaml"),
            project_root=tmp_path,
            data_version_path=None,
            output_artifacts={"metrics": artifact_path},
        )


def test_build_run_record_marks_untracked_preview_without_git(tmp_path: Path) -> None:
    artifact_path = tmp_path / "reports/metrics.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("{}", encoding="utf-8")
    config = load_config("configs/base.yaml")
    preview_config = config.model_copy(
        update={
            "tracking": config.tracking.model_copy(update={"require_git_commit": False}),
        }
    )

    record = build_run_record(
        stage="backtest",
        run_id="backtest-preview",
        config=preview_config,
        project_root=tmp_path,
        data_version_path=None,
        output_artifacts={"metrics": artifact_path},
    )

    assert record["git_commit_hash"] is None
    assert record["provenance_status"] == "untracked_preview"


def test_build_run_record_uses_commit_hash_and_relative_artifact_paths(tmp_path: Path) -> None:
    commit_hash = _initialize_git_repo(tmp_path)
    artifact_path = tmp_path / "reports/metrics.json"
    data_path = tmp_path / "data/processed/returns.parquet"
    artifact_path.parent.mkdir(parents=True)
    data_path.parent.mkdir(parents=True)
    artifact_path.write_text("{}", encoding="utf-8")
    data_path.write_text("returns", encoding="utf-8")

    record = build_run_record(
        stage="backtest",
        run_id="backtest-test",
        config=load_config("configs/base.yaml"),
        project_root=tmp_path,
        data_version_path=data_path,
        output_artifacts={"metrics": artifact_path},
    )

    assert record["git_commit_hash"] == commit_hash
    assert record["provenance_status"] == "tracked"
    assert record["data_version"]["path"] == "data/processed/returns.parquet"
    assert record["output_artifacts"]["metrics"]["path"] == "reports/metrics.json"
    assert not Path(record["output_artifacts"]["metrics"]["path"]).is_absolute()


def test_write_run_record_sanitizes_nonfinite_values_as_strict_json(tmp_path: Path) -> None:
    record = {
        "stage": "backtest",
        "run_id": "strict-json",
        "backtest_metrics": {
            "Sharpe Ratio": float("nan"),
            "Sortino Ratio": float("inf"),
            "Calmar Ratio": float("-inf"),
        },
    }

    output_path = write_run_record(record, artifact_dir=tmp_path)

    raw_text = output_path.read_text(encoding="utf-8")
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text
    parsed = json.loads(raw_text)
    assert parsed["backtest_metrics"] == {
        "Calmar Ratio": None,
        "Sharpe Ratio": None,
        "Sortino Ratio": None,
    }


def test_relative_to_project_root_rejects_external_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside project_root"):
        relative_to_project_root(tmp_path, tmp_path.parent / "outside.json")


def _initialize_git_repo(project_root: Path) -> str:
    subprocess.run(["git", "init"], cwd=project_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"],
        cwd=project_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test Runner"],
        cwd=project_root,
        check=True,
    )
    (project_root / "README.md").write_text("test repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=project_root, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=project_root, check=True)
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
