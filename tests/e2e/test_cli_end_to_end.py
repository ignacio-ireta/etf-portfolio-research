"""End-to-end CLI tests that drive `run-all` like a user would."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from etf_portfolio import cli
from etf_portfolio.pipeline.state import StageStatus

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

PIPELINE_STAGES = ["ingest", "validate", "features", "optimize", "backtest"]


def _load_state(project_root: Path) -> dict:
    return json.loads(
        (project_root / "reports/runs/pipeline_state.json").read_text(encoding="utf-8")
    )


def test_run_all_produces_artifacts_state_and_log(
    tiny_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["run-all", "--config", "configs/base.yaml", "--lookback-periods", "5"])
    assert exit_code == 0

    # Core artifacts exist.
    assert (tiny_project / "data/processed/returns.parquet").exists()
    assert (tiny_project / "reports/html/latest_report.html").exists()
    assert (tiny_project / "reports/metrics/backtest_metrics.json").exists()

    # Pipeline state records every stage as a success.
    state = _load_state(tiny_project)
    for stage in PIPELINE_STAGES:
        assert state["stages"][stage]["status"] == StageStatus.SUCCESS.value

    # A per-run log file was written for traceability.
    assert list((tiny_project / "reports/runs").glob("*/run.log"))

    # A human-readable summary went to stdout.
    assert "Pipeline run" in capsys.readouterr().out


def test_resume_skips_unchanged_stages(
    tiny_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["run-all", "--config", "configs/base.yaml", "--lookback-periods", "5"]) == 0
    returns_before = (tiny_project / "data/processed/returns.parquet").read_bytes()
    capsys.readouterr()  # clear

    # Re-running with --resume and no changes skips every stage.
    assert (
        cli.main(
            ["run-all", "--config", "configs/base.yaml", "--lookback-periods", "5", "--resume"]
        )
        == 0
    )
    state = _load_state(tiny_project)
    for stage in PIPELINE_STAGES:
        assert state["stages"][stage]["status"] == StageStatus.SKIPPED.value

    # Skipped stages leave outputs untouched.
    assert (tiny_project / "data/processed/returns.parquet").read_bytes() == returns_before
    assert "SKIPPED 5" in capsys.readouterr().out
