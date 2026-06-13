"""Integration tests for resume semantics across the real pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from etf_portfolio import cli
from etf_portfolio.pipeline.state import StageStatus

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _state(project_root: Path) -> dict:
    return json.loads(
        (project_root / "reports/runs/pipeline_state.json").read_text(encoding="utf-8")
    )["stages"]


def test_resume_reruns_only_backtest_when_lookback_changes(tiny_project: Path) -> None:
    assert cli.main(["run-all", "--config", "configs/base.yaml", "--lookback-periods", "5"]) == 0

    # Resume with a different backtest-only parameter: only backtest is invalidated;
    # ingest/validate/features/optimize have unchanged inputs and are skipped.
    assert (
        cli.main(
            ["run-all", "--config", "configs/base.yaml", "--lookback-periods", "6", "--resume"]
        )
        == 0
    )

    stages = _state(tiny_project)
    assert stages["backtest"]["status"] == StageStatus.SUCCESS.value
    for upstream in ("ingest", "validate", "features", "optimize"):
        assert stages[upstream]["status"] == StageStatus.SKIPPED.value
