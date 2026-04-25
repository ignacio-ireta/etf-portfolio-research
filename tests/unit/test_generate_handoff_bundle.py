import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "generate_handoff_bundle.py"
SPEC = importlib.util.spec_from_file_location("generate_handoff_bundle", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
handoff = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = handoff
SPEC.loader.exec_module(handoff)


def _write_file(path: Path, content: str = "x", *, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, (mtime, mtime))


def _write_required_logs(project_root: Path, *, mtime: float) -> None:
    for name in handoff.FRESHNESS_LOG_NAMES:
        _write_file(project_root / "handoff" / f"{name}.txt", "ok\n", mtime=mtime)


def test_verify_log_freshness_fails_when_source_file_is_newer_than_logs(
    tmp_path: Path,
) -> None:
    _write_file(tmp_path / "src" / "portfolio.py", "print('new')\n", mtime=200.0)
    _write_required_logs(tmp_path, mtime=100.0)

    with pytest.raises(RuntimeError, match="Validation logs are not fresh") as error:
        handoff._verify_log_freshness(tmp_path)

    message = str(error.value)
    assert "src/portfolio.py" in message
    assert "handoff/pytest.txt" in message
    assert "handoff/run_all.txt" in message


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("scripts/generate_handoff_bundle.py"),
        Path("pyproject.toml"),
    ],
)
def test_verify_log_freshness_checks_scripts_and_root_project_files(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    _write_file(tmp_path / relative_path, "new\n", mtime=200.0)
    _write_required_logs(tmp_path, mtime=100.0)

    with pytest.raises(RuntimeError) as error:
        handoff._verify_log_freshness(tmp_path)

    assert relative_path.as_posix() in str(error.value)


def test_verify_log_freshness_passes_when_logs_are_newer_than_inputs(tmp_path: Path) -> None:
    _write_file(tmp_path / "docs" / "runbook.md", "# Runbook\n", mtime=100.0)
    _write_required_logs(tmp_path, mtime=200.0)

    freshness = handoff._verify_log_freshness(tmp_path)

    assert freshness["newest_source_file"] == "docs/runbook.md"
    assert set(freshness["logs"]) == set(handoff.FRESHNESS_LOG_NAMES)


def test_verify_artifacts_requires_latest_run_record_to_match_run_all_log(
    tmp_path: Path,
) -> None:
    run_id = "run-all-20260425T000000Z-abcdef12"
    newer_run_id = "run-all-20260425T010000Z-deadbeef"
    run_record_path = tmp_path / "reports" / "runs" / f"backtest_{run_id}.json"
    metrics_path = tmp_path / "reports" / "metrics" / "backtest_metrics.json"
    run_all_log = tmp_path / "handoff" / "run_all.txt"

    _write_file(run_all_log, json.dumps({"run_id": run_id}) + "\n", mtime=100.0)
    _write_file(
        metrics_path,
        json.dumps(
            {
                "run_id": run_id,
                "run_record": run_record_path.relative_to(tmp_path).as_posix(),
            }
        ),
        mtime=100.0,
    )
    _write_file(
        run_record_path,
        json.dumps({"run_id": run_id, "output_artifacts": {}}),
        mtime=100.0,
    )
    _write_file(
        tmp_path / "reports" / "runs" / "manual-later.json",
        json.dumps({"run_id": newer_run_id, "output_artifacts": {}}),
        mtime=200.0,
    )

    with pytest.raises(RuntimeError, match="Latest run record by mtime"):
        handoff._verify_artifacts(tmp_path, run_all_log)


def test_write_preflight_summary_includes_required_provenance_fields(tmp_path: Path) -> None:
    log_path = tmp_path / "handoff" / "pytest.txt"
    _write_file(log_path, "ok\n", mtime=100.0)
    log = handoff.CommandLog(
        name="pytest",
        command=("pytest", "-q"),
        path=log_path,
        started_utc="2026-04-25T00:00:00Z",
        finished_utc="2026-04-25T00:00:01Z",
        exit_code=0,
    )

    summary_path = handoff.write_preflight_summary(
        tmp_path,
        logs={"pytest": log},
        freshness={
            "newest_source_file": "src/portfolio.py",
            "newest_source_timestamp": "2026-04-25T00:00:00Z",
        },
        artifacts={"run_id": "run-all-20260425T000000Z-abcdef12"},
        artifact_paths_included=["reports/metrics/backtest_metrics.json"],
        status="pass",
    )

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["git_commit"] is None
    assert payload["latest_run_id"] == "run-all-20260425T000000Z-abcdef12"
    assert payload["newest_source_timestamp"] == "2026-04-25T00:00:00Z"
    assert payload["artifact_paths_included"] == ["reports/metrics/backtest_metrics.json"]
    assert payload["status"] == "pass"
    assert payload["validation_log_timestamps"]["pytest"]["path"] == "handoff/pytest.txt"
