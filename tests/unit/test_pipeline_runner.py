"""Tests for the pipeline runner: ordering, resume, skip-unchanged, interruption."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import pytest

from etf_portfolio.config import AppConfig
from etf_portfolio.errors import InfeasibleConstraintsError, PipelineInterrupted
from etf_portfolio.io_utils import atomic_write_text
from etf_portfolio.pipeline.runner import Stage, run_pipeline
from etf_portfolio.pipeline.state import StageStatus, load_state


def _state_path(tmp_path: Path) -> Path:
    return tmp_path / "reports/runs/pipeline_state.json"


def test_runs_stages_in_order_and_records_success(tmp_path: Path, tiny_config: AppConfig) -> None:
    calls: list[str] = []
    out_a = tmp_path / "a.txt"
    out_b = tmp_path / "b.txt"
    stages = [
        Stage("a", lambda: (calls.append("a"), atomic_write_text(out_a, "A"))[0], outputs=(out_a,)),
        Stage("b", lambda: (calls.append("b"), atomic_write_text(out_b, "B"))[0], outputs=(out_b,)),
    ]
    state = run_pipeline(
        stages,
        config=tiny_config,
        project_root=tmp_path,
        run_id="r1",
        state_path=_state_path(tmp_path),
    )
    assert calls == ["a", "b"]
    assert state.stages["a"].status == StageStatus.SUCCESS
    assert state.stages["b"].status == StageStatus.SUCCESS
    assert state.stages["a"].outputs[str(out_a)] is not None


def test_resume_skips_unchanged_inputs(tmp_path: Path, tiny_config: AppConfig) -> None:
    src = tmp_path / "src.txt"
    src.write_text("v1", encoding="utf-8")
    out = tmp_path / "out.txt"
    calls: list[str] = []

    def build() -> list[Stage]:
        return [
            Stage(
                "x",
                lambda: (calls.append("x"), atomic_write_text(out, src.read_text()))[0],
                inputs=(src,),
                outputs=(out,),
            )
        ]

    state_path = _state_path(tmp_path)
    run_pipeline(
        build(), config=tiny_config, project_root=tmp_path, run_id="1", state_path=state_path
    )
    assert calls == ["x"]

    # Resume with nothing changed -> skipped.
    state = run_pipeline(
        build(),
        config=tiny_config,
        project_root=tmp_path,
        run_id="2",
        state_path=state_path,
        resume=True,
    )
    assert calls == ["x"]
    assert state.stages["x"].status == StageStatus.SKIPPED

    # Change the input file -> stage re-runs.
    src.write_text("v2", encoding="utf-8")
    state = run_pipeline(
        build(),
        config=tiny_config,
        project_root=tmp_path,
        run_id="3",
        state_path=state_path,
        resume=True,
    )
    assert calls == ["x", "x"]
    assert state.stages["x"].status == StageStatus.SUCCESS
    assert out.read_text(encoding="utf-8") == "v2"


def test_resume_skips_repeatedly_on_unchanged_inputs(
    tmp_path: Path, tiny_config: AppConfig
) -> None:
    """A stage skipped on one resume must stay skippable on the next.

    Regression: the skip gate previously required a prior ``SUCCESS``, so a stage
    persisted as ``SKIPPED`` re-executed on the following resume (skip-once bug).
    """

    src = tmp_path / "src.txt"
    src.write_text("v1", encoding="utf-8")
    out = tmp_path / "out.txt"
    calls: list[str] = []

    def build() -> list[Stage]:
        return [
            Stage(
                "x",
                lambda: (calls.append("x"), atomic_write_text(out, src.read_text()))[0],
                inputs=(src,),
                outputs=(out,),
            )
        ]

    state_path = _state_path(tmp_path)
    run_pipeline(
        build(), config=tiny_config, project_root=tmp_path, run_id="1", state_path=state_path
    )
    second = run_pipeline(
        build(),
        config=tiny_config,
        project_root=tmp_path,
        run_id="2",
        state_path=state_path,
        resume=True,
    )
    assert second.stages["x"].status == StageStatus.SKIPPED

    # Third resume with still-unchanged inputs must skip again, not re-execute.
    third = run_pipeline(
        build(),
        config=tiny_config,
        project_root=tmp_path,
        run_id="3",
        state_path=state_path,
        resume=True,
    )
    assert third.stages["x"].status == StageStatus.SKIPPED
    assert calls == ["x"]  # executed exactly once across three runs


def test_only_relevant_config_section_invalidates(tmp_path: Path, tiny_config: AppConfig) -> None:
    out = tmp_path / "out.txt"
    calls: list[str] = []

    def build() -> list[Stage]:
        return [
            Stage(
                "opt",
                lambda: (calls.append("opt"), atomic_write_text(out, "x"))[0],
                config_sections=("optimization",),
                outputs=(out,),
            )
        ]

    state_path = _state_path(tmp_path)
    run_pipeline(
        build(), config=tiny_config, project_root=tmp_path, run_id="1", state_path=state_path
    )
    assert calls == ["opt"]

    # Changing an UNRELATED section (universe) does not invalidate the stage.
    other_section = tiny_config.universe.model_copy(update={"tickers": ["VTI", "BND"]})
    unrelated = tiny_config.model_copy(update={"universe": other_section}, deep=True)
    run_pipeline(
        build(),
        config=unrelated,
        project_root=tmp_path,
        run_id="2",
        state_path=state_path,
        resume=True,
    )
    assert calls == ["opt"]  # still skipped

    # Changing the RELEVANT section (optimization) forces a re-run.
    opt_section = tiny_config.optimization.model_copy(update={"default_max_weight_per_etf": 0.6})
    related = tiny_config.model_copy(update={"optimization": opt_section}, deep=True)
    run_pipeline(
        build(),
        config=related,
        project_root=tmp_path,
        run_id="3",
        state_path=state_path,
        resume=True,
    )
    assert calls == ["opt", "opt"]


def test_fail_fast_stops_and_writes_errors_json(tmp_path: Path, tiny_config: AppConfig) -> None:
    calls: list[str] = []

    def boom() -> None:
        calls.append("a")
        raise InfeasibleConstraintsError("no feasible portfolio")

    stages = [
        Stage("a", boom),
        Stage("b", lambda: calls.append("b")),
    ]
    with pytest.raises(InfeasibleConstraintsError):
        run_pipeline(
            stages,
            config=tiny_config,
            project_root=tmp_path,
            run_id="run-x",
            state_path=_state_path(tmp_path),
        )
    assert calls == ["a"]  # b never runs under fail-fast

    state = load_state(_state_path(tmp_path))
    assert state is not None
    assert state.stages["a"].status == StageStatus.FAILED
    assert state.stages["a"].error_code == "infeasible_constraints"
    assert state.stages["b"].status == StageStatus.BLOCKED

    errors_path = tmp_path / "reports/runs/run-x/errors.json"
    payload = json.loads(errors_path.read_text(encoding="utf-8"))
    codes = {entry["error_code"] for entry in payload["errors"]}
    assert "infeasible_constraints" in codes


def test_continue_mode_attempts_independent_stages(tmp_path: Path, tiny_config: AppConfig) -> None:
    calls: list[str] = []

    def boom() -> None:
        calls.append("a")
        raise InfeasibleConstraintsError("fail")

    stages = [
        Stage("a", boom),
        Stage("b", lambda: calls.append("b")),
    ]
    with pytest.raises(InfeasibleConstraintsError):
        run_pipeline(
            stages,
            config=tiny_config,
            project_root=tmp_path,
            run_id="run-c",
            state_path=_state_path(tmp_path),
            fail_fast=False,
        )
    # In continue mode the independent stage still runs even though 'a' failed.
    assert calls == ["a", "b"]


def test_continue_mode_blocks_dependent_stages_cascade(
    tmp_path: Path, tiny_config: AppConfig
) -> None:
    """--continue must not run stages downstream of a failure against stale data.

    'a' (produces x) fails; 'b' consumes x and 'c' consumes b's output y, so both
    are blocked (cascade). 'd' is independent and still runs.
    """

    calls: list[str] = []
    x = tmp_path / "x.txt"
    y = tmp_path / "y.txt"

    def boom() -> None:
        calls.append("a")
        raise InfeasibleConstraintsError("fail")

    stages = [
        Stage("a", boom, outputs=(x,)),
        Stage("b", lambda: calls.append("b"), inputs=(x,), outputs=(y,)),
        Stage("c", lambda: calls.append("c"), inputs=(y,)),
        Stage("d", lambda: calls.append("d")),
    ]
    with pytest.raises(InfeasibleConstraintsError):
        run_pipeline(
            stages,
            config=tiny_config,
            project_root=tmp_path,
            run_id="run-dep",
            state_path=_state_path(tmp_path),
            fail_fast=False,
        )

    assert calls == ["a", "d"]  # 'b' and 'c' are blocked, not run
    state = load_state(_state_path(tmp_path))
    assert state is not None
    assert state.stages["a"].status == StageStatus.FAILED
    assert state.stages["b"].status == StageStatus.BLOCKED
    assert state.stages["c"].status == StageStatus.BLOCKED
    assert state.stages["d"].status == StageStatus.SUCCESS


def test_interrupt_between_stages_marks_interrupted_and_raises(
    tmp_path: Path, tiny_config: AppConfig
) -> None:
    calls: list[str] = []

    def request_stop() -> None:
        calls.append("a")
        # The runner's handler catches SIGINT (sets a flag), it does not raise.
        os.kill(os.getpid(), signal.SIGINT)

    stages = [
        Stage("a", request_stop),
        Stage("b", lambda: calls.append("b")),
    ]
    with pytest.raises(PipelineInterrupted):
        run_pipeline(
            stages,
            config=tiny_config,
            project_root=tmp_path,
            run_id="run-i",
            state_path=_state_path(tmp_path),
        )
    assert calls == ["a"]  # 'b' never starts; stop was requested between stages
    state = load_state(_state_path(tmp_path))
    assert state is not None
    assert state.stages["a"].status == StageStatus.SUCCESS
    assert state.stages["b"].status == StageStatus.INTERRUPTED
