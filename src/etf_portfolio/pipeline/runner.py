"""The pipeline runner: deterministic stage orchestration with resume support."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import signal
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

from etf_portfolio.config import AppConfig
from etf_portfolio.errors import PipelineInterrupted, error_code_for
from etf_portfolio.io_utils import atomic_write_text
from etf_portfolio.logging_config import get_logger, log_event
from etf_portfolio.pipeline import progress
from etf_portfolio.pipeline.state import (
    PipelineState,
    StageStatus,
    load_state,
    save_state,
    utc_now_iso,
)
from etf_portfolio.tracking import file_sha256

LOGGER = get_logger(__name__)


@dataclass
class Stage:
    """A single declarative pipeline stage.

    ``run`` performs the work (its return value is ignored; outputs are declared
    so the runner can hash and verify them). ``config_sections`` lists the
    top-level config keys the stage depends on, so editing an unrelated section
    does not invalidate it. ``inputs``/``outputs`` are the files the stage reads
    and writes; their SHA-256 hashes drive skip-unchanged and the cascade.
    """

    name: str
    run: Callable[[], Any]
    config_sections: tuple[str, ...] = ()
    inputs: tuple[Path, ...] = ()
    outputs: tuple[Path, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)


def _stage_input_hash(stage: Stage, config: AppConfig) -> str:
    config_dump = config.model_dump(mode="json")
    payload = {
        "config": {key: config_dump.get(key) for key in stage.config_sections},
        "params": stage.params,
        "inputs": {str(path): file_sha256(path) for path in stage.inputs},
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _outputs_intact(stage: Stage, recorded: dict[str, str | None]) -> bool:
    for path in stage.outputs:
        key = str(path)
        if key not in recorded or not path.exists():
            return False
        if recorded[key] != file_sha256(path):
            return False
    return True


class _InterruptGuard:
    """Install SIGINT/SIGTERM handlers that request a graceful stop between stages."""

    def __init__(self) -> None:
        self.stop_requested = False
        self._previous: list[tuple[int, Any]] = []

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        self.stop_requested = True

    def __enter__(self) -> _InterruptGuard:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous.append((sig, signal.getsignal(sig)))
                signal.signal(sig, self._handle)
            except (ValueError, OSError):
                # signal.signal only works in the main thread; degrade gracefully.
                pass
        return self

    def __exit__(self, *exc_info: object) -> None:
        for sig, handler in self._previous:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, handler)


def run_pipeline(
    stages: list[Stage],
    *,
    config: AppConfig,
    project_root: Path,
    run_id: str,
    state_path: Path,
    resume: bool = False,
    fail_fast: bool = True,
    resume_hint: str | None = None,
) -> PipelineState:
    """Run ``stages`` in order, persisting state and honoring resume/interruption.

    Raises the first stage exception (fail-fast) or an aggregate after a
    continue run, or ``PipelineInterrupted`` if stopped by a signal. Returns the
    final :class:`PipelineState` on success.
    """

    order = [stage.name for stage in stages]
    prior = load_state(state_path) if resume else None
    state = PipelineState(run_id=run_id)
    errors: list[BaseException] = []

    # Map each declared output file to its producing stage so a stage whose
    # upstream producer failed (or was itself blocked) this run can be blocked
    # rather than run. Without this, --continue (fail_fast=False) would run
    # dependent stages against stale/missing artifacts and could emit a
    # fresh-looking report from old prices/returns. Stages with no intra-pipeline
    # producer are genuinely independent and still run.
    produced_by = {str(path): stage.name for stage in stages for path in stage.outputs}
    tainted: set[str] = set()

    with _InterruptGuard() as guard:
        for stage in stages:
            record = state.record(stage.name)

            if guard.stop_requested:
                record.status = StageStatus.INTERRUPTED
                save_state(state, state_path)
                continue

            input_hash = _stage_input_hash(stage, config)
            record.input_hash = input_hash

            if resume and prior is not None:
                prior_record = prior.stages.get(stage.name)
                if (
                    prior_record is not None
                    # A prior SKIPPED stage was itself an unchanged success carried
                    # forward (with input_hash/outputs intact), so it must remain
                    # skippable on the next resume — otherwise resume skips only once.
                    and prior_record.status in (StageStatus.SUCCESS, StageStatus.SKIPPED)
                    and prior_record.input_hash == input_hash
                    and _outputs_intact(stage, prior_record.outputs)
                ):
                    record.status = StageStatus.SKIPPED
                    record.outputs = dict(prior_record.outputs)
                    record.started_at = prior_record.started_at
                    record.finished_at = prior_record.finished_at
                    log_event(
                        LOGGER,
                        logging.INFO,
                        "pipeline_stage_skipped",
                        run_id=run_id,
                        stage=stage.name,
                        reason="inputs_unchanged",
                    )
                    save_state(state, state_path)
                    continue

            blocked_by = {
                produced_by[str(path)] for path in stage.inputs if str(path) in produced_by
            } & tainted
            if blocked_by:
                record.status = StageStatus.BLOCKED
                tainted.add(stage.name)
                save_state(state, state_path)
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "pipeline_stage_blocked",
                    run_id=run_id,
                    stage=stage.name,
                    reason="upstream_failed",
                    upstream=sorted(blocked_by),
                )
                continue

            record.status = StageStatus.RUNNING
            record.started_at = utc_now_iso()
            save_state(state, state_path)

            try:
                stage.run()
            except BaseException as exc:
                record.status = StageStatus.FAILED
                record.finished_at = utc_now_iso()
                record.error_code = error_code_for(exc)
                record.error = str(exc)
                tainted.add(stage.name)
                save_state(state, state_path)
                log_event(
                    LOGGER,
                    logging.ERROR,
                    "pipeline_stage_failed",
                    run_id=run_id,
                    stage=stage.name,
                    error_type=type(exc).__name__,
                    error_code=record.error_code,
                    reason=str(exc),
                )
                errors.append(exc)
                if fail_fast or isinstance(exc, KeyboardInterrupt):
                    break
                continue

            record.status = StageStatus.SUCCESS
            record.finished_at = utc_now_iso()
            record.outputs = {str(path): file_sha256(path) for path in stage.outputs}
            save_state(state, state_path)

    # Any stage never reached is blocked (earlier failure) or interrupted (signal).
    for stage in stages:
        record = state.record(stage.name)
        if record.status == StageStatus.PENDING:
            record.status = StageStatus.INTERRUPTED if guard.stop_requested else StageStatus.BLOCKED
    save_state(state, state_path)

    hint = resume_hint or "etf-portfolio run-all --config <config> --resume"
    if guard.stop_requested:
        _write_errors(state, project_root, run_id)
        print(progress.interruption_message(state, resume_hint=hint))
        raise PipelineInterrupted(f"Pipeline interrupted between stages. Resume with: {hint}")

    print(progress.format_summary(state, order=order))

    if errors:
        _write_errors(state, project_root, run_id)
        raise errors[0]
    return state


def _write_errors(state: PipelineState, project_root: Path, run_id: str) -> None:
    failed = [
        {
            "stage": record.name,
            "status": record.status.value,
            "error_code": record.error_code,
            "reason": record.error,
        }
        for record in state.stages.values()
        if record.status in (StageStatus.FAILED, StageStatus.INTERRUPTED, StageStatus.BLOCKED)
    ]
    if not failed:
        return
    path = project_root / "reports/runs" / run_id / "errors.json"
    atomic_write_text(
        path,
        json.dumps({"run_id": run_id, "errors": failed}, indent=2, sort_keys=True),
    )


__all__ = ["Stage", "run_pipeline"]
