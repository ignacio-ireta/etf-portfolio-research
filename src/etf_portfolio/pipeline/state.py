"""Persistent pipeline state for resumable, skip-unchanged runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from etf_portfolio.io_utils import atomic_write_text

STATE_SCHEMA_VERSION = "1.0"


class StageStatus(StrEnum):
    """Lifecycle status of a single pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"
    INTERRUPTED = "interrupted"
    BLOCKED = "blocked"


@dataclass
class StageRecord:
    """Recorded execution state for one stage."""

    name: str
    status: StageStatus = StageStatus.PENDING
    input_hash: str | None = None
    outputs: dict[str, str | None] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    error_code: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageRecord:
        return cls(
            name=data["name"],
            status=StageStatus(data.get("status", "pending")),
            input_hash=data.get("input_hash"),
            outputs=dict(data.get("outputs") or {}),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            error_code=data.get("error_code"),
            error=data.get("error"),
        )


@dataclass
class PipelineState:
    """The full set of stage records for one pipeline, persisted between runs."""

    run_id: str
    schema_version: str = STATE_SCHEMA_VERSION
    stages: dict[str, StageRecord] = field(default_factory=dict)

    def record(self, name: str) -> StageRecord:
        """Return (creating if needed) the mutable record for ``name``."""

        return self.stages.setdefault(name, StageRecord(name=name))

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for stage_record in self.stages.values():
            key = stage_record.status.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "stages": {name: record.to_dict() for name, record in self.stages.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineState:
        stages = {
            name: StageRecord.from_dict(record)
            for name, record in (data.get("stages") or {}).items()
        }
        return cls(
            run_id=data.get("run_id", ""),
            schema_version=data.get("schema_version", STATE_SCHEMA_VERSION),
            stages=stages,
        )


def load_state(path: Path) -> PipelineState | None:
    """Load a prior pipeline state, or ``None`` if it is missing/unreadable."""

    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return PipelineState.from_dict(data)


def save_state(state: PipelineState, path: Path) -> Path:
    """Atomically persist the pipeline state."""

    atomic_write_text(path, json.dumps(state.to_dict(), indent=2, sort_keys=True))
    return path


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "STATE_SCHEMA_VERSION",
    "PipelineState",
    "StageRecord",
    "StageStatus",
    "load_state",
    "save_state",
    "utc_now_iso",
]
