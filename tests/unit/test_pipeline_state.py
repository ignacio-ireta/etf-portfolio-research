"""Tests for pipeline state serialization."""

from __future__ import annotations

from pathlib import Path

from etf_portfolio.pipeline.state import (
    PipelineState,
    StageRecord,
    StageStatus,
    load_state,
    save_state,
)


def test_stage_record_roundtrip() -> None:
    record = StageRecord(
        name="ingest",
        status=StageStatus.SUCCESS,
        input_hash="abc",
        outputs={"data/raw/prices.parquet": "deadbeef"},
        started_at="2026-06-13T00:00:00+00:00",
        finished_at="2026-06-13T00:00:01+00:00",
    )
    restored = StageRecord.from_dict(record.to_dict())
    assert restored == record


def test_pipeline_state_save_and_load(tmp_path: Path) -> None:
    state = PipelineState(run_id="run-1")
    record = state.record("ingest")
    record.status = StageStatus.SUCCESS
    record.input_hash = "hash"
    path = tmp_path / "reports/runs/pipeline_state.json"

    save_state(state, path)
    loaded = load_state(path)

    assert loaded is not None
    assert loaded.run_id == "run-1"
    assert loaded.stages["ingest"].status == StageStatus.SUCCESS
    assert loaded.stages["ingest"].input_hash == "hash"


def test_load_state_returns_none_for_missing_or_corrupt(tmp_path: Path) -> None:
    assert load_state(tmp_path / "missing.json") is None
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert load_state(corrupt) is None


def test_counts_aggregates_statuses() -> None:
    state = PipelineState(run_id="r")
    state.record("a").status = StageStatus.SUCCESS
    state.record("b").status = StageStatus.SKIPPED
    state.record("c").status = StageStatus.SUCCESS
    counts = state.counts()
    assert counts["success"] == 2
    assert counts["skipped"] == 1
