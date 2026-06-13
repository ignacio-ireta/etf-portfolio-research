"""Deterministic pipeline orchestration: state, resume, skip-unchanged, interruption.

This package formalizes the ordered research pipeline (ingest -> validate ->
features -> optimize -> backtest) as a small set of declarative stages run by a
single runner. The runner persists a state manifest so a run can resume after a
crash or interruption, skip stages whose inputs are unchanged, and stop safely
between stages on SIGINT/SIGTERM. See docs/engineering_standards.md sections
F (architecture), J (resumability), and R (graceful interruption).
"""

from __future__ import annotations

from etf_portfolio.pipeline.runner import Stage, run_pipeline
from etf_portfolio.pipeline.state import (
    PipelineState,
    StageRecord,
    StageStatus,
    load_state,
    save_state,
)

__all__ = [
    "PipelineState",
    "Stage",
    "StageRecord",
    "StageStatus",
    "load_state",
    "run_pipeline",
    "save_state",
]
