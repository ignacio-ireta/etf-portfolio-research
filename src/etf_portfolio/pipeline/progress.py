"""Human-readable console progress and end-of-run summary.

Structured JSON logs go to stderr (see logging_config); this module writes a
compact, scannable summary to stdout so a person watching the run can see what
happened without parsing JSON.
"""

from __future__ import annotations

from etf_portfolio.pipeline.state import PipelineState


def format_summary(state: PipelineState, *, order: list[str]) -> str:
    """Render a per-stage summary block for a completed (or stopped) run."""

    counts = state.counts()
    header = (
        f"SUCCESS {counts.get('success', 0)} / "
        f"WARNING {counts.get('warning', 0)} / "
        f"FAILED {counts.get('failed', 0)} / "
        f"SKIPPED {counts.get('skipped', 0)} / "
        f"INTERRUPTED {counts.get('interrupted', 0)} / "
        f"BLOCKED {counts.get('blocked', 0)}"
    )
    lines = [f"Pipeline run {state.run_id}: {header}"]
    for name in order:
        record = state.stages.get(name)
        if record is None:
            continue
        detail = f" — {record.error_code}: {record.error}" if record.error else ""
        lines.append(f"  [{record.status.value.upper():>11}] {name}{detail}")
    return "\n".join(lines)


def interruption_message(state: PipelineState, *, resume_hint: str) -> str:
    """Render the message shown when a run is interrupted between stages."""

    counts = state.counts()
    return (
        "Interrupted by user.\n"
        f"  Succeeded: {counts.get('success', 0)}\n"
        f"  Skipped:   {counts.get('skipped', 0)}\n"
        f"  Failed:    {counts.get('failed', 0)}\n"
        f"  Interrupted/blocked: {counts.get('interrupted', 0) + counts.get('blocked', 0)}\n"
        f"Resume with: {resume_hint}"
    )


__all__ = ["format_summary", "interruption_message"]
