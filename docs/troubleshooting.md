# Troubleshooting

Common failures, what they mean, and how to recover. The CLI uses meaningful
exit codes (see `docs/engineering_standards.md` §H):

| Exit code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Unexpected error |
| 2 | Configuration error |
| 3 | Data ingestion / validation error |
| 4 | Infeasible constraints / insufficient history |
| 5 | Provenance error (no git commit) |
| 130 | Interrupted (SIGINT/SIGTERM) |

Add `--log-level debug` to any command to see full tracebacks.

## "Run tracking requires a real git commit" (exit 5)

The pipeline records the git commit for provenance. Run from a git repository
with at least one commit, **or** set `tracking.require_git_commit: false` in the
config for an explicitly `untracked_preview` run.

## "Not enough return history" / "No backtest return history remains" (exit 4)

`--lookback-periods` is larger than the available history, or the backtest date
window (`backtest.start_date`/`end_date`) excludes too much. Lower the lookback,
widen the window, or ingest a longer history.

## Infeasible constraints (exit 4)

`optimization.default_max_weight_per_etf` and `constraints.ticker_bounds` cannot
sum to a fully-invested portfolio for the configured universe, or asset-class /
ticker bounds conflict. Loosen the caps or check the universe size.

## "ML is disabled in the current config" (exit 2)

The `ml` command requires `ml.enabled: true`. Enable it (and configure the `ml`
section) or omit the command. ML is off by default.

## Configuration errors (exit 2)

Pydantic validation rejected the YAML (unknown field, bad type, bounds that don't
sum, currency code length, etc.). The error names the offending field. Unknown
data providers and unsupported `active_objective` values also surface here.

## A figure is a blank placeholder

If headless image export (`kaleido`) is unavailable, the report writes a
placeholder PNG and logs `report_figure_export_fallback`. The HTML report still
renders interactive charts; only the static PNG export is affected.

## Interrupting and resuming a long run

Press Ctrl-C during `run-all`. The current stage finishes, remaining stages are
marked `interrupted`, a summary + resume hint print, and the process exits 130.
Continue where it left off:

```bash
uv run etf-portfolio run-all --config configs/base.yaml --resume
```

`--resume` also skips any stage whose inputs are unchanged since the last
successful run (see `reports/runs/pipeline_state.json`). To force a full re-run,
omit `--resume`.

## Inspecting what failed

A failed or interrupted `run-all` writes `reports/runs/<run_id>/errors.json` with
the stage, `error_code`, and reason. The per-run log is at
`reports/runs/<run_id>/run.log`.

## Tiingo provider errors

The Tiingo provider needs `TIINGO_API_KEY` in the environment (copy
`.env.example` to `.env`). The default provider (`yfinance`) needs no key.
