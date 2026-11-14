# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Audit remediation

Changes from the repository audit (step-by-step plans in `plans/`). Backtest numbers
change because of the first two items; regenerate reports before quoting any figure.

- **Changed:** the estimation lookback is now `backtest.lookback_periods` in the config
  (default 756) and is stored in every backtest run record; `--lookback-periods` is an
  optional override. Returns before `backtest.start_date` are used to warm up the
  lookback instead of being discarded.
- **Fixed:** daily returns between rebalance dates are computed as buy-and-hold,
  consistent with the weight drift, instead of as a costless daily reset.
- **Fixed:** the optimizer retries SLSQP on a tolerance ladder, so one non-converged
  month no longer aborts a whole backtest (`optimizer_tolerance_relaxed` warning).
- **Changed:** fixed-allocation benchmarks (60/40 and other `benchmark.secondary`
  entries) are simulated on the strategy's rebalance schedule and pay the same cost
  rate, instead of being rebalanced daily for free.
- **Added:** `implementation_gap` in the metrics JSON and report assumptions, plus the
  benchmark row `Optimizer Target (Full Rebalance)`.
- **Added:** `sharpe_difference_vs_benchmarks` (stationary-bootstrap confidence
  intervals) in the metrics JSON and a `Statistical uncertainty:` report assumption.
- **Added:** `etf-portfolio sensitivity` command writing `reports/metrics/sensitivity.json`.
- **Added:** `scripts/append_research_log_entry.py` to build research-log entries from
  the pipeline's own outputs.
- **Removed:** unused `pyportfolioopt` dependency.
- **Changed:** `investor_profile` is optional and documented as descriptive only.

Brought the project into compliance with the adapted engineering standards
(`docs/engineering_standards.md`).

### Added
- Domain error taxonomy (`errors.py`) with stable codes and meaningful CLI exit
  codes (config=2, data=3, infeasible/insufficient=4, provenance=5, interrupted=130).
- Resumable pipeline runner (`pipeline/` package): per-stage state manifest
  (`reports/runs/pipeline_state.json`), `--resume` with skip-unchanged-by-input-hash,
  graceful SIGINT/SIGTERM handling between stages, and a console summary.
- Atomic output writes (`io_utils.py`) for all generated artifacts (temp + `os.replace`).
- CLI flags: `--version`, `--log-level`, `--log-file`, `--resume`,
  `--fail-fast`/`--continue`; per-run log file for `run-all`.
- `schema_version` and `pipeline_steps` on run records and JSON outputs;
  `docs/output_contract.md`.
- Per-run `errors.json` for failed/interrupted runs.
- Test breadth: `tests/integration/`, `tests/e2e/`, `tests/golden/`, property/fuzz
  tests, a shared `tests/conftest.py`, and pytest markers.
- Tooling: mypy (lenient), `pytest-cov`, `pip-audit`, `pre-commit`, split PEP 735
  dependency groups, and an optional `ml` extra.
- CI: advisory mypy/coverage/pip-audit steps + coverage artifact; Dependabot config.
- Governance docs: `docs/engineering_standards.md`, `docs/testing.md`,
  `docs/troubleshooting.md`, `LICENSE` (MIT), `CONTRIBUTING.md`, `SECURITY.md`,
  populated `.env.example`.

### Changed
- Report presentation redesigned as a printed research note (no numbers, metrics,
  file names, or JSON keys changed):
  - `latest_report.html` drops the stacked cards for a masthead with real run facts
    (run id, backtest window, rebalances, universe, headline metrics), numbered
    sections with margin headings, numbered and captioned figures, and ruled tables
    with tabular monospace numerals. System font stacks only; still self-contained.
  - One figure language for the interactive charts and the PNG exports: the
    optimized strategy is the single warm accent, benchmarks keep fixed cool hues,
    gridlines are hairlines, legends are unboxed, key series are labelled directly,
    and hover shows formatted values per date.
  - Benchmark comparison is now four small multiples (one scale per metric) instead
    of one grouped bar chart mixing ratios and percentages; asset-class and region
    exposure are sorted labelled bars instead of donuts; the weighted expense ratio
    is drawn as a step line.
  - `frontier.html` is rendered with the same layout instead of a bare Plotly page.
  - PNG exports are rendered at 1000x600 CSS px, scale 2.8 (still 2800x1680 px) so
    type is legible at gallery size.
  - New `reporting/layout.py`; `plots.py` splits `build_*` (compute) from `render_*`
    (style). `build_group_exposure_pie_figure` was removed in favour of
    `build_group_exposure_figure`.
- The committed `reports/html/*.html` and `reports/figures/*.png` were re-rendered
  from the series and table cells stored in the previous report
  (`scripts/restyle_committed_reports.py`), not by re-running the pipeline, so every
  published number is unchanged. Consequence: the `sha256` values recorded for those
  files in `reports/runs/backtest_run-all-20260425T083756Z-f6f9560b.json` describe
  the pre-restyle renderings.
- Data validation and provenance failures now raise typed domain exceptions
  (still subclassing `ValueError`/`RuntimeError` for backward compatibility).
- Package version is dynamic, sourced from `etf_portfolio.__version__`.
- Expanded ruff rule set (bugbear, comprehensions, simplify, ruff-specific).

## [0.1.0]

- Initial reproducible ETF portfolio research pipeline: universe definition, data
  ingestion + validation, returns, optimization, walk-forward backtesting,
  reporting, run records, and an optional ML track.
