# Output Contract

This is the predictable, versioned contract for the artifacts the pipeline
writes. It complements `docs/data_dictionary.md` (input schemas) and
`docs/metric_dictionary.md` (what each metric means). It is the analogue of an
"extraction contract": *what the tool guarantees about its outputs*.

## Schema version

JSON outputs carry a top-level `schema_version` (currently `"1.0"`). Bump it on
any breaking change to the JSON/Excel/HTML output structure, and note the change
in `CHANGELOG.md`. The constant lives at
`etf_portfolio.tracking.OUTPUT_SCHEMA_VERSION`.

## Artifacts

| Artifact | Path | Description |
| --- | --- | --- |
| Backtest metrics | `reports/metrics/backtest_metrics.json` | `schema_version`, `run_id`, `provenance_status`, `optimized_strategy` (metric → value), `benchmarks` (name → metrics), `implementation_gap` (`rebalance_count`, `mean_one_way_gap`, `max_one_way_gap`, `latest_one_way_gap`, `rebalances_with_constraint_violation`), `sharpe_difference_vs_benchmarks` (benchmark name → `sharpe_difference`, `ci_lower`, `ci_upper`, `probability_not_positive`, `confidence_level`, `n_observations`, `n_resamples`, `mean_block_length`, `seed`), `run_record` (relative path) |
| Sensitivity grid | `reports/metrics/sensitivity.json` | Written only by the `sensitivity` command: `schema_version`, `run_id`, `optimization_method`, `benchmark`, `common_start_date`, `end_date`, `configured`, `summary` (`run_count`, `ok_count`, `sharpe_min`, `sharpe_max`, `share_of_runs_beating_benchmark`), `runs` (one row per lookback × risk-free rate × rebalance mode) |
| Validation summary | `reports/metrics/validation_summary.json` | `schema_version`, `missing_data_fraction`, `history_coverage`, `suspicious_jump_count` |
| Run record | `reports/runs/<stage>_<run_id>.json` | `schema_version`, `pipeline_steps`, `git_commit_hash`, `config_hash`, `universe_id`, `data_version`, `output_artifacts` (name → {path, exists, sha256}), method metadata; backtest records also carry `lookback_periods` (observations actually used), `configured_lookback_periods`, and `configured_backtest_start_date` |
| Pipeline state | `reports/runs/pipeline_state.json` | Per-stage `status`, `input_hash`, `outputs` (path → sha256), timestamps; drives `--resume` |
| Errors report | `reports/runs/<run_id>/errors.json` | Failed/interrupted/blocked stages with `error_code` and `reason` (written only on failure/interruption) |
| Run log | `reports/runs/<run_id>/run.log` | Per-run structured JSON log (for `run-all` or when `--log-file` is given) |
| HTML report | `reports/html/latest_report.html` | Self-contained narrative report (reader guide, metric dictionary, trust & safety, numbered tables and figures, assumptions). `reports/html/backtest_report.html` is a byte-identical legacy copy |
| Frontier explorer | `reports/html/frontier.html` | Self-contained interactive efficient-frontier page from the `optimize` stage |
| Excel workbook | `reports/excel/portfolio_results.xlsx` | Metrics, weights history, exposures, attribution, assumptions sheets |
| Figures | `reports/figures/*.png` | Static exports of the report's figures, 2800x1680 px, same palette and type as the HTML |

All final files are written **atomically** (temp file + `os.replace`); a reader
never observes a partially written artifact.

## Determinism

Running the same command on the same input data with the same code version (and
the same seed, `random_state=42` for ML) produces **identical numeric outputs**,
except for intentionally varying fields: `timestamp_utc`, `run_id`, and paths
derived from the run id. Non-finite floats are sanitized out of JSON
(`allow_nan=false`). This determinism is what makes the golden-file tests
(`tests/golden/`) and skip-unchanged resume meaningful.

## Provenance

Every metrics file and run record records whether the run was `tracked` (a real
git commit was present) or `untracked_preview`. Run records hash inputs and
outputs (SHA-256) and capture the config hash and `universe_id`, so any Markdown
or downstream chunk derived from these outputs can be traced back to the exact
inputs, code, and configuration that produced it. See `docs/engineering_standards.md` §S.
