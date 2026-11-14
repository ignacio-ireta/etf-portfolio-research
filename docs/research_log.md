# Research Log

Every material experiment should be recorded here. The purpose is to prevent undocumented parameter drift and post-hoc storytelling.

## How To Add An Entry

Do not type results by hand. After a backtest (and, ideally, a sensitivity run), let the
script build the entry from the files the pipeline wrote:

```bash
uv run etf-portfolio backtest --config configs/base.yaml
uv run etf-portfolio sensitivity --config configs/base.yaml
uv run python scripts/append_research_log_entry.py --hypothesis "One sentence: what are you testing?"
```

Write the hypothesis **before** looking at the result. The script appends the entry to the
end of this file and refuses to add a second entry for the same run ID.

`--decision auto` (the default) uses a fixed rule on the 95% bootstrap interval of the
Sharpe difference versus the primary benchmark: interval entirely above zero gives
`adopt`, entirely below zero gives `reject`, anything else gives `revisit later`. Pass
`--decision adopt`, `--decision reject` or `--decision "revisit later"` to override it,
and say why in a follow-up line.

## Template

### Date

`YYYY-MM-DD`

### Hypothesis

One sentence. What are you testing?

### Config

Reference the exact config path, overlay, or run ID.

### Result

Record the relevant metrics and artifacts.

### Decision

Choose one:

- adopt
- reject
- revisit later

## Entries

### Date

`2026-04-23`

### Hypothesis

The repository needs a minimum documentation set tied to the implemented pipeline so users can install, run, reproduce, and audit the research process without relying on notebook state.

### Config

- docs refresh against `configs/base.yaml`
- CLI surface `etf-portfolio {ingest,validate,features,optimize,backtest,report,run-all}`

### Result

- README updated with install, run, outputs, and assumptions
- methodology, data dictionary, runbook, and model card populated
- research log added

### Decision

adopt

### Date

`2026-11-14`

### Hypothesis

After the audit remediation (plans 01 to 09), the max-Sharpe contribution-only strategy has a risk-adjusted advantage over VT that is distinguishable from noise and that comes from the optimizer.

### Config

- run ID `backtest-20261114T031301Z-ee163e1c` (run record: `reports/runs/backtest_backtest-20261114T031301Z-ee163e1c.json`)
- config hash `551ea46127e7ba59ec6c5ebd3693d55245d1e428ca2ec9e8804304bc6c80ed76`, git commit `db4377eddc2aea6c206203a0b4f578bc6eae67ff`
- objective `max_sharpe`, lookback 756 observations, simulated window 2014-02-03 to 2026-04-24 (configured start 2012-01-01)

### Result

- Optimized Strategy: CAGR 10.36%, volatility 11.33%, Sharpe 0.66, max drawdown -24.98%
- Primary benchmark (VT): CAGR 10.71%, Sharpe 0.51
- Sharpe difference versus primary benchmark: +0.15, 95% bootstrap interval [-0.06, +0.36] (contains zero)
- Optimizer Target (Full Rebalance): Sharpe 0.60; strategy minus target: +0.06, 95% bootstrap interval [-0.06, +0.22] (contains zero)
- Implementation gap: mean 26.1%, max 46.9% (one-way); configured constraints broken at 146 of 147 rebalances
- Sensitivity grid (16 of 16 runs completed, common window from 2016-02-01): Sharpe from 0.67 to 1.04; 100% of runs above the primary benchmark
- Artifacts: `reports/metrics/backtest_metrics.json`, `reports/html/latest_report.html`, `reports/metrics/sensitivity.json`

### Decision

revisit later

Reason: The Sharpe difference versus the primary benchmark is not distinguishable from sampling noise. Do not describe this run as outperformance.
