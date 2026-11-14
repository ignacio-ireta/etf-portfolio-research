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
