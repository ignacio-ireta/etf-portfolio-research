# Research Log

Every material experiment should be recorded here. The purpose is to prevent undocumented parameter drift and post-hoc storytelling.

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
