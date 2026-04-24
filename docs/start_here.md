# Start Here

This project is a reproducible ETF portfolio research tool. It helps compare ETF allocation strategies under explicit assumptions, then produces reports that show portfolio weights, historical backtest results, risk metrics, benchmark comparisons, and known limitations.

The goal is not to tell someone what to buy. The goal is to make portfolio research easier to inspect, reproduce, and challenge.

## Who This Is For

This project is useful for:

- investors who want to understand ETF portfolio tradeoffs before making decisions
- researchers who want a reproducible pipeline for portfolio experiments
- builders who want a tested foundation for portfolio analytics
- learners who want to understand how common portfolio metrics are produced

This project assumes the reader can work with command-line tools, but the concepts should be understandable without a finance background.

## What The Tool Does

At a high level, the pipeline:

1. Defines an ETF universe from configuration and metadata.
2. Downloads or loads historical price data.
3. Validates the data for common quality problems.
4. Converts prices into returns.
5. Estimates expected return and risk from historical data.
6. Optimizes a portfolio under configured constraints.
7. Runs a walk-forward backtest to simulate how the strategy would have behaved historically.
8. Generates HTML, Excel, figure, metric, and run-record artifacts.

The default output to read first is:

```text
reports/html/latest_report.html
```

## What Questions It Can Help Answer

The tool can help investigate questions like:

- How did this ETF allocation perform historically against a benchmark?
- How volatile was the portfolio?
- How large were historical drawdowns?
- Which ETFs or asset classes dominated the portfolio?
- How often did rebalancing change the portfolio?
- Did contribution-only investing drift away from the optimizer target?
- How sensitive is the result to the selected universe, constraints, and benchmark?

## What It Cannot Answer

The tool cannot answer:

- which ETF you personally should buy
- whether a portfolio will outperform in the future
- how taxes will affect a specific investor
- whether an ETF is suitable for a specific legal, income, or retirement situation
- how real broker execution would behave during market stress

The output is research evidence, not financial advice.

## Recommended Reading Path

If you are new to the project, read these in order:

1. [Glossary](glossary.md) for the main terms used by the project.
2. [How To Read The Report](how_to_read_the_report.md) to understand the generated HTML output.
3. [Interpretation Guide](interpretation_guide.md) to avoid common mistakes.
4. [Assumptions And Limitations](assumptions_and_limitations.md) to understand what the research does not prove.
5. [Runbook](runbook.md) when you are ready to reproduce or modify results.

If you are already comfortable with portfolio research, use:

- [Methodology](methodology.md) for the quantitative implementation.
- [Data Dictionary](data_dictionary.md) for schemas and metadata fields.
- [Model Card](model_card.md) for ML governance expectations.

## First Practical Workflow

Install dependencies:

```bash
uv sync --group dev
```

Run the full pipeline:

```bash
uv run etf-portfolio run-all --config configs/base.yaml
```

Open the generated report:

```text
reports/html/latest_report.html
```

Then read the report alongside [How To Read The Report](how_to_read_the_report.md).

## How To Think About Results

Treat every result as conditional:

- conditional on the ETF universe
- conditional on the historical period
- conditional on the benchmark
- conditional on the optimizer objective
- conditional on the configured constraints
- conditional on the data source and data quality

A backtest is a structured historical experiment. It is useful because it makes assumptions visible, not because it predicts the future.
