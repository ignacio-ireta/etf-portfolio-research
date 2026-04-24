# How To Read The Report

The main generated report is:

```text
reports/html/latest_report.html
```

The report is generated from pipeline outputs. It is not manually edited and does not depend on notebook state.

## Suggested Reading Order

Read the report in this order:

1. ETF universe and data coverage.
2. Latest realized portfolio and optimizer target portfolio.
3. Exposure tables by asset class and region.
4. Benchmark comparison and performance metrics.
5. Drawdown and rolling risk charts.
6. Stress periods.
7. Return and risk attribution.
8. Assumptions, limitations, and warnings.

This order starts with what data was used, then what portfolio was built, then how it behaved historically.

## ETF Universe Summary

This section shows the ETFs available to the strategy and their metadata.

Use it to check:

- which ETFs were included
- which asset classes and regions are represented
- whether expense ratios look reasonable
- whether the portfolio is missing an exposure you expected

If the universe is too narrow or biased, the optimizer can only choose from that limited set.

## Data Coverage And Missing Data

These tables show whether price history is available and complete enough for each ETF.

Use them to check:

- whether all ETFs have enough history
- whether the benchmark overlaps with the ETF universe
- whether missing data could distort results

Weak data coverage can make backtests unreliable, especially for newer ETFs.

## Efficient Frontier Chart

The efficient frontier shows estimated risk and return tradeoffs for portfolios built from the ETF universe.

Use it to understand:

- whether the selected portfolio sits near other reasonable alternatives
- how much extra risk is associated with higher estimated return
- whether constraints are forcing the optimizer into a narrow area

Do not treat the frontier as a forecast. It is based on historical estimates.

## Latest Realized Portfolio

This table shows the latest simulated holdings after the backtest rules are applied.

This matters because the actual simulated portfolio can differ from the optimizer target, especially under contribution-only rebalancing.

Use it to answer:

- what the simulated investor actually ended up holding
- whether any ETF became too large
- whether asset-class exposure drifted over time

## Optimizer Target Portfolio

This table shows what the optimizer wanted at the latest rebalance.

Compare it with the latest realized portfolio:

- if they are similar, the real portfolio stayed close to target
- if they are different, rebalancing rules, drift, or constraints affected the actual holdings

The optimizer target is a recommendation inside the model, not a real-world execution guarantee.

## Portfolio Weights

This chart shows how ETF weights changed over time.

Use it to see:

- which ETFs dominated the portfolio
- whether the strategy changed aggressively
- whether a few ETFs drove most of the result

Large and frequent changes may imply higher turnover, higher costs, and more fragility.

## Exposure

Exposure sections summarize portfolio weights by categories such as asset class and region.

Use them to answer:

- how much of the portfolio is equity, fixed income, commodity, or real estate
- whether geographic exposure is concentrated
- whether the portfolio matches the intended investor profile

ETF names can hide concentration. Exposure tables make the underlying structure easier to inspect.

## Benchmark Comparison

This section compares the researched portfolio against benchmark alternatives.

Use it to ask:

- did the strategy outperform or underperform historically?
- did it take more or less risk than the benchmark?
- did it behave differently enough to justify its complexity?

Outperforming a benchmark in a backtest does not prove future outperformance.

## Backtest Performance

This section shows historical performance metrics and cumulative return.

Common things to inspect:

- CAGR for long-term compounded growth
- volatility for return variability
- Sharpe and Sortino ratios for risk-adjusted return
- maximum drawdown for worst historical loss from a previous high
- turnover for how much trading the strategy required

No single metric is enough. A portfolio can have strong return and still be unsuitable if drawdowns or turnover are too high.

## Drawdown Chart

The drawdown chart shows how far the portfolio fell from previous highs.

Use it to answer:

- how painful historical losses were
- how long losses persisted
- whether the portfolio behaved acceptably during major market stress

Drawdown is often easier to understand than volatility because it resembles the experience of watching account value decline.

## Rolling Risk Metrics

Rolling metrics show how risk changed through time instead of summarizing the whole period with one number.

Use them to check:

- whether volatility was stable or clustered in crises
- whether Sharpe ratio was consistent or period-dependent
- whether correlation to the benchmark changed

A good full-period result can hide long weak stretches.

## Stress Periods

Stress-period tables and charts show how the portfolio behaved during difficult historical windows.

Use them to ask:

- did the portfolio hold up when markets were stressed?
- did it protect capital better or worse than the benchmark?
- did defensive assets actually help?

Stress periods are historical examples, not exhaustive tests of every possible crisis.

## Weighted Expense Ratio

This section estimates the portfolio-level fund fee through time.

Use it to check:

- whether the portfolio is becoming more expensive
- whether high-cost ETFs are receiving large weights
- whether a similar exposure might be available more cheaply

Expense ratios are not the only cost, but they are persistent and worth monitoring.

## Return Attribution

Return attribution estimates which ETFs or asset classes contributed to portfolio return.

Use it to understand:

- whether gains came from broad diversification or a few holdings
- whether the strategy depended heavily on one asset class
- which exposures helped or hurt

Attribution explains the past. It does not prove the same drivers will work again.

## Risk Attribution

Risk attribution estimates which holdings contributed most to portfolio risk.

Use it to identify:

- hidden concentration
- assets with small weights but large risk impact
- whether risk is diversified or dominated by one exposure

Risk contribution can differ from portfolio weight.

## Realized Constraint Warnings

This section appears when realized holdings drift beyond configured constraints.

This is especially relevant for contribution-only rebalancing, where the strategy may avoid selling and instead uses new contributions to move toward target weights.

Warnings do not always mean the run failed. They mean the actual simulated portfolio violated a configured target or cap and should be interpreted carefully.

## Assumptions And Limitations

Always read this section before trusting a result.

It should clarify:

- data source limitations
- modeling assumptions
- benchmark assumptions
- cost and slippage assumptions
- what the backtest excludes

If the assumptions do not match the question you care about, the report may not answer that question.
