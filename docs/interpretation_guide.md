# Interpretation Guide

This guide explains how to interpret project outputs without overstating what they prove.

## Core Principle

Every result is conditional. A backtest result depends on the ETF universe, data source, historical window, optimizer objective, constraints, benchmark, cost assumptions, and rebalance rules.

A good result means the strategy looked good under those conditions. It does not mean the strategy will work in the future.

## What A Strong Result Looks Like

A result is more credible when:

- performance is competitive against simple benchmarks
- drawdowns are acceptable for the intended investor profile
- risk-adjusted metrics are strong without extreme concentration
- turnover is reasonable
- results are not driven by one ETF or one short period
- the portfolio remains feasible under constraints
- data coverage is strong
- assumptions are simple enough to explain

Simple, robust results are usually more trustworthy than complex results that only work under narrow assumptions.

## What A Weak Result Looks Like

Be cautious when:

- outperformance comes with much larger drawdowns
- one ETF explains most of the return
- the strategy changes weights aggressively
- turnover is high
- data coverage is poor
- the benchmark is too easy or inappropriate
- the selected time period excludes major bad periods
- optimizer targets differ sharply from realized holdings
- constraint warnings are frequent or severe

A strategy can look sophisticated and still be fragile.

## Common Mistakes

**Mistake: Treating historical return as expected future return**

Historical return is evidence, not a promise. Markets change.

**Mistake: Trusting one metric**

Sharpe ratio, CAGR, and maximum drawdown each show different things. No single number defines portfolio quality.

**Mistake: Ignoring drawdowns**

High long-term return may not matter if the path includes losses an investor could not tolerate.

**Mistake: Ignoring benchmark choice**

A strategy can look strong against a weak benchmark and weak against a better one.

**Mistake: Ignoring constraints**

Constraints shape the result. A portfolio may look safer because the config forced it to hold bonds or capped risky ETFs.

**Mistake: Confusing optimizer target with actual holdings**

The optimizer target is what the model wants. Realized holdings show what the simulated portfolio actually held after rebalance rules.

**Mistake: Assuming contribution-only rebalancing keeps constraints hard**

Contribution-only rebalancing can allow drift because it avoids selling. Read realized constraint warnings.

**Mistake: Ignoring costs**

Transaction costs, slippage, taxes, spreads, and fund expenses can materially affect real-world results.

## How To Compare Two Runs

When comparing two reports, change as few variables as possible.

Good comparisons:

- same universe, different objective
- same objective, different constraints
- same strategy, different benchmark
- same strategy, different backtest window

Weak comparisons:

- different universe, objective, benchmark, and dates all at once
- comparing a simple benchmark against an overfit strategy
- judging only by final cumulative return

When possible, inspect differences in return, drawdown, volatility, turnover, exposure, and concentration together.

## How To Read High Returns

High historical returns are interesting, but ask:

- did they come from higher risk?
- did they depend on one ETF?
- did they occur mostly in one market regime?
- did they survive transaction costs?
- did they beat a simple benchmark?
- would the drawdowns have been tolerable?

High return with hidden fragility is not necessarily a good portfolio.

## How To Read Low Volatility

Low volatility can be useful, but ask:

- did the portfolio give up too much return?
- did it rely heavily on bonds or cash-like assets?
- did it still suffer large drawdowns during stress periods?
- does the low-risk profile match the investor's horizon?

Low volatility is not the same as no risk.

## How To Read Maximum Drawdown

Maximum drawdown is one of the most practical risk metrics.

Ask:

- how large was the worst decline?
- how does it compare with the benchmark?
- would an investor have stayed invested during that decline?
- did the strategy recover quickly or remain impaired?

Behavioral tolerance matters. A strategy that investors abandon during drawdowns may fail in practice even if its long-term backtest looks good.

## How To Read Sharpe Ratio

Sharpe ratio helps compare return per unit of volatility.

Use it carefully:

- higher is generally better when comparing similar strategies
- it can hide tail risk and large drawdowns
- it depends on the selected historical period
- it can be inflated by overfitting

Sharpe ratio is useful, but it should not be the only decision metric.

## How To Read Turnover

Turnover shows how much the portfolio trades.

High turnover can mean:

- higher transaction costs
- more tax relevance in taxable accounts
- more dependence on perfect execution
- more sensitivity to model noise

Lower turnover is not automatically better, but it is usually easier to implement.

## Practical Interpretation Checklist

Before trusting a report, check:

- Is the ETF universe reasonable?
- Is the benchmark appropriate?
- Is data coverage strong enough?
- Are constraints aligned with the intended portfolio?
- Is performance better than simple alternatives?
- Are drawdowns acceptable?
- Is turnover reasonable?
- Is concentration controlled?
- Are warning sections empty or explainable?
- Are the assumptions acceptable for the question being asked?

If the answer to any of these is unclear, treat the result as preliminary.
