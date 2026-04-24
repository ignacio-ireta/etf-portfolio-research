# Assumptions And Limitations

This project is a research tool. It is designed to make assumptions visible and results reproducible, not to provide personalized financial advice.

## Intended Use

Use this project to:

- compare ETF allocation strategies
- inspect historical risk and return tradeoffs
- test portfolio constraints
- generate reproducible reports
- learn how portfolio analytics are constructed

Do not use this project as the sole basis for financial, tax, legal, or retirement decisions.

## Data Assumptions

The default price provider is `yfinance`, which is suitable for research and prototyping but not production-grade market data.

The pipeline assumes:

- adjusted close prices are appropriate for return calculations
- metadata such as asset class, region, currency, and expense ratio is accurate
- ETF histories are long enough for the selected backtest window
- benchmark data overlaps sufficiently with the ETF universe
- missing data and suspicious jumps are detected well enough for research use

If a result matters, validate data against another source.

## Historical Return Assumptions

The current expected return estimator is historical mean return. This means the optimizer uses past returns from the available trailing window as an estimate.

This is simple and transparent, but fragile:

- past winners may not remain winners
- market regimes change
- short windows can be noisy
- long windows can include stale regimes
- unusual crisis or boom periods can dominate estimates

Historical mean return should be treated as a baseline, not a reliable forecast.

## Risk Model Assumptions

The project currently supports sample covariance and Ledoit-Wolf covariance estimation.

These models estimate how assets moved together historically. They do not know whether future correlations will change during stress.

In real crises, assets that looked diversified can become more correlated.

## Optimization Assumptions

The optimizer chooses weights based on the configured objective and constraints.

The result depends heavily on:

- ETF universe
- expected return estimator
- covariance estimator
- maximum ETF weights
- asset-class bounds
- ticker-specific bounds
- bond exposure requirements
- expense-ratio inputs

An optimized portfolio is not automatically better than a simple portfolio. It is only the best portfolio according to the selected model and constraints.

## Backtest Assumptions

The project uses walk-forward backtesting, which helps reduce lookahead bias by using only trailing data before each rebalance date.

Even with walk-forward design, the backtest still assumes:

- historical market conditions are informative
- trades happen at simplified prices
- costs and slippage are represented by simple assumptions
- taxes are not modeled
- intraday behavior is not modeled
- market impact is not modeled
- ETF liquidity is not fully modeled

A backtest is a historical simulation, not a prediction.

## Rebalancing Assumptions

The default configuration uses contribution-only rebalancing. This means new cash contributions are used to move the portfolio toward target weights, while selling can be restricted.

This can be useful for tax-aware or long-horizon accumulation scenarios, but it also means realized holdings can drift away from optimizer targets.

If realized constraint warnings appear, interpret the actual simulated portfolio using the realized holdings, not only the optimizer target.

## Benchmark Assumptions

Benchmark choice strongly affects interpretation.

A good benchmark should be:

- simple
- investable
- relevant to the investor's opportunity set
- similar enough to make comparison meaningful
- hard enough that outperformance matters

If the benchmark is too weak or unrelated, outperformance may not be meaningful.

## Cost Assumptions

The project can model transaction costs, slippage, and ETF expense ratios, but the model is simplified.

It does not fully capture:

- bid-ask spreads
- broker-specific execution quality
- market impact
- tax lots
- capital gains taxes
- dividend withholding taxes
- account-level fees
- foreign exchange costs

Costs should be reviewed before applying results to real portfolios.

## Tax And Legal Limitations

The project does not provide tax advice or legal advice.

It does not model:

- investor residency
- account type
- capital gains rules
- dividend taxation
- estate tax
- withholding tax
- local fund regulation
- UCITS versus US-domiciled ETF suitability

These issues can dominate real-world portfolio outcomes.

## Machine Learning Limitations

The repository contains an ML research layer, but ML models are not automatically approved for portfolio construction.

Any model used for portfolio decisions should pass governance checks, including:

- leakage checks
- chronological validation
- baseline comparison
- stability across windows
- documented failure modes
- reproducible training

Attractive in-sample ML results are not enough.

## Not Financial Advice

The project output is educational and research-oriented.

It should not be interpreted as:

- a recommendation to buy or sell securities
- personalized investment advice
- tax advice
- legal advice
- a guarantee of future performance

Use the project to ask better questions, document assumptions, and compare alternatives more rigorously.
