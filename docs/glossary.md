# Glossary

This glossary explains the main terms used in the ETF portfolio research pipeline. The definitions are intentionally plain-language first.

## Core Terms

**ETF**

An exchange-traded fund. It is a fund that trades like a stock and usually holds a basket of assets such as stocks, bonds, commodities, or real estate securities.

**Portfolio**

The collection of ETFs and weights being analyzed. A portfolio with 60% `VT` and 40% `BND` means 60 cents of every dollar is assigned to `VT` and 40 cents to `BND`.

**Weight**

The percentage of the portfolio allocated to one ETF, asset class, or region.

**Universe**

The set of ETFs the optimizer is allowed to choose from. If an ETF is not in the universe, the optimizer cannot allocate to it.

**Benchmark**

A reference portfolio used for comparison. A benchmark helps answer whether the researched portfolio performed better, worse, or differently than a simple alternative.

**Asset Class**

A broad category of investment exposure, such as equity, fixed income, commodity, or real estate.

**Expense Ratio**

The annual fund fee charged by an ETF, expressed as a percentage or decimal. In this project it is stored as a decimal, so `0.0003` means 0.03% per year.

## Return And Risk Terms

**Return**

The percentage gain or loss over a period.

**Cumulative Return**

The total compounded return over time. This shows how $1 would have grown or shrunk historically.

**Volatility**

How much returns fluctuate. Higher volatility usually means a bumpier ride, even if the final return is attractive.

**Drawdown**

The decline from a previous high point. A 20% drawdown means the portfolio fell 20% from its prior peak before recovering or falling further.

**Maximum Drawdown**

The worst peak-to-trough decline in the analyzed period. This is often one of the most intuitive risk measures for investors.

**Sharpe Ratio**

A measure of return per unit of volatility after accounting for a risk-free rate. Higher can be better, but it does not guarantee future performance.

**Sortino Ratio**

A risk-adjusted return measure that focuses more on downside volatility than total volatility.

**Calmar Ratio**

A return-to-drawdown measure. It compares compounded return to maximum drawdown.

**Beta**

How sensitive the portfolio was to the benchmark. A beta above 1 means the portfolio tended to move more than the benchmark; below 1 means it tended to move less.

**Alpha**

The portion of return not explained by benchmark exposure in a simple model. Positive historical alpha is not proof of skill or future outperformance.

**Tracking Error**

How differently the portfolio moved compared with the benchmark.

**Information Ratio**

Excess return versus the benchmark divided by tracking error. It measures benchmark-relative performance consistency.

## Optimization Terms

**Optimization**

The process of choosing portfolio weights that best satisfy an objective while respecting constraints.

**Objective**

The thing the optimizer is trying to achieve, such as maximum Sharpe ratio, minimum variance, equal weight, or risk parity.

**Constraint**

A rule the optimizer must follow, such as no short selling, maximum ETF weight, minimum bond exposure, or asset-class limits.

**Long-Only**

A portfolio rule that weights cannot be negative. The strategy can own an ETF or avoid it, but cannot short it.

**Minimum Variance**

An objective that seeks the lowest historical volatility portfolio within the allowed universe and constraints.

**Maximum Sharpe**

An objective that seeks the highest historical risk-adjusted return within the allowed universe and constraints.

**Risk Parity**

An objective that tries to balance risk contribution across holdings rather than simply balancing dollar weights.

**Efficient Frontier**

A chart of portfolios showing the historical tradeoff between expected return and risk. It is based on estimates and should not be treated as a forecast.

## Backtesting Terms

**Backtest**

A historical simulation of how a strategy would have behaved if it had been run in the past.

**Walk-Forward Backtest**

A backtest that only uses information available before each rebalance date. This reduces lookahead bias.

**Lookahead Bias**

A research error where future information accidentally influences a past decision. This can make results look much better than they really are.

**Rebalance**

The act of changing portfolio weights back toward a target allocation.

**Contribution-Only Rebalancing**

A rebalancing style that uses new cash contributions to move toward target weights without selling existing holdings unless a configured fallback allows it.

**Realized Holdings**

The actual simulated portfolio weights after accounting for contribution-only behavior, drift, trades, and costs.

**Optimizer Target**

The portfolio weights the optimizer wants before real-world frictions or contribution-only behavior are applied.

**Turnover**

How much of the portfolio is traded. Higher turnover can imply higher costs and more tax relevance in real accounts.

**Slippage**

The difference between an assumed trade price and the actual execution price. This project models slippage simply.

**Transaction Cost**

The estimated cost of buying or selling, modeled as basis points in this project.

## Data And Governance Terms

**Data Validation**

Checks that price and metadata inputs are plausible, complete enough, and aligned with expected schemas.

**Run Record**

A JSON artifact that records the effective configuration, method, timestamp, and run metadata for reproducibility.

**Model Card**

A governance document for machine learning models. It explains intended use, validation requirements, and approval criteria.

**Research-Grade Data**

Data that is acceptable for exploration but should be independently verified before important decisions.
