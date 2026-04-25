# Metric Dictionary

This page is generated from `src/etf_portfolio/metric_dictionary.py`, which is the canonical explanation source for metrics shown by the pipeline and report.

Each metric is defined with plain-English meaning, formula-level summary, good/bad interpretation, and caveats.

## Portfolio Construction

### Portfolio Weight

**Plain-English meaning:** The share of the portfolio allocated to one ETF or group.

**Formula-level summary:** Asset weight = asset market value / total portfolio value.

**Good/bad interpretation:** Weights should match the intended exposure. Very large weights indicate concentration.

**Caveats:** A small weight can still create large risk if the asset is very volatile.

## Optimization Inputs

### Expected Return

**Plain-English meaning:** The return estimate the optimizer uses before choosing weights.

**Formula-level summary:** Currently estimated from historical mean returns in the trailing training window.

**Good/bad interpretation:** Higher expected return can make an asset more attractive to the optimizer.

**Caveats:** This is backward-looking and should not be read as a reliable return forecast.

### Covariance

**Plain-English meaning:** How asset returns moved together historically.

**Formula-level summary:** Covariance matrix estimated from aligned asset return histories using the configured risk model.

**Good/bad interpretation:** Lower or negative covariance can improve diversification if relationships persist.

**Caveats:** Covariance estimates can change quickly across market regimes.

## Performance

### Portfolio Return

**Plain-English meaning:** The portfolio's percentage gain or loss over one period.

**Formula-level summary:** Sum of asset weight times asset return for the period.

**Good/bad interpretation:** Higher is better for a single period, all else equal.

**Caveats:** One-period return says little about risk, consistency, or future returns.

### Cumulative Return

**Plain-English meaning:** How much the portfolio grew or shrank over the full path.

**Formula-level summary:** Compound each period: product of (1 + return) minus 1.

**Good/bad interpretation:** Higher cumulative return is better, but only after risk is checked.

**Caveats:** Can hide severe losses or long weak stretches along the way.

### CAGR

**Plain-English meaning:** The annualized growth rate implied by the compounded return path.

**Formula-level summary:** Compound total return over the sample, then convert it to a one-year rate.

**Good/bad interpretation:** Higher CAGR is better if risk and drawdowns are acceptable.

**Caveats:** Smooths the path into one number and can make volatile results look cleaner than they felt.

## Risk

### Annualized Volatility

**Plain-English meaning:** How much portfolio returns fluctuated, scaled to a yearly number.

**Formula-level summary:** Standard deviation of periodic returns times sqrt(periods per year).

**Good/bad interpretation:** Lower volatility is usually a smoother ride.

**Caveats:** Does not distinguish upside surprises from downside losses.

### Portfolio Volatility

**Plain-English meaning:** Estimated total variability of a weighted portfolio.

**Formula-level summary:** Square root of weights' transpose times covariance matrix times weights.

**Good/bad interpretation:** Lower estimated volatility means less modeled return variability.

**Caveats:** Only as reliable as the covariance estimate and portfolio weights.

## Risk-Adjusted Return

### Sharpe Ratio

**Plain-English meaning:** Return earned per unit of total volatility after the risk-free rate.

**Formula-level summary:** Annualized excess return divided by annualized volatility.

**Good/bad interpretation:** Higher is generally better; below zero means underperforming cash.

**Caveats:** Penalizes upside and downside volatility equally. If volatility is zero or undefined, the pipeline reports 0.0 rather than NaN or Infinity.

### Sortino Ratio

**Plain-English meaning:** Return earned per unit of downside volatility.

**Formula-level summary:** Annualized excess return divided by annualized downside deviation.

**Good/bad interpretation:** Higher is generally better when downside risk matters most.

**Caveats:** Can be unstable when there are few negative observations. If downside deviation is zero or undefined, the pipeline reports 0.0.

### Calmar Ratio

**Plain-English meaning:** Compounded return compared with the worst drawdown.

**Formula-level summary:** CAGR divided by absolute value of maximum drawdown.

**Good/bad interpretation:** Higher is generally better if the drawdown estimate is credible.

**Caveats:** Very sensitive to one worst drawdown observation. If maximum drawdown is zero or undefined, the pipeline reports 0.0.

## Drawdown

### Max Drawdown

**Plain-English meaning:** The worst historical fall from a previous peak.

**Formula-level summary:** Minimum value of cumulative wealth divided by its prior running peak minus 1.

**Good/bad interpretation:** Closer to zero is better. A more negative value means deeper loss.

**Caveats:** Only captures the worst observed historical loss, not every possible loss.

### Drawdown

**Plain-English meaning:** The current decline from a previous high-water mark.

**Formula-level summary:** Current cumulative wealth / prior running maximum wealth minus 1.

**Good/bad interpretation:** Closer to zero is better; deeper negatives mean larger losses.

**Caveats:** Does not show how long recovery took unless read with the full chart.

## Implementation

### Turnover

**Plain-English meaning:** How much of the portfolio changed at rebalances on average.

**Formula-level summary:** Average gross traded-weight change across rebalance-to-rebalance transitions, excluding the initial allocation from cash.

**Good/bad interpretation:** Lower usually means fewer trades, lower costs, and less tax drag.

**Caveats:** The project models costs simply; real execution and taxes can differ.

## Concentration

### Average Number of Holdings

**Plain-English meaning:** The average count of ETFs with non-trivial portfolio weights.

**Formula-level summary:** Average count of weights above a small tolerance at each rebalance.

**Good/bad interpretation:** Higher can mean broader diversification.

**Caveats:** More holdings do not guarantee better diversification if exposures overlap.

### Largest Position

**Plain-English meaning:** The biggest single ETF weight observed in the portfolio history.

**Formula-level summary:** Maximum asset weight across all rebalance dates.

**Good/bad interpretation:** Lower usually means less single-ETF concentration.

**Caveats:** A broad ETF can be less risky than its weight suggests; a narrow ETF can be riskier.

### Herfindahl Concentration Index

**Plain-English meaning:** A concentration score based on squared portfolio weights.

**Formula-level summary:** Average across rebalances of the sum of squared weights.

**Good/bad interpretation:** Lower means more evenly spread weights; higher means concentration.

**Caveats:** Does not know whether ETFs hold overlapping underlying securities.

## Period Extremes

### Worst Month

**Plain-English meaning:** The worst compounded calendar-month return in the sample.

**Formula-level summary:** Compound returns by month, then take the minimum monthly return.

**Good/bad interpretation:** Closer to zero is better.

**Caveats:** A bad period just outside month boundaries may be split across months.

### Worst Quarter

**Plain-English meaning:** The worst compounded calendar-quarter return in the sample.

**Formula-level summary:** Compound returns by quarter, then take the minimum quarterly return.

**Good/bad interpretation:** Closer to zero is better.

**Caveats:** Quarterly windows are conventional but not the only stress window that matters.

### Best Month

**Plain-English meaning:** The best compounded calendar-month return in the sample.

**Formula-level summary:** Compound returns by month, then take the maximum monthly return.

**Good/bad interpretation:** Higher is better, but not if it came with unacceptable risk.

**Caveats:** Upside extremes can make a strategy look exciting without proving robustness.

## Benchmark Relative

### Beta

**Plain-English meaning:** How sensitive the portfolio was to benchmark moves.

**Formula-level summary:** Covariance of portfolio and benchmark returns divided by benchmark variance.

**Good/bad interpretation:** Beta above 1 moved more than the benchmark; below 1 moved less; near 0 moved independently.

**Caveats:** Beta depends on the chosen benchmark and historical window. If benchmark variance is zero or undefined, the pipeline reports 0.0.

### Alpha

**Plain-English meaning:** Return not explained by benchmark exposure in a simple beta model.

**Formula-level summary:** Portfolio CAGR minus (risk-free rate plus beta times benchmark excess return).

**Good/bad interpretation:** Higher historical alpha is better, but it is not proof of skill.

**Caveats:** Alpha can vanish when the benchmark, period, or risk model changes.

### Tracking Error

**Plain-English meaning:** How much the portfolio's returns differed from the benchmark.

**Formula-level summary:** Standard deviation of active returns, annualized. Active return is portfolio return minus benchmark return.

**Good/bad interpretation:** Lower means benchmark-like behavior; higher means more benchmark-relative risk.

**Caveats:** Low tracking error is not automatically good if the benchmark is unsuitable. With fewer than two aligned observations, the pipeline reports 0.0.

### Information Ratio

**Plain-English meaning:** Benchmark-relative return per unit of tracking error.

**Formula-level summary:** Average active return divided by active return volatility, annualized.

**Good/bad interpretation:** Higher is better for benchmark-relative strategies.

**Caveats:** Can be unstable when tracking error is very small. If tracking error is zero or undefined, the pipeline reports 0.0.

## Rolling Risk

### Rolling Volatility

**Plain-English meaning:** Volatility measured repeatedly over moving windows.

**Formula-level summary:** Rolling standard deviation of returns times sqrt(periods per year).

**Good/bad interpretation:** Stable or lower rolling volatility suggests steadier behavior.

**Caveats:** Window length strongly affects the result.

### Rolling Sharpe

**Plain-English meaning:** Sharpe ratio measured repeatedly over moving windows.

**Formula-level summary:** Rolling mean excess return divided by rolling volatility, annualized.

**Good/bad interpretation:** Consistently positive values are better than one isolated high value.

**Caveats:** Short windows are noisy and can flip quickly.

### Rolling Correlation

**Plain-English meaning:** How closely the portfolio moved with the benchmark over moving windows.

**Formula-level summary:** Rolling correlation between portfolio and benchmark returns.

**Good/bad interpretation:** Lower correlation can mean diversification; higher correlation means benchmark-like movement.

**Caveats:** Correlation can rise during crises when diversification is most needed.

## Stress Testing

### Stress-Period Return

**Plain-English meaning:** Compounded return during a named difficult historical period.

**Formula-level summary:** Product of (1 + return) within the stress window minus 1.

**Good/bad interpretation:** Less negative is better during market stress.

**Caveats:** Historical stress windows do not cover every future crisis shape.

## Cost

### Weighted Expense Ratio

**Plain-English meaning:** The portfolio-level annual ETF fee implied by current weights.

**Formula-level summary:** Sum of each ETF weight times its expense ratio.

**Good/bad interpretation:** Lower is usually better for similar exposures.

**Caveats:** Does not include taxes, spreads, market impact, or broker-specific costs.

## Attribution

### Return Attribution

**Plain-English meaning:** How much each asset or group contributed to portfolio return.

**Formula-level summary:** Per-period contribution is asset weight times asset return, then aggregated.

**Good/bad interpretation:** Positive contributors helped historical return; negative hurt it.

**Caveats:** Explains past contribution and does not identify future winners.

### Risk Attribution

**Plain-English meaning:** How much each asset or group contributed to portfolio volatility.

**Formula-level summary:** Euler decomposition using weights and the covariance matrix.

**Good/bad interpretation:** Lower, more balanced risk contribution usually means less hidden risk.

**Caveats:** Depends heavily on the covariance estimate.

## Optimization Outputs

### Efficient Frontier

**Plain-English meaning:** A set of modeled portfolios showing risk and return tradeoffs.

**Formula-level summary:** Optimized portfolios plotted by expected return and estimated volatility.

**Good/bad interpretation:** Portfolios higher and left look better under the model, subject to constraints.

**Caveats:** The frontier is estimate-driven and not a forecast.

## ML Evaluation

### Accuracy

**Plain-English meaning:** The share of classification predictions that matched actual outcomes.

**Formula-level summary:** Correct predictions divided by total predictions.

**Good/bad interpretation:** Higher is better when classes are balanced and errors cost the same.

**Caveats:** Can mislead when one class is much more common than another.

### Log Loss

**Plain-English meaning:** How well classification probabilities matched actual outcomes.

**Formula-level summary:** Negative average log likelihood of the true class probability.

**Good/bad interpretation:** Lower is better; confident wrong predictions are penalized heavily.

**Caveats:** Requires calibrated probabilities to be meaningful.

### ROC AUC

**Plain-English meaning:** How well a classifier ranked positives above negatives.

**Formula-level summary:** Area under the receiver operating characteristic curve.

**Good/bad interpretation:** Higher is better; 0.5 is roughly random ranking.

**Caveats:** Does not choose a trading threshold or account for economic payoff.

### RMSE

**Plain-English meaning:** Typical regression prediction error with larger errors penalized more.

**Formula-level summary:** Square root of mean squared prediction error.

**Good/bad interpretation:** Lower is better.

**Caveats:** Sensitive to outliers.

### MAE

**Plain-English meaning:** Typical absolute regression prediction error.

**Formula-level summary:** Mean absolute value of actual minus predicted values.

**Good/bad interpretation:** Lower is better.

**Caveats:** Does not penalize large misses as strongly as RMSE.

### R2

**Plain-English meaning:** How much target variation the regression model explained.

**Formula-level summary:** One minus residual sum of squares divided by total sum of squares.

**Good/bad interpretation:** Higher is better; values below zero are worse than a mean forecast.

**Caveats:** Can look good in-sample while failing out of sample.
