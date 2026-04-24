# Trust And Safety

This page is generated from `src/etf_portfolio/trust_safety.py`, which is the canonical source for common false conclusions and safer report readings.

The project is a research tool, not financial advice. Use these warnings before turning any metric, chart, or optimized allocation into a conclusion.

## Common False Conclusions

### A high Sharpe ratio guarantees strong future returns.

**Safer reading:** A high Sharpe ratio means the portfolio earned more historical excess return per unit of historical volatility in this sample.

**Why it matters:** Sharpe ratios can fall when returns, volatility, correlations, or rates change.

**Where to check:** Metric Dictionary, Backtest Performance, Rolling Risk Metrics

### A good backtest proves the strategy will work.

**Safer reading:** A backtest is a historical experiment under chosen data, costs, constraints, rebalance rules, and benchmark assumptions.

**Why it matters:** Strategies can overfit the past, especially when many universes, objectives, windows, or constraints are tried.

**Where to check:** Reader Guide, Assumptions and Limitations, Stress Periods

### Historical mean returns are reliable forecasts.

**Safer reading:** Historical mean returns are transparent baseline estimates, not dependable predictions of future expected returns.

**Why it matters:** Mean estimates are noisy and can be dominated by sample period, regime changes, and recent winners.

**Where to check:** Metric Dictionary, Efficient Frontier Chart, Assumptions and Limitations

### ETF expense ratios are the only investment cost that matters.

**Safer reading:** Expense ratios matter because they compound, but real implementation can also include spreads, slippage, taxes, FX, account fees, and liquidity costs.

**Why it matters:** A low expense ratio does not automatically make an ETF cheaper to hold or trade in every investor's situation.

**Where to check:** Weighted Expense Ratio Over Time, Assumptions and Limitations

### Contribution-only portfolios always match optimizer targets.

**Safer reading:** Contribution-only rebalancing uses new cash to move toward targets, but realized holdings can drift when selling is restricted.

**Why it matters:** The actual simulated portfolio may carry different risks than the optimizer target, especially after large market moves.

**Where to check:** Latest Realized Portfolio Table, Optimizer Target Portfolio Table, Realized Constraint Warnings
