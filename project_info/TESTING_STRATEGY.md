12. Add testing from the beginning

Testing strategy:

Unit tests:

Returns calculation
Annualization
Sharpe calculation
Beta calculation
Covariance shape/alignment
Weights sum to 1
Constraint enforcement
Transaction cost calculation
Drawdown calculation

Integration tests:

Run full pipeline on tiny synthetic dataset
Run optimizer on known toy data
Run backtest with 3 ETFs and 12 months of fake prices
Confirm output files are created

Regression tests:

Given fixed input data/config, metrics should not change unexpectedly
Golden-file tests for reports/tables

Bias/leakage tests:

Backtest cannot use future returns
Rebalance date uses only prior data
Feature window ends before prediction window
ETF unavailable before inception date

Example test cases:

def test_weights_sum_to_one():
    assert abs(weights.sum() - 1.0) < 1e-8

def test_long_only_constraints():
    assert (weights >= 0).all()

def test_no_future_data_in_backtest():
    assert training_window.max_date < rebalance_date

def test_transaction_costs_reduce_returns():
    assert net_return <= gross_return

This is boring. Boring is good. Boring is how the portfolio doesn’t lie to you.
