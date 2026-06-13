"""Property-based / fuzz tests for invariants and graceful failure.

These exercise the data-validation and optimization layers across many inputs:
malformed data must raise a domain error (not crash), and optimizer output must
respect the long-only / fully-invested invariants. See docs/testing.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from etf_portfolio.errors import DataValidationError
from etf_portfolio.optimization.optimizer import optimize_portfolio

pytestmark = pytest.mark.property


@given(
    bad_price=st.one_of(
        st.floats(max_value=0.0, min_value=-1e6, allow_nan=False, allow_infinity=False),
        st.just(0.0),
    )
)
def test_non_positive_prices_raise_domain_error(bad_price: float) -> None:
    from etf_portfolio.data.validate import validate_price_data

    prices = pd.DataFrame(
        {"AAA": [100.0, bad_price, 102.0]},
        index=pd.bdate_range("2020-01-01", periods=3),
    )
    with pytest.raises(DataValidationError):
        validate_price_data(prices)


@given(
    tickers=st.lists(
        st.text(
            alphabet=st.characters(min_codepoint=0x41, max_codepoint=0x2122),
            min_size=1,
            max_size=6,
        ),
        min_size=2,
        max_size=5,
        unique=True,
    )
)
@settings(max_examples=25, deadline=None)
def test_unicode_tickers_do_not_crash_returns(tickers: list[str]) -> None:
    from etf_portfolio.features.returns import simple_returns

    index = pd.bdate_range("2020-01-01", periods=10)
    data = {ticker: 100 * np.cumprod(1 + np.full(len(index), 0.001)) for ticker in tickers}
    prices = pd.DataFrame(data, index=index)
    returns = simple_returns(prices)
    assert list(returns.columns) == tickers
    assert not returns.isna().all().all()


@given(
    variances=st.lists(
        st.floats(min_value=1e-3, max_value=0.5, allow_nan=False),
        min_size=2,
        max_size=6,
    ),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=30, deadline=None)
def test_optimized_weights_are_long_only_and_fully_invested(
    variances: list[float], seed: int
) -> None:
    rng = np.random.default_rng(seed)
    n = len(variances)
    tickers = [f"T{i}" for i in range(n)]
    expected_returns = pd.Series(rng.uniform(-0.2, 0.4, size=n), index=tickers)
    covariance = pd.DataFrame(np.diag(variances), index=tickers, columns=tickers)

    weights = optimize_portfolio(
        expected_returns,
        covariance,
        method="min_volatility",
        max_weight=1.0,
    )

    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= -1e-6).all()
