from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_portfolio.data.validate import validate_price_data
from etf_portfolio.features.returns import calculate_simple_returns


def test_simple_returns_middle_missing_produces_nas() -> None:
    """Ensure missing middle prices don't cause implicit forward-fill catch-up."""
    prices = pd.Series(
        [100.0, np.nan, 110.0, 115.5],
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
        name="price",
    )

    # By default, missing="preserve"
    returns = calculate_simple_returns(prices)

    # 2024-01-01: price 100
    # 2024-01-02: price NaN -> return at 01-02 should be NaN
    # 2024-01-03: price 110 -> return at 01-03 should be NaN (compared to 01-02 NaN)
    # 2024-01-04: price 115.5 -> return at 01-04 should be 0.05 (compared to 01-03 110)

    # pandas pct_change(fill_method=None) behavior:
    # 01-02: (NaN - 100)/100 = NaN
    # 01-03: (110 - NaN)/NaN = NaN
    # 01-04: (115.5 - 110)/110 = 0.05

    assert pd.isna(returns.reindex(prices.index).loc["2024-01-02"])
    assert pd.isna(returns.reindex(prices.index).loc["2024-01-03"])
    assert returns.loc["2024-01-04"] == pytest.approx(0.05)


def test_validate_price_data_rejects_zero_prices() -> None:
    """Ensure zero prices are rejected during validation."""
    prices = pd.DataFrame(
        {"VTI": [100.0, 0.0, 101.0]}, index=pd.date_range("2024-01-01", periods=3, freq="D")
    )

    with pytest.raises(ValueError, match="Observed prices must be strictly positive"):
        validate_price_data(prices)


def test_validate_price_data_rejects_negative_prices() -> None:
    """Ensure negative prices are rejected during validation."""
    prices = pd.DataFrame(
        {"VTI": [100.0, -1.0, 101.0]}, index=pd.date_range("2024-01-01", periods=3, freq="D")
    )

    with pytest.raises(ValueError, match="Observed prices must be strictly positive"):
        validate_price_data(prices)
