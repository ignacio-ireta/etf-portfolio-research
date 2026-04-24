"""Pandera schemas for ETF portfolio datasets."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column

VALID_ETF_ROLES = ("core", "satellite", "benchmark", "excluded")

ETF_UNIVERSE_METADATA_SCHEMA = pa.DataFrameSchema(
    {
        "ticker": Column(str, checks=Check.str_length(min_value=1), nullable=False),
        "name": Column(str, checks=Check.str_length(min_value=1), nullable=False),
        "asset_class": Column(str, checks=Check.str_length(min_value=1), nullable=False),
        "region": Column(str, checks=Check.str_length(min_value=1), nullable=False),
        "currency": Column(str, checks=Check.str_length(min_value=3, max_value=3), nullable=False),
        "expense_ratio": Column(
            float,
            checks=Check.in_range(min_value=0.0, max_value=0.01),
            nullable=False,
        ),
        "benchmark_index": Column(str, checks=Check.str_length(min_value=1), nullable=False),
        "is_leveraged": Column(bool, nullable=False),
        "is_inverse": Column(bool, nullable=False),
        "inception_date": Column(pa.DateTime, nullable=False),
        "role": Column(str, checks=Check.isin(VALID_ETF_ROLES), nullable=False),
    },
    strict=True,
    ordered=True,
)


def validate_etf_universe_metadata(data: pd.DataFrame) -> pd.DataFrame:
    """Validate ETF universe metadata and return a normalized dataframe."""

    normalized = data.copy()
    normalized["inception_date"] = pd.to_datetime(normalized["inception_date"])
    return ETF_UNIVERSE_METADATA_SCHEMA.validate(normalized)


__all__ = ["ETF_UNIVERSE_METADATA_SCHEMA", "VALID_ETF_ROLES", "validate_etf_universe_metadata"]
