from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaError

from etf_portfolio.config import load_config, load_config_files
from etf_portfolio.data.schemas import VALID_ETF_ROLES, validate_etf_universe_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
ETF_UNIVERSE_CSV = REPO_ROOT / "data" / "metadata" / "etf_universe.csv"
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
UNIVERSE_US_CORE_CONFIG_PATH = REPO_ROOT / "configs" / "universe_us_core.yaml"
CONSTRAINTS_LONG_ONLY_CONFIG_PATH = REPO_ROOT / "configs" / "constraints_long_only.yaml"


def make_valid_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["SPY", "VXUS"],
            "name": ["SPDR S&P 500 ETF Trust", "Vanguard Total International Stock ETF"],
            "asset_class": ["Equity", "Equity"],
            "region": ["US", "Global ex-US"],
            "currency": ["USD", "USD"],
            "expense_ratio": [0.0009, 0.0007],
            "benchmark_index": ["S&P 500", "FTSE Global All Cap ex US"],
            "is_leveraged": [False, False],
            "is_inverse": [False, False],
            "inception_date": ["1993-01-22", "2011-01-26"],
            "role": ["benchmark", "benchmark"],
        }
    )


def test_validate_etf_universe_metadata_accepts_valid_data() -> None:
    validated = validate_etf_universe_metadata(make_valid_metadata())

    assert list(validated.columns) == [
        "ticker",
        "name",
        "asset_class",
        "region",
        "currency",
        "expense_ratio",
        "benchmark_index",
        "is_leveraged",
        "is_inverse",
        "inception_date",
        "role",
    ]
    assert pd.api.types.is_datetime64_any_dtype(validated["inception_date"])


def test_validate_etf_universe_metadata_rejects_missing_required_column() -> None:
    data = make_valid_metadata().drop(columns=["benchmark_index"])

    with pytest.raises(SchemaError):
        validate_etf_universe_metadata(data)


def test_validate_etf_universe_metadata_rejects_invalid_expense_ratio() -> None:
    data = make_valid_metadata()
    data.loc[0, "expense_ratio"] = -0.01

    with pytest.raises(SchemaError):
        validate_etf_universe_metadata(data)


def test_validate_etf_universe_metadata_rejects_invalid_currency_code() -> None:
    data = make_valid_metadata()
    data.loc[0, "currency"] = "US"

    with pytest.raises(SchemaError):
        validate_etf_universe_metadata(data)


def test_validate_etf_universe_metadata_rejects_non_boolean_flags() -> None:
    data = make_valid_metadata()
    data["is_leveraged"] = data["is_leveraged"].astype(object)
    data.loc[0, "is_leveraged"] = "no"

    with pytest.raises(SchemaError):
        validate_etf_universe_metadata(data)


def test_validate_etf_universe_metadata_rejects_invalid_role() -> None:
    data = make_valid_metadata()
    data.loc[0, "role"] = "speculative"

    with pytest.raises(SchemaError):
        validate_etf_universe_metadata(data)


def test_validate_etf_universe_metadata_rejects_missing_role_column() -> None:
    data = make_valid_metadata().drop(columns=["role"])

    with pytest.raises(SchemaError):
        validate_etf_universe_metadata(data)


def _load_universe_metadata() -> pd.DataFrame:
    raw = pd.read_csv(ETF_UNIVERSE_CSV)
    schema_columns = [
        "ticker",
        "name",
        "asset_class",
        "region",
        "currency",
        "expense_ratio",
        "benchmark_index",
        "is_leveraged",
        "is_inverse",
        "inception_date",
        "role",
    ]
    return validate_etf_universe_metadata(raw.loc[:, schema_columns])


def test_etf_universe_csv_validates_against_schema() -> None:
    metadata = _load_universe_metadata()

    assert metadata["role"].isin(VALID_ETF_ROLES).all()
    assert metadata["expense_ratio"].between(0.0, 0.01).all()


def test_every_universe_ticker_has_core_or_satellite_role() -> None:
    metadata = _load_universe_metadata().set_index("ticker")
    config = load_config(BASE_CONFIG_PATH)

    missing = [ticker for ticker in config.universe.tickers if ticker not in metadata.index]
    assert missing == [], f"Universe tickers missing metadata rows: {missing}"

    roles = metadata.loc[config.universe.tickers, "role"]
    invalid = roles[~roles.isin({"core", "satellite"})]
    assert invalid.empty, (
        f"Universe tickers must have role 'core' or 'satellite': {invalid.to_dict()}"
    )


def test_primary_benchmark_has_benchmark_role() -> None:
    metadata = _load_universe_metadata().set_index("ticker")
    config = load_config(BASE_CONFIG_PATH)

    primary = config.benchmark.primary
    assert primary in metadata.index, f"Primary benchmark {primary} missing metadata row"
    assert metadata.loc[primary, "role"] == "benchmark", (
        f"Primary benchmark {primary} must have role 'benchmark', "
        f"found {metadata.loc[primary, 'role']!r}"
    )


def test_all_secondary_benchmark_tickers_have_metadata_rows() -> None:
    metadata = _load_universe_metadata().set_index("ticker")
    config = load_config(BASE_CONFIG_PATH)

    secondary_tickers: set[str] = set()
    for mix in config.benchmark.secondary.values():
        secondary_tickers.update(mix.allocations.keys())

    missing = sorted(ticker for ticker in secondary_tickers if ticker not in metadata.index)
    assert missing == [], f"Secondary benchmark tickers missing metadata rows: {missing}"


def test_selected_universe_asset_class_bounds_exist_in_metadata_taxonomy() -> None:
    metadata = _load_universe_metadata().set_index("ticker")
    config = load_config_files(
        BASE_CONFIG_PATH,
        UNIVERSE_US_CORE_CONFIG_PATH,
        CONSTRAINTS_LONG_ONLY_CONFIG_PATH,
    )

    universe_asset_classes = set(
        metadata.loc[config.universe.tickers, "asset_class"].dropna().astype(str)
    )
    configured_asset_classes = set(config.constraints.asset_class_bounds)
    missing = sorted(configured_asset_classes - universe_asset_classes)

    assert missing == [], (
        "Configured constraints.asset_class_bounds keys must exist in ETF metadata "
        f"for the selected universe. missing={missing}, "
        f"available={sorted(universe_asset_classes)}"
    )
