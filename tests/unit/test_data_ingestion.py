from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from etf_portfolio.data.ingest import ingest_price_data
from etf_portfolio.data.providers import PriceDataProvider
from etf_portfolio.data.validate import validate_price_data


class FakePriceProvider(PriceDataProvider):
    provider_name = "fake"

    def __init__(self, prices: pd.DataFrame) -> None:
        self._prices = prices

    def get_prices(
        self,
        tickers: list[str],
        start_date: date | str,
        end_date: date | str | None,
    ) -> pd.DataFrame:
        return self._prices.loc[:, tickers]


class OmittedColumnProvider(PriceDataProvider):
    provider_name = "omitted"

    def __init__(self, prices: pd.DataFrame) -> None:
        self._prices = prices

    def get_prices(
        self,
        tickers: list[str],
        start_date: date | str,
        end_date: date | str | None,
    ) -> pd.DataFrame:
        return self._prices.drop(columns=["BND"])


@pytest.fixture
def clean_prices() -> pd.DataFrame:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    return pd.DataFrame(
        {
            "VOO": [100.0, 101.0, 102.0, 103.0],
            "BND": [80.0, 80.5, 81.0, 81.5],
            "ACWI": [90.0, 90.4, 90.8, 91.2],
        },
        index=index,
    )


@pytest.fixture
def metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["VOO", "BND"],
            "name": ["Vanguard S&P 500 ETF", "Vanguard Total Bond Market ETF"],
            "asset_class": ["equity", "fixed_income"],
            "region": ["US", "US"],
            "currency": ["USD", "USD"],
            "expense_ratio": [0.0003, 0.0003],
            "benchmark_index": ["S&P 500", "Bloomberg U.S. Aggregate Float Adjusted Index"],
            "is_leveraged": [False, False],
            "is_inverse": [False, False],
            "inception_date": ["2010-09-07", "2007-04-03"],
            "role": ["core", "core"],
        }
    )


def test_validate_price_data_accepts_clean_data_and_flags_suspicious_jumps(
    clean_prices: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    prices = clean_prices.copy()
    prices.loc[pd.Timestamp("2024-01-04"), "VOO"] = 140.0

    result = validate_price_data(
        prices,
        metadata=metadata,
        benchmark_ticker="ACWI",
        max_missing_fraction=0.0,
        max_jump_abs_return=0.20,
    )

    assert result.suspicious_jumps.shape[0] == 2
    assert set(result.suspicious_jumps["ticker"]) == {"VOO"}


def test_validate_price_data_rejects_non_positive_observed_prices(
    clean_prices: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    prices = clean_prices.copy()
    prices.loc[pd.Timestamp("2024-01-03"), "BND"] = 0.0

    with pytest.raises(ValueError, match="strictly positive"):
        validate_price_data(prices, metadata=metadata)


def test_validate_price_data_does_not_forward_fill_suspicious_jump_gaps(
    clean_prices: pd.DataFrame,
) -> None:
    prices = clean_prices.copy()
    prices.loc[pd.Timestamp("2024-01-03"), "VOO"] = pd.NA
    prices.loc[pd.Timestamp("2024-01-04"), "VOO"] = 140.0
    prices.loc[pd.Timestamp("2024-01-05"), "VOO"] = 141.0

    result = validate_price_data(
        prices,
        min_history_ratio=0.0,
        max_missing_fraction=1.0,
        max_jump_abs_return=0.20,
    )

    assert result.suspicious_jumps.empty


def test_validate_price_data_rejects_pre_inception_history(
    clean_prices: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    prices = clean_prices.copy()
    prices.index = pd.to_datetime(["2009-01-02", "2009-01-05", "2009-01-06", "2009-01-07"])

    with pytest.raises(ValueError, match="before its inception date"):
        validate_price_data(prices, metadata=metadata)


def test_validate_price_data_requires_benchmark_overlap(clean_prices: pd.DataFrame) -> None:
    prices = clean_prices.copy()
    prices["ACWI"] = [None, None, 90.8, 91.2]
    prices.loc[pd.Timestamp("2024-01-04") :, ["VOO", "BND"]] = None

    with pytest.raises(ValueError, match="does not overlap"):
        validate_price_data(
            prices,
            benchmark_ticker="ACWI",
            min_history_ratio=0.0,
            max_missing_fraction=1.0,
        )


def test_ingest_price_data_writes_only_canonical_raw_artifact(
    clean_prices: pd.DataFrame,
    metadata: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = FakePriceProvider(clean_prices)

    def fake_to_parquet(
        self: pd.DataFrame, path: str | Path, *args: object, **kwargs: object
    ) -> None:
        Path(path).write_text(self.to_csv())

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    artifacts = ingest_price_data(
        provider,
        ["VOO", "BND", "ACWI"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        metadata=metadata,
        benchmark_ticker="ACWI",
        raw_dir=raw_dir,
        max_missing_fraction=0.0,
    )

    assert artifacts.raw_prices_path == raw_dir / "prices.parquet"
    assert artifacts.raw_prices_path.exists()
    assert list(artifacts.raw_prices.columns) == ["VOO", "BND", "ACWI"]

    extra_raw_files = [path for path in raw_dir.iterdir() if path.name != "prices.parquet"]
    assert extra_raw_files == [], (
        f"Ingestion should only write prices.parquet, found extras: {extra_raw_files}"
    )
    assert not processed_dir.exists() or not any(processed_dir.iterdir()), (
        "Ingestion stage must not write into data/processed; that is owned by validate/features."
    )


def test_ingest_price_data_uses_single_canonical_filename(
    clean_prices: pd.DataFrame,
    metadata: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = FakePriceProvider(clean_prices)

    def fake_to_parquet(
        self: pd.DataFrame, path: str | Path, *args: object, **kwargs: object
    ) -> None:
        Path(path).write_text(self.to_csv())

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    raw_dir = tmp_path / "raw"
    ingest_price_data(
        provider,
        ["VOO", "BND", "ACWI"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        metadata=metadata,
        benchmark_ticker="ACWI",
        raw_dir=raw_dir,
        max_missing_fraction=0.0,
    )

    files = sorted(path.name for path in raw_dir.iterdir())
    assert files == ["prices.parquet"], (
        f"Only prices.parquet should be present in data/raw, found: {files}"
    )
    assert not list(raw_dir.glob("prices_*.parquet")), (
        "Dated/provider-suffixed parquet files must not be created."
    )


def test_ingest_price_data_rejects_omitted_provider_columns(
    clean_prices: pd.DataFrame,
    metadata: pd.DataFrame,
    tmp_path: Path,
) -> None:
    provider = OmittedColumnProvider(clean_prices)

    with pytest.raises(ValueError, match="requested tickers: BND"):
        ingest_price_data(
            provider,
            ["VOO", "BND", "ACWI"],
            start_date="2024-01-01",
            end_date="2024-01-31",
            metadata=metadata,
            benchmark_ticker="ACWI",
            raw_dir=tmp_path / "raw",
            max_missing_fraction=0.0,
        )

    assert not (tmp_path / "raw" / "prices.parquet").exists()
