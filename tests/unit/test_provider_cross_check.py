from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from etf_portfolio.data.ingest import ingest_price_data
from etf_portfolio.data.providers import PriceDataProvider
from etf_portfolio.data.validate import (
    CrossCheckReport,
    cross_check_price_data,
    validate_price_data,
)


class _FakeProvider(PriceDataProvider):
    def __init__(self, prices: pd.DataFrame, *, name: str = "fake") -> None:
        self._prices = prices
        self.provider_name = name

    def get_prices(
        self,
        tickers,
        start_date: date | str,
        end_date: date | str | None,
    ) -> pd.DataFrame:
        return self._prices.loc[:, list(tickers)].copy()


def _make_prices(values: dict[str, list[float]]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(next(iter(values.values()))), freq="D")
    return pd.DataFrame(values, index=index)


def test_cross_check_passes_on_aligned_prices() -> None:
    primary = _make_prices({"VOO": [100.0 + i * 0.1 for i in range(40)]})
    reference = primary.copy()

    report = cross_check_price_data(
        primary,
        reference,
        max_relative_divergence=0.001,
        min_overlap_observations=20,
    )

    assert isinstance(report, CrossCheckReport)
    assert report.divergent_tickers() == []
    assert report.per_ticker_overlap.loc["VOO"] == 40
    assert report.per_ticker_max_divergence.loc["VOO"] == pytest.approx(0.0)


def test_cross_check_raises_on_material_divergence() -> None:
    primary = _make_prices({"VOO": [100.0 + i * 0.1 for i in range(40)]})
    reference = primary.copy()
    reference.iloc[10, 0] = 200.0

    with pytest.raises(ValueError, match="Cross-check failed: relative divergence"):
        cross_check_price_data(
            primary,
            reference,
            max_relative_divergence=0.005,
            min_overlap_observations=20,
        )


def test_cross_check_rejects_insufficient_overlap() -> None:
    primary = _make_prices({"VOO": [100.0 + i * 0.1 for i in range(40)]})
    reference = primary.copy()
    reference.iloc[5:] = pd.NA

    with pytest.raises(ValueError, match="insufficient overlapping observations"):
        cross_check_price_data(
            primary,
            reference,
            max_relative_divergence=0.005,
            min_overlap_observations=20,
        )


def test_cross_check_rejects_disjoint_universes() -> None:
    primary = _make_prices({"VOO": [100.0 + i * 0.1 for i in range(40)]})
    reference = _make_prices({"BND": [80.0 + i * 0.05 for i in range(40)]})

    with pytest.raises(ValueError, match="no overlapping tickers"):
        cross_check_price_data(
            primary,
            reference,
            max_relative_divergence=0.005,
            min_overlap_observations=20,
        )


def test_validate_price_data_attaches_cross_check_report() -> None:
    index = pd.date_range("2024-01-01", periods=40, freq="D")
    primary = pd.DataFrame(
        {
            "VOO": [100.0 + i * 0.1 for i in range(40)],
            "BND": [80.0 + i * 0.05 for i in range(40)],
        },
        index=index,
    )
    reference = primary.copy() * 1.0005

    result = validate_price_data(
        primary,
        cross_check_prices=reference,
        cross_check_max_relative_divergence=0.001,
        cross_check_min_overlap=20,
        primary_provider_name="yfinance",
        cross_check_provider_name="tiingo",
    )

    assert result.cross_check is not None
    assert result.cross_check.primary_provider == "yfinance"
    assert result.cross_check.reference_provider == "tiingo"
    assert "VOO" in result.cross_check.per_ticker_overlap.index


def test_ingest_price_data_runs_cross_check_and_fails_on_divergence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    index = pd.date_range("2024-01-01", periods=40, freq="D")
    primary_prices = pd.DataFrame(
        {
            "VOO": [100.0 + i * 0.1 for i in range(40)],
            "BND": [80.0 + i * 0.05 for i in range(40)],
        },
        index=index,
    )
    reference_prices = primary_prices.copy()
    reference_prices.iloc[15, 0] = primary_prices.iloc[15, 0] * 1.05

    primary = _FakeProvider(primary_prices, name="yfinance")
    reference = _FakeProvider(reference_prices, name="tiingo")

    def fake_to_parquet(self: pd.DataFrame, path, *args: object, **kwargs: object) -> None:
        Path(path).write_text(self.to_csv())

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    raw_dir = tmp_path / "raw"

    with pytest.raises(ValueError, match="Cross-check failed: relative divergence"):
        ingest_price_data(
            primary,
            ["VOO", "BND"],
            start_date="2024-01-01",
            end_date="2024-02-15",
            raw_dir=raw_dir,
            cross_check_provider=reference,
            cross_check_max_relative_divergence=0.005,
            cross_check_min_overlap=20,
            min_history_ratio=0.0,
            max_missing_fraction=1.0,
        )
    assert not (raw_dir / "prices.parquet").exists()


def test_ingest_price_data_runs_cross_check_and_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    index = pd.date_range("2024-01-01", periods=40, freq="D")
    primary_prices = pd.DataFrame(
        {
            "VOO": [100.0 + i * 0.1 for i in range(40)],
            "BND": [80.0 + i * 0.05 for i in range(40)],
        },
        index=index,
    )
    reference_prices = primary_prices.copy()

    primary = _FakeProvider(primary_prices, name="yfinance")
    reference = _FakeProvider(reference_prices, name="tiingo")

    def fake_to_parquet(self: pd.DataFrame, path, *args: object, **kwargs: object) -> None:
        Path(path).write_text(self.to_csv())

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    artifacts = ingest_price_data(
        primary,
        ["VOO", "BND"],
        start_date="2024-01-01",
        end_date="2024-02-15",
        raw_dir=tmp_path / "raw",
        cross_check_provider=reference,
        cross_check_max_relative_divergence=0.005,
        cross_check_min_overlap=20,
        min_history_ratio=0.0,
        max_missing_fraction=1.0,
    )

    assert artifacts.validation_result.cross_check is not None
    assert artifacts.validation_result.cross_check.primary_provider == "yfinance"
    assert artifacts.validation_result.cross_check.reference_provider == "tiingo"
    assert artifacts.validation_result.cross_check.divergent_tickers() == []
    assert (tmp_path / "raw" / "prices.parquet").exists()
