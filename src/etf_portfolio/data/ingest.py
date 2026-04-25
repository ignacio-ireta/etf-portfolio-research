"""Data ingestion entrypoints for adjusted ETF price histories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from etf_portfolio.data.providers import PriceDataProvider
from etf_portfolio.data.schemas import ETF_UNIVERSE_METADATA_SCHEMA, validate_etf_universe_metadata
from etf_portfolio.data.validate import PriceValidationResult, validate_price_data

DEFAULT_METADATA_PATH = Path("data/metadata/etf_universe.csv")
CANONICAL_RAW_PRICES_FILENAME = "prices.parquet"


@dataclass(frozen=True)
class IngestionArtifacts:
    """Canonical output of one ingestion run.

    Only `raw_prices.parquet` is written to disk by this function; the
    `validate` and `features` pipeline stages own the downstream files
    (`prices_validated.parquet`, `returns.parquet`).
    """

    raw_prices_path: Path
    raw_prices: pd.DataFrame
    validation_result: PriceValidationResult


def load_etf_universe_metadata(metadata_path: str | Path = DEFAULT_METADATA_PATH) -> pd.DataFrame:
    """Load and validate ETF universe metadata from CSV."""

    metadata = pd.read_csv(metadata_path)
    required_columns = list(ETF_UNIVERSE_METADATA_SCHEMA.columns.keys())
    return validate_etf_universe_metadata(metadata.loc[:, required_columns])


def ingest_price_data(
    provider: PriceDataProvider,
    tickers: list[str],
    *,
    start_date: date | str,
    end_date: date | str | None,
    metadata: pd.DataFrame | None = None,
    benchmark_ticker: str | None = None,
    raw_dir: str | Path = Path("data/raw"),
    min_history_ratio: float = 0.8,
    max_missing_fraction: float = 0.1,
    max_jump_abs_return: float = 0.25,
    cross_check_provider: PriceDataProvider | None = None,
    cross_check_max_relative_divergence: float = 0.005,
    cross_check_min_overlap: int = 20,
) -> IngestionArtifacts:
    """Fetch and validate raw prices, then persist the canonical raw parquet.

    The canonical output path is `<raw_dir>/prices.parquet`. This is the
    single source of truth for the `validate` pipeline stage. When
    ``cross_check_provider`` is supplied, the same tickers are fetched from
    the alternate source and compared; material divergence raises a
    ``ValueError`` before any artifact is written.
    """

    requested_tickers = list(dict.fromkeys(tickers))
    raw_prices = provider.get_prices(tickers=tickers, start_date=start_date, end_date=end_date)
    adjusted_prices = _normalize_prices(raw_prices)
    _assert_requested_tickers_returned(
        adjusted_prices, requested_tickers, provider_name=provider.provider_name
    )

    cross_check_prices: pd.DataFrame | None = None
    cross_check_provider_name = "reference"
    if cross_check_provider is not None:
        cross_check_provider_name = getattr(cross_check_provider, "provider_name", "reference")
        reference_raw = cross_check_provider.get_prices(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )
        cross_check_prices = _normalize_prices(reference_raw)
        _assert_requested_tickers_returned(
            cross_check_prices,
            requested_tickers,
            provider_name=cross_check_provider_name,
        )

    validation_result = validate_price_data(
        adjusted_prices,
        metadata=metadata,
        benchmark_ticker=benchmark_ticker,
        min_history_ratio=min_history_ratio,
        max_missing_fraction=max_missing_fraction,
        max_jump_abs_return=max_jump_abs_return,
        cross_check_prices=cross_check_prices,
        cross_check_max_relative_divergence=cross_check_max_relative_divergence,
        cross_check_min_overlap=cross_check_min_overlap,
        primary_provider_name=getattr(provider, "provider_name", "primary"),
        cross_check_provider_name=cross_check_provider_name,
    )

    raw_dir_path = Path(raw_dir)
    raw_dir_path.mkdir(parents=True, exist_ok=True)
    raw_prices_path = raw_dir_path / CANONICAL_RAW_PRICES_FILENAME
    _write_parquet(adjusted_prices, raw_prices_path)

    return IngestionArtifacts(
        raw_prices_path=raw_prices_path,
        raw_prices=adjusted_prices,
        validation_result=validation_result,
    )


def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted datetime-indexed price frame."""

    normalized = prices.copy()
    normalized.index = pd.to_datetime(normalized.index)
    return normalized.sort_index()


def _assert_requested_tickers_returned(
    prices: pd.DataFrame,
    requested_tickers: list[str],
    *,
    provider_name: str,
) -> None:
    """Ensure providers did not silently omit requested ticker columns."""

    returned = {str(column) for column in prices.columns}
    missing = [ticker for ticker in requested_tickers if ticker not in returned]
    if missing:
        raise ValueError(
            f"{provider_name} did not return price columns for requested tickers: "
            f"{', '.join(missing)}."
        )


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to parquet with a clear dependency error if the engine is missing."""

    try:
        frame.to_parquet(path)
    except ImportError as exc:
        raise ImportError(
            "Writing parquet files requires an engine such as `pyarrow`. "
            "Install project dependencies with `uv sync`."
        ) from exc


__all__ = [
    "CANONICAL_RAW_PRICES_FILENAME",
    "DEFAULT_METADATA_PATH",
    "IngestionArtifacts",
    "ingest_price_data",
    "load_etf_universe_metadata",
]
