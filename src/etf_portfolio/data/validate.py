"""Validation utilities for ingested market data."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from etf_portfolio.data.schemas import ETF_UNIVERSE_METADATA_SCHEMA, validate_etf_universe_metadata


@dataclass(frozen=True)
class CrossCheckReport:
    """Summary of a price cross-check against an alternate provider."""

    primary_provider: str
    reference_provider: str
    max_relative_divergence: float
    min_overlap_observations: int
    per_ticker_max_divergence: pd.Series
    per_ticker_overlap: pd.Series

    def divergent_tickers(self) -> list[str]:
        """Return tickers where the relative divergence exceeded the threshold."""

        return self.per_ticker_max_divergence[
            self.per_ticker_max_divergence > self.max_relative_divergence
        ].index.tolist()


@dataclass(frozen=True)
class PriceValidationResult:
    """Summary of validation checks on a price matrix."""

    missing_data_fraction: pd.Series
    history_coverage: pd.Series
    suspicious_jumps: pd.DataFrame
    cross_check: CrossCheckReport | None = field(default=None)


def validate_price_data(
    prices: pd.DataFrame,
    *,
    metadata: pd.DataFrame | None = None,
    benchmark_ticker: str | None = None,
    min_history_ratio: float = 0.8,
    max_missing_fraction: float = 0.1,
    max_jump_abs_return: float = 0.25,
    cross_check_prices: pd.DataFrame | None = None,
    cross_check_max_relative_divergence: float = 0.005,
    cross_check_min_overlap: int = 20,
    primary_provider_name: str = "primary",
    cross_check_provider_name: str = "reference",
) -> PriceValidationResult:
    """Validate an adjusted-price matrix and return flagged jump diagnostics."""

    normalized = _normalize_price_frame(prices)
    _validate_index_and_columns(normalized)
    _validate_non_empty_columns(normalized)
    _validate_strictly_positive_observed_prices(normalized)

    missing_data_fraction = normalized.isna().mean()
    if (missing_data_fraction > max_missing_fraction).any():
        failing = missing_data_fraction[missing_data_fraction > max_missing_fraction]
        raise ValueError(
            "Missing data exceeds the allowed threshold for tickers: "
            f"{', '.join(failing.index.tolist())}."
        )

    history_coverage = normalized.notna().mean()
    if (history_coverage < min_history_ratio).any():
        failing = history_coverage[history_coverage < min_history_ratio]
        raise ValueError(
            f"Insufficient price history for tickers: {', '.join(failing.index.tolist())}."
        )

    normalized_metadata = _normalize_metadata(metadata) if metadata is not None else None
    if normalized_metadata is not None:
        _validate_inception_dates(normalized, normalized_metadata)

    if benchmark_ticker is not None:
        _validate_benchmark_overlap(normalized, benchmark_ticker)

    suspicious_jumps = _flag_suspicious_jumps(normalized, max_jump_abs_return=max_jump_abs_return)

    cross_check_report: CrossCheckReport | None = None
    if cross_check_prices is not None:
        cross_check_report = cross_check_price_data(
            normalized,
            cross_check_prices,
            max_relative_divergence=cross_check_max_relative_divergence,
            min_overlap_observations=cross_check_min_overlap,
            primary_provider_name=primary_provider_name,
            reference_provider_name=cross_check_provider_name,
        )

    return PriceValidationResult(
        missing_data_fraction=missing_data_fraction,
        history_coverage=history_coverage,
        suspicious_jumps=suspicious_jumps,
        cross_check=cross_check_report,
    )


def cross_check_price_data(
    primary_prices: pd.DataFrame,
    reference_prices: pd.DataFrame,
    *,
    max_relative_divergence: float = 0.005,
    min_overlap_observations: int = 20,
    primary_provider_name: str = "primary",
    reference_provider_name: str = "reference",
) -> CrossCheckReport:
    """Compare two price matrices and fail on material divergence.

    The relative divergence is measured as ``|p_primary - p_reference| / p_primary``
    on dates where both providers have observed prices. Any ticker whose
    maximum relative divergence exceeds ``max_relative_divergence`` triggers a
    ``ValueError``. Tickers with fewer than ``min_overlap_observations``
    overlapping observations are also rejected because the comparison would be
    statistically uninformative.
    """

    primary_normalized = _normalize_price_frame(primary_prices)
    reference_normalized = _normalize_price_frame(reference_prices)
    _validate_strictly_positive_observed_prices(primary_normalized)
    _validate_strictly_positive_observed_prices(reference_normalized)

    common_tickers = primary_normalized.columns.intersection(reference_normalized.columns)
    if common_tickers.empty:
        raise ValueError(
            "Cross-check failed: no overlapping tickers between "
            f"{primary_provider_name} and {reference_provider_name}."
        )

    per_ticker_overlap: dict[str, int] = {}
    per_ticker_max_divergence: dict[str, float] = {}
    insufficient_overlap: list[str] = []

    for ticker in common_tickers:
        joined = pd.concat(
            [
                primary_normalized[ticker].rename("primary"),
                reference_normalized[ticker].rename("reference"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        per_ticker_overlap[ticker] = int(joined.shape[0])
        if joined.shape[0] < min_overlap_observations:
            insufficient_overlap.append(ticker)
            per_ticker_max_divergence[ticker] = float("nan")
            continue
        denominator = joined["primary"].abs().replace(0.0, pd.NA)
        relative = (joined["primary"] - joined["reference"]).abs() / denominator
        per_ticker_max_divergence[ticker] = float(relative.dropna().max() or 0.0)

    if insufficient_overlap:
        raise ValueError(
            "Cross-check failed: insufficient overlapping observations "
            f"(< {min_overlap_observations}) for tickers: "
            f"{', '.join(sorted(insufficient_overlap))}."
        )

    divergence_series = pd.Series(per_ticker_max_divergence, dtype=float).sort_index()
    overlap_series = pd.Series(per_ticker_overlap, dtype=int).sort_index()

    divergent = divergence_series[divergence_series > max_relative_divergence]
    if not divergent.empty:
        offenders = ", ".join(
            f"{ticker}: {value:.4%}"
            for ticker, value in divergent.sort_values(ascending=False).items()
        )
        raise ValueError(
            "Cross-check failed: relative divergence between "
            f"{primary_provider_name} and {reference_provider_name} exceeded "
            f"{max_relative_divergence:.4%} for tickers: {offenders}."
        )

    return CrossCheckReport(
        primary_provider=primary_provider_name,
        reference_provider=reference_provider_name,
        max_relative_divergence=max_relative_divergence,
        min_overlap_observations=min_overlap_observations,
        per_ticker_max_divergence=divergence_series,
        per_ticker_overlap=overlap_series,
    )


def _normalize_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, datetime-indexed price frame."""

    if prices.empty:
        raise ValueError("Price data must not be empty.")

    normalized = prices.copy()
    normalized.index = pd.to_datetime(normalized.index)
    return normalized.sort_index()


def _validate_index_and_columns(prices: pd.DataFrame) -> None:
    """Validate uniqueness of the time index and ticker columns."""

    if prices.index.duplicated().any():
        duplicated = prices.index[prices.index.duplicated()].unique()
        raise ValueError(f"Duplicate dates found in price data: {duplicated.tolist()}.")

    if prices.columns.duplicated().any():
        duplicated = prices.columns[prices.columns.duplicated()].unique()
        raise ValueError(f"Duplicate tickers found in price data: {duplicated.tolist()}.")


def _validate_non_empty_columns(prices: pd.DataFrame) -> None:
    """Ensure no ticker column is entirely null."""

    empty_columns = prices.columns[prices.isna().all()].tolist()
    if empty_columns:
        raise ValueError(f"Entirely null price columns found: {empty_columns}.")


def _validate_strictly_positive_observed_prices(prices: pd.DataFrame) -> None:
    """Reject non-positive observed prices while allowing configured missing data."""

    if (prices <= 0).fillna(False).any().any():
        raise ValueError("Observed prices must be strictly positive.")


def _normalize_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Select the schema columns needed for inception-date validation."""

    required_columns = list(ETF_UNIVERSE_METADATA_SCHEMA.columns.keys())
    missing_columns = [column for column in required_columns if column not in metadata.columns]
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {missing_columns}.")
    return validate_etf_universe_metadata(metadata.loc[:, required_columns])


def _validate_inception_dates(prices: pd.DataFrame, metadata: pd.DataFrame) -> None:
    """Ensure no prices are present before an ETF's inception date."""

    inception_dates = metadata.set_index("ticker")["inception_date"]
    for ticker in prices.columns.intersection(inception_dates.index):
        first_valid_date = prices[ticker].dropna().index.min()
        inception_date = inception_dates.loc[ticker]
        if pd.notna(first_valid_date) and first_valid_date < inception_date:
            raise ValueError(
                f"Ticker {ticker} has price history before its inception date "
                f"{inception_date.date()}."
            )


def _validate_benchmark_overlap(prices: pd.DataFrame, benchmark_ticker: str) -> None:
    """Ensure benchmark and asset series have overlapping observed dates."""

    if benchmark_ticker not in prices.columns:
        raise ValueError(f"Benchmark ticker {benchmark_ticker} was not found in the price data.")

    benchmark_dates = set(prices[benchmark_ticker].dropna().index)
    asset_frame = prices.drop(columns=[benchmark_ticker], errors="ignore")
    if asset_frame.empty:
        return

    asset_dates = set(asset_frame.dropna(how="all").index)
    if not benchmark_dates.intersection(asset_dates):
        raise ValueError(
            f"Benchmark ticker {benchmark_ticker} does not overlap the asset date range."
        )


def _flag_suspicious_jumps(prices: pd.DataFrame, *, max_jump_abs_return: float) -> pd.DataFrame:
    """Return one-day absolute returns that exceed the configured threshold."""

    absolute_returns = prices.pct_change(fill_method=None).abs()
    flagged = absolute_returns.stack().rename("abs_return").reset_index()
    flagged.columns = ["date", "ticker", "abs_return"]
    return flagged.loc[flagged["abs_return"] > max_jump_abs_return].reset_index(drop=True)


__all__ = [
    "CrossCheckReport",
    "PriceValidationResult",
    "cross_check_price_data",
    "validate_price_data",
]
