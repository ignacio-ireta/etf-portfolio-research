"""Market data provider abstractions."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date

import pandas as pd


class PriceDataProvider(ABC):
    """Abstract interface for fetching adjusted price histories."""

    provider_name = "unknown"

    @abstractmethod
    def get_prices(
        self,
        tickers: Sequence[str],
        start_date: date | str,
        end_date: date | str | None,
    ) -> pd.DataFrame:
        """Return adjusted prices indexed by date and keyed by ticker."""


class YFinancePriceProvider(PriceDataProvider):
    """Prototype yfinance-backed provider for adjusted ETF prices."""

    provider_name = "yfinance"

    def get_prices(
        self,
        tickers: Sequence[str],
        start_date: date | str,
        end_date: date | str | None,
    ) -> pd.DataFrame:
        """Download adjusted close prices from Yahoo Finance."""

        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required to use YFinancePriceProvider. Add it to the environment "
                "with `uv sync`."
            ) from exc

        unique_tickers = list(dict.fromkeys(tickers))
        if not unique_tickers:
            raise ValueError("At least one ticker is required to fetch prices.")

        downloaded = yf.download(
            tickers=unique_tickers,
            start=str(start_date),
            end=None if end_date is None else str(end_date),
            auto_adjust=False,
            actions=False,
            progress=False,
            group_by="column",
        )
        prices = self._extract_adjusted_prices(downloaded, unique_tickers)
        if prices.empty:
            raise ValueError("The provider returned an empty price dataset.")

        prices.index = pd.to_datetime(prices.index)
        if getattr(prices.index, "tz", None) is not None:
            prices.index = prices.index.tz_localize(None)
        return prices.sort_index()

    @staticmethod
    def _extract_adjusted_prices(downloaded: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
        """Normalize yfinance output into a ticker-column price frame."""

        if downloaded.empty:
            return pd.DataFrame(columns=list(tickers))

        if isinstance(downloaded.columns, pd.MultiIndex):
            field_names = downloaded.columns.get_level_values(0)
            if "Adj Close" in field_names:
                prices = downloaded["Adj Close"]
            elif "Close" in field_names:
                prices = downloaded["Close"]
            else:
                raise ValueError(
                    "Unable to locate an adjusted or close price field in yfinance data."
                )
        else:
            price_field = "Adj Close" if "Adj Close" in downloaded.columns else "Close"
            if price_field not in downloaded.columns:
                raise ValueError(
                    "Unable to locate an adjusted or close price field in yfinance data."
                )
            ticker = tickers[0]
            prices = downloaded[[price_field]].rename(columns={price_field: ticker})

        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        prices = prices.copy()
        prices.columns = [str(column) for column in prices.columns]
        return prices


class TiingoPriceProvider(PriceDataProvider):
    """Tiingo-backed provider for adjusted ETF prices.

    Uses the Tiingo end-of-day prices REST API and returns adjusted closes
    aligned across the requested tickers. An API token is required; it is
    read from the ``api_key`` argument or the ``TIINGO_API_KEY`` environment
    variable.
    """

    provider_name = "tiingo"
    BASE_URL = "https://api.tiingo.com/tiingo/daily"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        session: object | None = None,
        request_timeout: float = 30.0,
    ) -> None:
        token = api_key or os.environ.get("TIINGO_API_KEY")
        if not token:
            raise ValueError(
                "TiingoPriceProvider requires an API token via api_key or TIINGO_API_KEY."
            )
        self._api_key = token
        self._session = session
        self._timeout = request_timeout

    def get_prices(
        self,
        tickers: Sequence[str],
        start_date: date | str,
        end_date: date | str | None,
    ) -> pd.DataFrame:
        """Download adjusted closes for the requested tickers from Tiingo."""

        unique_tickers = list(dict.fromkeys(tickers))
        if not unique_tickers:
            raise ValueError("At least one ticker is required to fetch prices.")

        session = self._session or self._build_session()
        start_str = str(start_date)
        end_str = None if end_date is None else str(end_date)

        per_ticker: dict[str, pd.Series] = {}
        for ticker in unique_tickers:
            payload = self._fetch_ticker_payload(
                session=session,
                ticker=ticker,
                start_date=start_str,
                end_date=end_str,
            )
            per_ticker[ticker] = self._payload_to_series(payload, ticker=ticker)

        prices = pd.concat(per_ticker, axis=1).sort_index()
        prices.columns = [str(column) for column in prices.columns]
        if prices.empty:
            raise ValueError("Tiingo returned an empty price dataset.")
        if getattr(prices.index, "tz", None) is not None:
            prices.index = prices.index.tz_localize(None)
        return prices

    def _build_session(self) -> object:
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "TiingoPriceProvider requires the 'requests' package. "
                "Install project dependencies with `uv sync`."
            ) from exc
        return requests.Session()

    def _fetch_ticker_payload(
        self,
        *,
        session: object,
        ticker: str,
        start_date: str,
        end_date: str | None,
    ) -> list[dict[str, object]]:
        url = f"{self.BASE_URL}/{ticker.lower()}/prices"
        params: dict[str, str] = {
            "format": "json",
            "resampleFreq": "daily",
            "startDate": start_date,
            "token": self._api_key,
        }
        if end_date is not None:
            params["endDate"] = end_date

        response = session.get(url, params=params, timeout=self._timeout)
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(
                f"Unexpected Tiingo payload for {ticker}: expected a list, "
                f"got {type(payload).__name__}."
            )
        return payload

    @staticmethod
    def _payload_to_series(
        payload: list[dict[str, object]],
        *,
        ticker: str,
    ) -> pd.Series:
        if not payload:
            return pd.Series(dtype=float, name=ticker)

        records = pd.DataFrame.from_records(payload)
        if "date" not in records.columns:
            raise ValueError(f"Tiingo payload for {ticker} is missing a 'date' field.")
        if "adjClose" in records.columns:
            price_field = "adjClose"
        elif "close" in records.columns:
            price_field = "close"
        else:
            raise ValueError(
                f"Tiingo payload for {ticker} is missing both 'adjClose' and 'close' fields."
            )

        series = pd.Series(
            data=pd.to_numeric(records[price_field], errors="raise").to_numpy(),
            index=pd.to_datetime(records["date"]).dt.tz_localize(None),
            name=ticker,
        ).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        return series


__all__ = ["PriceDataProvider", "TiingoPriceProvider", "YFinancePriceProvider"]
