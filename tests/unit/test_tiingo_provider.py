from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from etf_portfolio.data.providers import TiingoPriceProvider


class _FakeResponse:
    def __init__(self, payload: list[dict[str, object]]) -> None:
        self._payload = payload
        self.status_code = 200

    def json(self) -> list[dict[str, object]]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    def __init__(self, payloads: dict[str, list[dict[str, object]]]) -> None:
        self._payloads = payloads
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, params: dict[str, str], timeout: float) -> _FakeResponse:
        self.calls.append((url, dict(params)))
        for ticker, payload in self._payloads.items():
            if url.endswith(f"/{ticker.lower()}/prices"):
                return _FakeResponse(payload)
        raise AssertionError(f"Unexpected Tiingo URL: {url}")


def test_tiingo_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API token"):
        TiingoPriceProvider()


def test_tiingo_provider_reads_token_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TIINGO_API_KEY", "env-token")
    provider = TiingoPriceProvider(session=_FakeSession({}))
    assert provider._api_key == "env-token"


def test_tiingo_provider_returns_aligned_adjusted_closes() -> None:
    payloads = {
        "VOO": [
            {"date": "2024-01-02T00:00:00.000Z", "adjClose": 100.0, "close": 99.0},
            {"date": "2024-01-03T00:00:00.000Z", "adjClose": 101.0, "close": 100.0},
        ],
        "BND": [
            {"date": "2024-01-02T00:00:00.000Z", "adjClose": 80.0, "close": 79.5},
            {"date": "2024-01-03T00:00:00.000Z", "adjClose": 80.4, "close": 79.9},
        ],
    }
    session = _FakeSession(payloads)
    provider = TiingoPriceProvider(api_key="test-token", session=session)

    prices = provider.get_prices(
        tickers=["VOO", "BND"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    )

    assert list(prices.columns) == ["VOO", "BND"]
    assert prices.index.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert prices.loc[pd.Timestamp("2024-01-02"), "VOO"] == pytest.approx(100.0)
    assert prices.loc[pd.Timestamp("2024-01-03"), "BND"] == pytest.approx(80.4)
    assert all("token" in params and params["token"] == "test-token" for _, params in session.calls)
    assert all(
        params["startDate"] == "2024-01-01" and params["endDate"] == "2024-01-31"
        for _, params in session.calls
    )


def test_tiingo_provider_falls_back_to_close_when_adj_close_missing() -> None:
    payloads = {
        "VOO": [
            {"date": "2024-01-02", "close": 99.0},
            {"date": "2024-01-03", "close": 100.0},
        ],
    }
    provider = TiingoPriceProvider(
        api_key="test-token",
        session=_FakeSession(payloads),
    )

    prices = provider.get_prices(
        tickers=["VOO"],
        start_date="2024-01-01",
        end_date="2024-01-31",
    )

    assert prices.loc[pd.Timestamp("2024-01-02"), "VOO"] == pytest.approx(99.0)
    assert prices.loc[pd.Timestamp("2024-01-03"), "VOO"] == pytest.approx(100.0)


def test_tiingo_provider_rejects_empty_universe() -> None:
    provider = TiingoPriceProvider(api_key="test-token", session=_FakeSession({}))
    with pytest.raises(ValueError, match="At least one ticker"):
        provider.get_prices(tickers=[], start_date="2024-01-01", end_date=None)


def test_tiingo_provider_rejects_unexpected_payload_shape() -> None:
    class BrokenSession:
        def get(self, url: str, params: dict[str, str], timeout: float) -> _FakeResponse:
            response = _FakeResponse([])
            response._payload = {"detail": "rate limited"}  # type: ignore[assignment]
            return response

    provider = TiingoPriceProvider(api_key="test-token", session=BrokenSession())
    with pytest.raises(ValueError, match="Unexpected Tiingo payload"):
        provider.get_prices(tickers=["VOO"], start_date="2024-01-01", end_date=None)
