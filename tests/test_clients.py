"""Unit tests for AClient, FMPClient exception handling and request plumbing."""

import pytest
import httpx

from stock_signal.core.clients import (
    AClient,
    AClientException,
    FMPClient,
    PaywallError,
    RateLimitError,
)
from stock_signal.config import DataStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TestClient(AClient):
    """Minimal concrete AClient for testing _request logic without HTTP."""
    @staticmethod
    def _parse(resp: httpx.Response) -> list[dict]:
        data = resp.json()
        return data if isinstance(data, list) else [data]


@pytest.fixture
def client() -> _TestClient:
    c = _TestClient.__new__(_TestClient)
    return c


async def _resp(status: int, data: object = None) -> httpx.Response:
    return httpx.Response(status, json=data or {})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_paywall_error_on_402(client):
    with pytest.raises(PaywallError):
        await client._request(_resp(402))


async def test_rate_limit_error_on_429(client):
    with pytest.raises(RateLimitError):
        await client._request(_resp(429))


async def test_error_counter_increments():
    AClientException._errors.clear()
    try:
        raise PaywallError("hit paywall")
    except PaywallError:
        pass
    try:
        raise PaywallError("hit again")
    except PaywallError:
        pass
    try:
        raise RateLimitError("rate limited")
    except RateLimitError:
        pass

    report = AClientException.get_report()
    assert report["PaywallError"] == 2
    assert report["RateLimitError"] == 1


async def test_fmp_injects_apikey_without_mutating_input():
    fmp = FMPClient.__new__(FMPClient)
    fmp._api_key = "secret-key"

    captured: dict = {}

    async def mock_get(path: str, params: dict) -> httpx.Response:
        captured.update(params)
        return httpx.Response(200, json=[{"ticker": "NVDA"}])

    fmp._get = mock_get

    original = {"symbol": "NVDA"}
    await fmp._get_request("/income-statement", original)

    assert captured["apikey"] == "secret-key"
    assert captured["symbol"] == "NVDA"
    assert "apikey" not in original  # input dict not mutated


def test_datastore_resolves_relative_to_absolute():
    ds = DataStore(db_path="data/test.duckdb", output_dir="data/output")
    assert ds.db_path.is_absolute()
    assert ds.db_path.name == "test.duckdb"
    assert ds.output_dir.is_absolute()
    assert ds.output_dir.name == "output"
