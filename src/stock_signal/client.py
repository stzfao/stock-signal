"""Async FMP API client with rate limiting and retries."""

import asyncio
import time
from collections import deque

import httpx

from stock_signal.config import FMP_API_KEY, FMP_BASE_URL, FMP_MAX_CONCURRENT, FMP_RATE_LIMIT_PER_MIN


class RateLimiter:
    """Sliding-window rate limiter."""

    def __init__(self, max_calls: int, period: float = 60.0):
        self._max_calls = max_calls
        self._period = period
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            # Evict timestamps outside the window
            while self._timestamps and self._timestamps[0] <= now - self._period:
                self._timestamps.popleft()
            if len(self._timestamps) >= self._max_calls:
                sleep_until = self._timestamps[0] + self._period
                await asyncio.sleep(sleep_until - now)
                # Re-evict after sleeping
                now = time.monotonic()
                while self._timestamps and self._timestamps[0] <= now - self._period:
                    self._timestamps.popleft()
            self._timestamps.append(time.monotonic())


class FMPClient:
    """Async FMP API client."""

    def __init__(
        self,
        api_key: str = FMP_API_KEY,
        max_concurrent: int = FMP_MAX_CONCURRENT,
        calls_per_minute: int = FMP_RATE_LIMIT_PER_MIN,
    ):
        self._api_key = api_key
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rate_limiter = RateLimiter(calls_per_minute)
        self._client = httpx.AsyncClient(
            base_url=FMP_BASE_URL,
            timeout=30.0,
            headers={"Accept": "application/json"},
        )

    async def __aenter__(self) -> "FMPClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(self, path: str, params: dict | None = None) -> list[dict]:
        """Make a rate-limited, retried GET request."""
        params = params or {}
        params["apikey"] = self._api_key

        max_retries = 3
        for attempt in range(max_retries):
            async with self._semaphore:
                await self._rate_limiter.acquire()
                try:
                    resp = await self._client.get(path, params=params)
                    if resp.status_code == 429 or resp.status_code >= 500:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2**attempt)
                            continue
                        resp.raise_for_status()
                    resp.raise_for_status()
                    data = resp.json()
                    if isinstance(data, list):
                        return data
                    if isinstance(data, dict) and "historical" in data:
                        return data["historical"]
                    if isinstance(data, dict):
                        return [data]
                    return []
                except httpx.HTTPStatusError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    raise
        return []

    # -- Price data --

    async def get_daily_prices(self, symbol: str) -> list[dict]:
        return await self._request(f"/historical-price-full/{symbol}")

    # -- Financial statements --

    async def get_income_statement(
        self, symbol: str, period: str = "annual", limit: int = 5
    ) -> list[dict]:
        return await self._request(
            f"/income-statement/{symbol}", {"period": period, "limit": limit}
        )

    async def get_balance_sheet(
        self, symbol: str, period: str = "annual", limit: int = 5
    ) -> list[dict]:
        return await self._request(
            f"/balance-sheet-statement/{symbol}", {"period": period, "limit": limit}
        )

    async def get_cash_flow(
        self, symbol: str, period: str = "annual", limit: int = 5
    ) -> list[dict]:
        return await self._request(
            f"/cash-flow-statement/{symbol}", {"period": period, "limit": limit}
        )

    # -- Earnings & estimates --

    async def get_earnings_surprises(self, symbol: str) -> list[dict]:
        return await self._request(f"/earnings-surprises/{symbol}")

    async def get_analyst_estimates(
        self, symbol: str, period: str = "annual", limit: int = 12
    ) -> list[dict]:
        return await self._request(
            f"/analyst-estimates/{symbol}", {"period": period, "limit": limit}
        )

    # -- Metrics --

    async def get_key_metrics(
        self, symbol: str, period: str = "annual", limit: int = 5
    ) -> list[dict]:
        return await self._request(
            f"/key-metrics/{symbol}", {"period": period, "limit": limit}
        )

    # -- Universe --

    async def get_sp500_constituents(self) -> list[dict]:
        return await self._request("/sp500_constituent")

    async def get_nasdaq_constituents(self) -> list[dict]:
        return await self._request("/nasdaq_constituent")

    async def get_stock_screener(self, **kwargs: object) -> list[dict]:
        return await self._request("/stock-screener", dict(kwargs))
