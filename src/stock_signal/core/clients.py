"""Clients for financial providers with rate limiting and retries."""

import asyncio

import re
import json5
import logging
from collections import defaultdict
from typing import override
import pandas as pd
from bs4 import BeautifulSoup
import httpx
from aiolimiter import AsyncLimiter
from curl_cffi.requests import AsyncSession
from httpx_retries import Retry, RetryTransport

from ..config import FMP, SlickCharts, NASDAQ, NYSE, StockAnalysis

log = logging.getLogger(__name__)

class MClientException(Exception):
    """Base for all client errors. Tracks per-subclass counts for reporting."""
    _errors: defaultdict = defaultdict(int)

    def __init__(self, *args):
        super().__init__(*args)
        MClientException._errors[self.__class__] += 1

    @classmethod
    def get_report(cls) -> dict[str, int]:
        return {k.__name__: v for k, v in cls._errors.items()}


class PaywallError(MClientException):
    """FMP returned 402 — endpoint requires a higher subscription tier."""


class RateLimitError(MClientException):
    """FMP returned 429 after all retries — rate limit exhausted."""


class MClient:
    """Base class providing rate limiting and concurrency control for subclasses."""

    def __init__(self, config: FMP | StockAnalysis) -> None:
        self._base_url = config.base_url
        self._semaphore = asyncio.Semaphore(config.max_conn)
        self._limiter = AsyncLimiter(config.rpm, 60)


class FMPClient(MClient):
    """Async FMP API client with per-minute rate limiting and concurrency cap."""

    def __init__(self, config: FMP):
        super().__init__(config)
        self._api_key = config.api_key
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=30.0,
            headers={"Accept": "application/json"},
            transport=RetryTransport(retry=Retry(
                total=3,
                backoff_factor=1.0,
                respect_retry_after_header=True,
                allowed_methods=["GET", "POST"],
                status_forcelist={429, *range(500, 512)},
            )),
        )

    async def __aenter__(self) -> "FMPClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: dict = {}) -> httpx.Response:
        params = {**params, "apikey": self._api_key}
        async with self._limiter:
            async with self._semaphore:
                return await self._client.get(path, params=params)

    @staticmethod
    def _parse(resp: httpx.Response) -> list[dict]:
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "historical" in data:
            return data["historical"]
        if isinstance(data, dict):
            return [data]
        return []

    async def _request(self, path: str, params: dict) -> list[dict]:
        try:
            resp = await self._get(path, params=params)
            if resp.status_code == 402:
                raise PaywallError(f"pay-walled: {resp.url}")
            if resp.status_code == 429:
                raise RateLimitError(f"rate limit exhausted: {resp.url}")
            resp.raise_for_status()
            return self._parse(resp)
        except MClientException:
            raise
        except (httpx.HTTPError, httpx.RequestError) as exc:
            raise RuntimeError(f"request failed: {exc}") from exc

    # -- Price data --

    async def get_daily_prices(self, symbol: str) -> list[dict]:
        return await self._request("/historical-price-eod/full", {"symbol": symbol})

    # -- Financial statements --

    async def get_income_statement(self, symbol: str, period: str = "annual", limit: int = 5) -> list[dict]:
        return await self._request("/income-statement", {"symbol": symbol, "period": period, "limit": limit})

    async def get_balance_sheet(self, symbol: str, period: str = "annual", limit: int = 5) -> list[dict]:
        return await self._request("/balance-sheet-statement", {"symbol": symbol, "period": period, "limit": limit})

    async def get_cash_flow(self, symbol: str, period: str = "annual", limit: int = 5) -> list[dict]:
        return await self._request("/cash-flow-statement", {"symbol": symbol, "period": period, "limit": limit})

    # -- Earnings & estimates --

    async def get_earnings(self, symbol: str, limit: int = 40) -> list[dict]:
        """epsActual, epsEstimated, revenueActual, revenueEstimated. Up to 99 quarters."""
        return await self._request("/earnings", {"symbol": symbol, "limit": limit})

    async def get_analyst_estimates(self, symbol: str, period: str = "quarter", limit: int = 40) -> list[dict]:
        """Forward EPS estimates with high/low/avg and analyst count per quarter."""
        return await self._request("/analyst-estimates", {"symbol": symbol, "period": period, "limit": limit})

    # -- Insider trading --

    async def get_insider_statistics(self, symbol: str) -> list[dict]:
        """Aggregate insider buy/sell statistics for a symbol."""
        return await self._request("/insider-trading/statistics", {"symbol": symbol})

    async def get_insider_latest(self, page: int = 0, limit: int = 100, date: str | None = None) -> list[dict]:
        """Latest insider transactions across all symbols."""
        params: dict = {"page": page, "limit": limit}
        if date:
            params["date"] = date
        return await self._request("/insider-trading/latest", params)


class SClient:
    def __init__(self, config: SlickCharts | NASDAQ | NYSE):
        self._base_url = config.base_url

    async def __aenter__(self) -> "SClient":
        return self

    async def __aexit__(self, *exc: object):
        pass

    @staticmethod
    def _parse(resp) -> set | pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def _traverse_json(label: str, text: str) -> str:
        try:
            # get the first '[' after the label
            label_index = text.index(label)
            start = text.index("[", label_index)
        except ValueError:
            raise ValueError(f"could not find {label!r} start point")
        raw = ""
        depth = 0
        for i in range(start, len(text)):
            depth += 1 if text[i] == "[" else -1 if text[i] == "]" else 0
            if depth == 0:
                raw = text[start : i + 1]
                break

        if not raw:
            raise ValueError(f"{label!r} array not terminated")

        return raw

    async def _post_request(self, path: str, data: dict):
        async with httpx.AsyncClient(base_url=self._base_url,
                                     timeout=60.0,
                                     headers={"Accept": "application/json"}) as client:
            resp = await client.post(f"{self._base_url}{path}", json=data)
        resp.raise_for_status()
        return self._parse(resp)

    async def _request(self, path: str, params: dict = {}):
        async with AsyncSession(impersonate="chrome124") as session:
            resp = await session.get(f"{self._base_url}{path}", params=params)
        resp.raise_for_status()
        return self._parse(resp)


class SlickChartsClient(SClient):
    def __init__(self, config: SlickCharts):
        super().__init__(config)

    @staticmethod
    def _parse(resp) -> list[dict]:
        soup = BeautifulSoup(resp.text, "html.parser")
        script = soup.find(string=re.compile(r'sp500List\s*:'))
        if not script or not script.string:
            raise ValueError("sp500List not found - Check if Cloudflare is blocking or DOM changed.")

        # bracket-match to extract the full array
        list_label = "sp500List:"
        raw = SClient._traverse_json(list_label, script.string)

        return json5.loads(raw)

    async def get_sp500_constituents(self) -> set:
        resp = await self._request("/sp500")
        constituents = set(map(lambda x: x["symbol"], resp))
        return constituents


class NYSEClient(SClient):
    def __init__(self, config: NYSE):
        super().__init__(config)
        self.data = config.serialized_body

    @staticmethod
    def _parse(resp) -> set:
        rows = resp.json()
        df = pd.DataFrame(rows)
        column_to_keep = "normalizedTicker"
        df[column_to_keep] = df[column_to_keep].str.strip()
        return set(df[column_to_keep].tolist())

    async def get_nyse_constituents(self) -> set:
        return await self._post_request("/quotes/filter", data=self.data)


class NASDAQClient(SClient):
    def __init__(self, config: NASDAQ):
        super().__init__(config)
        self.params = {
            "tableonly": config.table_only,
            "limit": config.limit
        }

    @staticmethod
    def _parse(resp) -> pd.DataFrame:
        table = resp.json()["data"]["table"]
        df = pd.DataFrame(table["rows"])
        df["symbol"] = df["symbol"].str.strip()
        df["marketCap"] = pd.to_numeric(df["marketCap"].str.replace(",", ""), errors="coerce").astype("Int64")
        return df[["symbol", "marketCap"]].dropna(subset=["marketCap"])

    async def get_nasdaq_constituents(self) -> pd.DataFrame:
        return await self._request("/stocks", params=self.params)


# ---------------------------------------------------------------------------
# StockAnalysis screener — free, no auth, full universe in one call per metric
# ---------------------------------------------------------------------------

# All data points fetched on each refresh. Mapped to snake_case column names.
_SA_DATA_POINTS: dict[str, str] = {
    # -- Hard filters --
    "zScore":                    "altman_z",               # exclude < 1.8
    "marketCap":                 "market_cap",              # exclude < 300M USD
    "analystCount":              "analyst_count",           # exclude < 3 (micro-cap noise)
    "price":                     "price",                   # exclude < 1.0 (penny stocks)
    "interestCoverage":          "interest_coverage",       # exclude < 1.5 (can't service debt)
    # -- Price signals: replaces FMP historical price calls --
    "ch1y":                      "ch_1y",                   # 12-month return
    "ch1m":                      "ch_1m",                   # 1-month return (skip month)
    "positionInRange":           "position_in_range",       # price/52wk_high *100; proximity factor precomputed
    "high52":                    "high_52w",                # 52wk high (raw, for manual calc)
    "dollarVolume":              "dollar_volume",           # replaces pipeline ADV calc
    "relativeVolume":            "relative_volume",         # unusual activity flag
    # -- Factors: already in signal stack --
    "fScore":                    "f_score",                 # Piotroski 0-9
    "grossMargin":               "gross_margin",            # gross profitability proxy
    "roa":                       "roa",                     # return on assets
    "roic":                      "roic",                    # quality anchor for US-long track
    # -- Factors: net issuance (replaces FMP cashflow computation) --
    "sharesQoQ":                 "shares_qoq",              # quarterly share dilution %
    "sharesYoY":                 "shares_yoy",              # annual share dilution %
    # -- Factors: gameplan #1 (analyst conviction) --
    "priceTargetChange":         "price_target_change",     # % PT revision; highest-ROI addition
    "analystRatings":            "analyst_ratings",         # Strong Buy → Strong Sell
    "earningsEpsEstimateGrowth": "eps_estimate_growth",     # EPS revision breadth proxy
    "epsNextYear":               "eps_next_year",           # forward EPS
    # -- Factors: gameplan #2 (valuation-aware momentum) --
    "evEbitda":                  "ev_ebitda",               # value trend signal
    "peForward":                 "pe_forward",              # forward P/E (better than trailing)
    "peRatio":                   "pe_ratio",                # trailing P/E
    # -- Factors: Tier 2 --
    "shortFloat":                "short_float",             # % float short
    "shortRatio":                "short_ratio",             # days to cover
    "revenueGrowth":             "revenue_growth",          # YoY revenue %
    "fcfPerShare":               "fcf_per_share",           # FCF quality
    "fcfYield":                  "fcf_yield",               # FCF yield
    "netCash":                   "net_cash",                # net cash (negative = net debt)
    # -- Entry/exit guards --
    "ma200ch":                   "ma_200_pct",              # % from SMA-200 → mean_reversion_risk
    "ma50vs200":                 "ma_50_vs_200",            # golden/death cross
    "peRatio3Y":                 "pe_ratio_3y",             # 3yr avg PE → valuation_penalty
    "rsi":                       "rsi",                     # RSI 14-day
    # -- Revision blend --
    "earningsRevenueEstimateGrowth": "rev_estimate_growth", # revenue revision
    # -- Earnings proximity + metadata --
    "nextEarningsDate":          "next_earnings_date",      # binary event flag
    "sector":                    "sector",                  # sector concentration
    # -- Enhanced fundamentals --
    "buybackYield":              "buyback_yield",           # capital return %
    "epsGrowthQ":                "eps_growth_q",            # QoQ EPS growth (SUE supplement)
    "sbcByRevenue":              "sbc_by_revenue",          # stock comp dilution quality
    "pegRatio":                  "peg_ratio",               # growth-adjusted PE
}


class StockAnalysisClient(MClient):
    """Async client for stockanalysis.com screener data-point API.

    Fetches all configured metrics in parallel using a single shared session.
    No auth required — browser impersonation via curl_cffi.

    Usage:
        async with StockAnalysisClient(config.stockanalysis) as client:
            df = await client.fetch_screener_data()
            # df indexed by symbol, columns = snake_case metric names
    """

    def __init__(self, config: StockAnalysis) -> None:
        super().__init__(config)
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> "StockAnalysisClient":
        self._session = AsyncSession(impersonate="chrome124")
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def _fetch_one(self, sa_id: str) -> tuple[str, list[list]]:
        """Fetch a single data-point for the full universe."""
        assert self._session is not None, "client must be used as async context manager"
        url = f"{self._base_url}/_api/endpoints/screener/data-point"
        async with self._limiter:
            async with self._semaphore:
                resp = await self._session.get(url, params={"type": "s", "id": sa_id})
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != 200:
            raise RuntimeError(f"StockAnalysis returned status {data.get('status')} for id={sa_id!r}")
        rows = data["data"]["data"]
        if not rows:
            log.warning("StockAnalysis: empty response for id=%r", sa_id)
        return sa_id, rows

    async def fetch_screener_data(
        self,
        symbols: set[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch all metrics in parallel, return a DataFrame indexed by symbol.

        Args:
            symbols: Optional set of symbols to filter to. None = full universe.

        Returns:
            DataFrame with snake_case columns for every metric in _SA_DATA_POINTS.
            Rows with no data at all for a symbol are dropped.
        """
        results = await asyncio.gather(
            *[self._fetch_one(sa_id) for sa_id in _SA_DATA_POINTS],
            return_exceptions=True,
        )

        series: dict[str, pd.Series] = {}
        for sa_id, col_name in _SA_DATA_POINTS.items():
            result = results[list(_SA_DATA_POINTS).index(sa_id)]
            if isinstance(result, Exception):
                log.error("StockAnalysis fetch failed for id=%r: %s", sa_id, result)
                continue
            _, rows = result
            if not rows:
                continue
            s = pd.Series(
                {sym: val for sym, val in rows},
                name=col_name,
                dtype=object,
            )
            series[col_name] = s

        if not series:
            return pd.DataFrame()

        df = pd.DataFrame(series)
        df.index.name = "symbol"

        # Coerce numeric columns; leave analyst_ratings as str
        str_cols = {"analyst_ratings", "sector", "next_earnings_date"}
        for col in df.columns:
            if col not in str_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if symbols:
            df = df[df.index.isin(symbols)]

        log.info(
            "StockAnalysis: fetched %d metrics for %d symbols",
            len(df.columns), len(df),
        )
        return df
