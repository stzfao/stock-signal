"""Universe management — S&P 500 + Nasdaq constituents from local CSVs."""

import asyncio
import logging

from ..config import Config, Scope
from .clients import NASDAQClient, NYSEClient, SlickChartsClient
from enum import Enum
from datetime import date
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


class Universe:
    def __init__(self, config: Config):
        self._config = config
        self.nasdaq_constituents = None
        self.nyse_constituents = set()
        self.sp500_constituents = set()
        self.targets = set()

    async def _fetch_nasdaq_constituents(self):
        async with NASDAQClient(self._config.nasdaq) as client:
            try:
                self.nasdaq_constituents = await client.get_nasdaq_constituents()
            except RuntimeError as exc:
                log.error(f"failed to get NASDAQ constituents: {exc}")


    async def _fetch_nyse_constituents(self):
        async with NYSEClient(self._config.nyse) as client:
            try:
                self.nyse_constituents = await client.get_nyse_constituents()
            except RuntimeError as exc:
                log.error(f"failed to get NYSE constituents: {exc}")


    async def _fetch_sp500_constituents(self):
        async with SlickChartsClient(self._config.slickcharts) as client:
            try:
                self.sp500_constituents = await client.get_sp500_constituents()
            except RuntimeError as exc:
                log.error(f"failed to get S&P500 constituents: {exc}")

    def _coalesce_targets(self):
        targets: set[str] = set(self.nasdaq_constituents["symbol"].tolist()) if self.nasdaq_constituents is not None else set()
        targets.update(self.nyse_constituents, self.sp500_constituents)
        self.targets = targets

    async def fetch_us_universe(self, scope: list[str]):
        """Fetch all requested universe lists in parallel, then coalesce.
        :param scope: members of the Enum: Scope.
        """
        if not scope:
            scope = [Scope.SP500.value]

        fetch_map = {
            Scope.SP500.value:   self._fetch_sp500_constituents,
            Scope.NASDAQ.value:  self._fetch_nasdaq_constituents,
            Scope.NYSE.value:    self._fetch_nyse_constituents,
        }
        tasks = [fetch_map[s]() for s in scope if s in fetch_map]
        await asyncio.gather(*tasks)

        log.info(
            "US universe fetched — S&P500: %d, NASDAQ: %d, NYSE: %d",
            len(self.sp500_constituents),
            len(self.nasdaq_constituents) if self.nasdaq_constituents is not None else 0,
            len(self.nyse_constituents),
        )
        self._coalesce_targets()

    # async def join

#
# _DATA_DIR = Path(__file__).resolve().parents[2] / "data"
# _SP500_CSV = _DATA_DIR / "sp500.csv"
# _NASDAQ_CSV = _DATA_DIR / "nasdaq.csv"
#
#
# def get_sp500_at_date(as_of: date | None = None) -> list[str]:
#     """Return S&P 500 constituents as of `as_of` date (default: most recent).
#
#     sp500.csv format: date,tickers  where tickers is comma-separated string.
#     """
#     df = pd.read_csv(_SP500_CSV, parse_dates=["date"])
#     df = df.sort_values("date")
#
#     if as_of is not None:
#         df = df[df["date"].dt.date <= as_of]
#
#     if df.empty:
#         raise ValueError(f"No S&P 500 data on or before {as_of}")
#
#     tickers_str = df.iloc[-1]["tickers"]
#     tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
#     return tickers
#
#
# def get_nasdaq_current() -> list[str]:
#     """Return current Nasdaq listings from nasdaq.csv.
#
#     nasdaq.csv format: Symbol,Security Name
#     """
#     df = pd.read_csv(_NASDAQ_CSV)
#     # Column may be 'Symbol' or 'symbol'
#     col = next((c for c in df.columns if c.lower() == "symbol"), None)
#     if col is None:
#         raise ValueError(f"No 'Symbol' column in {_NASDAQ_CSV}. Columns: {list(df.columns)}")
#     return [s.strip() for s in df[col].dropna().tolist() if str(s).strip()]
#
#
# def fetch_us_universe(scope: str = "sp500", as_of: date | None = None) -> list[str]:
#     """Return deduplicated universe from local CSVs.
#
#     Args:
#         scope: "sp500" (503 symbols), "nasdaq" (Nasdaq only), or "full" (union, ~5700).
#         as_of: Point-in-time date for S&P 500 constituents (backtest use).
#                None = most recent available row.
#     """
#     if scope == "sp500":
#         symbols = sorted(set(get_sp500_at_date(as_of)))
#         log.info("US universe (S&P500): %d symbols", len(symbols))
#     elif scope == "nasdaq":
#         symbols = sorted(set(get_nasdaq_current()))
#         log.info("US universe (Nasdaq): %d symbols", len(symbols))
#     elif scope == "full":
#         sp500 = get_sp500_at_date(as_of)
#         nasdaq = get_nasdaq_current()
#         symbols = sorted(set(sp500) | set(nasdaq))
#         log.info(
#             "US universe (full): %d symbols (S&P500=%d, Nasdaq=%d)",
#             len(symbols), len(sp500), len(nasdaq),
#         )
#     else:
#         raise ValueError(f"Unknown universe scope: {scope!r}. Choose sp500 / nasdaq / full.")
#
#     return symbols
