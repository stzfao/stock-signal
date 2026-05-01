"""Integration tests — hits live APIs, run with: pytest tests/test_integration.py -s -v"""
import json

import asyncio
import pytest
from src.stock_signal.config import Config, Scope
from src.stock_signal.core.clients import FMPClient, SlickChartsClient, NASDAQClient, NYSEClient, StockAnalysisClient
from src.stock_signal.core.universe import Universe
# from src.stock_signal.core.discovery import

_CONFIG = Config.new("src/stock_signal/config.toml")
SYMBOL = "NVDA"


# ---------------------------------------------------------------------------
# Universe clients
# ---------------------------------------------------------------------------

async def test_nasdaq_constituents():
    async with NASDAQClient(_CONFIG.nasdaq) as client:
        data = await client.get_nasdaq_constituents()
        print(f"\n NASDAQ constituents: {data["symbol"].tolist()}")
        print(f"\nNASDAQ data: {data.head()}")


async def test_sp500_constituents():
    async with SlickChartsClient(_CONFIG.slickcharts) as client:
        data = await client.get_sp500_constituents()
        print(f"\nS&P500 data: {data}")


async def test_nyse_constituents():
    async with NYSEClient(_CONFIG.nyse) as client:
        data = await client.get_nyse_constituents()
        print(f"\nNYSE data: {data}")


async def test_common_constituents():
    async with (
        SlickChartsClient(_CONFIG.slickcharts) as sp500_client,
        NYSEClient(_CONFIG.nyse) as nyse_client,
        NASDAQClient(_CONFIG.nasdaq) as nasdaq_client,
    ):
        sp500_set, nyse_set, nasdaq_df = await asyncio.gather(
            sp500_client.get_sp500_constituents(),
            nyse_client.get_nyse_constituents(),
            nasdaq_client.get_nasdaq_constituents(),
        )

    nasdaq_set = set(nasdaq_df["symbol"].tolist())

    print(f"\nS&P 500 : {len(sp500_set)} symbols")
    print(f"NYSE    : {len(nyse_set)} symbols")
    print(f"NASDAQ  : {len(nasdaq_set)} symbols")

    print(f"\nS&P 500 ∩ NYSE   : {len(sp500_set & nyse_set)}")
    print(f"S&P 500 ∩ NASDAQ : {len(sp500_set & nasdaq_set)}")
    print(f"NYSE    ∩ NASDAQ : {len(nyse_set & nasdaq_set)}")
    print(f"All three        : {len(sp500_set & nyse_set & nasdaq_set)}")

    sp500_only = sp500_set - nyse_set - nasdaq_set
    nyse_only  = nyse_set  - sp500_set - nasdaq_set
    nasdaq_only = nasdaq_set - sp500_set - nyse_set


    print(f"\nIn S&P 500 only  : {len(sp500_only)}  — sample: {sorted(sp500_only)[:5]}")
    print(f"In NYSE only     : {len(nyse_only)}   — sample: {sorted(nyse_only)[:5]}")
    print(f"In NASDAQ only   : {len(nasdaq_only)} — sample: {sorted(nasdaq_only)[:5]}")

@pytest.mark.parametrize("scope,expected_attr", [
    ([Scope.SP500.value],                          "sp500_constituents"),
    ([Scope.NASDAQ.value],                         "nasdaq_constituents"),
    ([Scope.NYSE.value],                           "nyse_constituents"),
    ([Scope.SP500.value, Scope.NASDAQ.value],      None),
    ([Scope.SP500.value, Scope.NYSE.value],        None),
    ([Scope.SP500.value, Scope.NASDAQ.value, Scope.NYSE.value], None),
    ([],                                           "sp500_constituents"),  # empty defaults to SP500
])
async def test_universe_fetch(scope, expected_attr):
    u = Universe(_CONFIG)
    await u.fetch_us_universe(scope)

    assert u.targets, f"targets empty for scope={scope!r}"

    if expected_attr == "sp500_constituents":
        assert len(u.sp500_constituents) > 0, "sp500_constituents not populated"
    elif expected_attr == "nasdaq_constituents":
        assert u.nasdaq_constituents is not None and len(u.nasdaq_constituents) > 0, \
            "nasdaq_constituents not populated"
    elif expected_attr == "nyse_constituents":
        assert len(u.nyse_constituents) > 0, "nyse_constituents not populated"

    print(f"\nscope={scope!r}  targets={len(u.targets)}"
          f"  sp500={len(u.sp500_constituents)}"
          f"  nasdaq={len(u.nasdaq_constituents) if u.nasdaq_constituents is not None else 0}"
          f"  nyse={len(u.nyse_constituents)}")


# ---------------------------------------------------------------------------
# StockAnalysis screener
# ---------------------------------------------------------------------------

async def test_stockanalysis_full_universe():
    async with StockAnalysisClient(_CONFIG.stockanalysis) as client:
        df = await client.fetch_screener_data()
    assert not df.empty, "expected non-empty DataFrame"
    assert "altman_z" in df.columns
    assert "f_score" in df.columns
    assert "price_target_change" in df.columns
    print(f"\nStockAnalysis: {len(df)} symbols × {len(df.columns)} metrics")
    print(df[["altman_z", "market_cap", "analyst_count", "f_score", "price_target_change"]].describe())


async def test_stockanalysis_filtered():
    symbols = {"NVDA", "AAPL", "MSFT", "TSLA", "GOOG"}
    async with StockAnalysisClient(_CONFIG.stockanalysis) as client:
        df = await client.fetch_screener_data(symbols=symbols)
    assert set(df.index) <= symbols
    print(f"\n{df}")


# ---------------------------------------------------------------------------
# FMP — all endpoints for NVDA
# ---------------------------------------------------------------------------

async def test_fmp_daily_prices():
    async with FMPClient(_CONFIG.fmp) as client:
        data = await client.get_daily_prices(SYMBOL)
        print(f"\ndaily_prices ({SYMBOL}): {len(data)} rows — latest: {data[0]}")


async def test_fmp_income_statement():
    async with FMPClient(_CONFIG.fmp) as client:
        data = await client.get_income_statement(SYMBOL)
        print(f"\nincome_statement ({SYMBOL}): {len(data)} periods — latest: {data[0]}")


async def test_fmp_balance_sheet():
    async with FMPClient(_CONFIG.fmp) as client:
        data = await client.get_balance_sheet(SYMBOL)
        print(f"\nbalance_sheet ({SYMBOL}): {len(data)} periods — latest: {data[0]}")


async def test_fmp_cash_flow():
    async with FMPClient(_CONFIG.fmp) as client:
        data = await client.get_cash_flow(SYMBOL)
        print(f"\ncash_flow ({SYMBOL}): {len(data)} periods — latest: {data[0]}")


async def test_fmp_earnings():
    async with FMPClient(_CONFIG.fmp) as client:
        data = await client.get_earnings(SYMBOL)
        print(f"\nearnings ({SYMBOL}): {len(data)} quarters — latest: {data[0]}")


# async def test_fmp_key_metrics():
#     async with FMPClient(_CONFIG.fmp) as client:
#         data = await client.get_key_metrics(SYMBOL)
#         print(f"\nkey_metrics ({SYMBOL}): {len(data)} periods — latest: {data[0]}")
#
#
# async def test_fmp_stock_screener():
#     async with FMPClient(_CONFIG.fmp) as client:
#         data = await client.get_stock_screener(marketCapMoreThan=500_000_000_000, limit=10)
#         print(f"\nstock_screener (mega cap): {len(data)} results — sample: {data[:2]}")
