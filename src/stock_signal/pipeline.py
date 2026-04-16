"""Pipeline orchestrator: ingest → compute factors → score → rank → CSV."""

import asyncio
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from stock_signal.client import FMPClient
from stock_signal.config import OUTPUT_DIR
from stock_signal.factors.fscore import piotroski_fscore
from stock_signal.factors.momentum import momentum_12_1, proximity_52wk_high
from stock_signal.factors.quality import gross_profitability
from stock_signal.factors.revisions import eps_revision_breadth
from stock_signal.factors.sue import sue
from stock_signal.scoring import apply_hard_filters, composite_score, rank_top_decile
from stock_signal.store import (
    get_connection,
    is_stale,
    load_analyst_estimates,
    load_earnings_surprises,
    load_financials,
    load_prices,
    upsert_analyst_estimates,
    upsert_earnings_surprises,
    upsert_financials,
    upsert_prices,
)
from stock_signal.universe import fetch_us_universe

logger = logging.getLogger(__name__)


async def _ingest_symbol(client: FMPClient, conn, symbol: str, force: bool = False) -> None:
    """Fetch all data for a single symbol if stale or forced."""
    tasks = []

    if force or is_stale(conn, "prices", symbol):
        tasks.append(("prices", client.get_daily_prices(symbol)))

    if force or is_stale(conn, "financials", symbol):
        tasks.append(("income", client.get_income_statement(symbol)))
        tasks.append(("balance", client.get_balance_sheet(symbol)))
        tasks.append(("cashflow", client.get_cash_flow(symbol)))

    if force or is_stale(conn, "earnings_surprises", symbol):
        tasks.append(("earnings", client.get_earnings_surprises(symbol)))

    if force or is_stale(conn, "analyst_estimates", symbol):
        tasks.append(("estimates", client.get_analyst_estimates(symbol)))

    if not tasks:
        return

    # Gather all fetches for this symbol
    results = await asyncio.gather(*(t[1] for t in tasks), return_exceptions=True)

    data: dict[str, list[dict]] = {}
    for (name, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            logger.warning("Failed to fetch %s for %s: %s", name, symbol, result)
            data[name] = []
        else:
            data[name] = result

    # Upsert into DuckDB
    if "prices" in data:
        upsert_prices(conn, symbol, data["prices"])

    if "income" in data:
        upsert_financials(conn, symbol, data.get("income", []), data.get("balance", []), data.get("cashflow", []))

    if "earnings" in data:
        upsert_earnings_surprises(conn, symbol, data["earnings"])

    if "estimates" in data:
        upsert_analyst_estimates(conn, symbol, data["estimates"])


async def run_us_pipeline(refresh: bool = False, symbols: list[str] | None = None) -> Path:
    """Full US pipeline: ingest → compute factors → score → rank → CSV.

    Args:
        refresh: Force re-fetch all data regardless of staleness.
        symbols: Override universe with specific symbols (for testing).

    Returns:
        Path to output CSV.
    """
    conn = get_connection()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with FMPClient() as client:
        # 1. Get universe
        if symbols is None:
            symbols = await fetch_us_universe(client)
        logger.info("Processing %d symbols", len(symbols))

        # 2. Ingest data
        for i, sym in enumerate(symbols):
            if (i + 1) % 50 == 0:
                logger.info("Ingesting %d/%d: %s", i + 1, len(symbols), sym)
            await _ingest_symbol(client, conn, sym, force=refresh)

    # 3. Load data from DuckDB
    logger.info("Loading data from DuckDB...")
    prices_df = load_prices(conn, symbols)
    financials_df = load_financials(conn, symbols, period="annual")
    earnings_df = load_earnings_surprises(conn, symbols)
    estimates_df = load_analyst_estimates(conn, symbols)

    # 4. Compute factors
    logger.info("Computing factors...")
    factors: dict[str, pd.Series] = {}

    if not prices_df.empty:
        factors["momentum"] = momentum_12_1(prices_df)
        factors["proximity_52wk_high"] = proximity_52wk_high(prices_df)

    if not financials_df.empty:
        factors["fscore"] = piotroski_fscore(financials_df)
        factors["gross_profitability"] = gross_profitability(financials_df)

    if not earnings_df.empty:
        factors["sue"] = sue(earnings_df)

    if not estimates_df.empty:
        factors["revisions"] = eps_revision_breadth(estimates_df)

    if not factors:
        logger.error("No factors computed — check data ingestion")
        conn.close()
        raise RuntimeError("No factors computed")

    logger.info("Computed %d factors: %s", len(factors), list(factors.keys()))
    for name, series in factors.items():
        logger.info("  %s: %d symbols", name, len(series))

    # 5. Composite score
    scores = composite_score(factors)

    # 6. Apply hard filters (build metrics df from available data)
    # For now, compute avg daily volume from prices
    metrics_data: dict[str, dict] = {}
    if not prices_df.empty:
        for sym, group in prices_df.groupby("symbol"):
            recent = group.tail(30)
            adv = (recent["close"] * recent["volume"]).mean() if not recent.empty else 0
            metrics_data[sym] = {"avg_daily_volume": adv}

    if metrics_data:
        metrics_df = pd.DataFrame.from_dict(metrics_data, orient="index")
        scores = apply_hard_filters(scores, metrics_df)

    # 7. Rank top decile
    ranked = rank_top_decile(scores)

    # 8. Write output
    today = date.today().isoformat()
    output_path = OUTPUT_DIR / f"us_ranked_{today}.csv"
    ranked.to_csv(output_path, index=False)
    logger.info("Wrote %d ranked stocks to %s", len(ranked), output_path)

    conn.close()
    return output_path
