"""Universe management — S&P 500 + Nasdaq 100 ticker lists."""

import logging

from stock_signal.client import FMPClient

logger = logging.getLogger(__name__)


async def fetch_us_universe(client: FMPClient) -> list[str]:
    """Fetch S&P 500 + Nasdaq 100 constituents from FMP, deduplicated."""
    sp500 = await client.get_sp500_constituents()
    nasdaq = await client.get_nasdaq_constituents()

    symbols: set[str] = set()
    for entry in sp500:
        if sym := entry.get("symbol"):
            symbols.add(sym)
    for entry in nasdaq:
        if sym := entry.get("symbol"):
            symbols.add(sym)

    logger.info("US universe: %d symbols (S&P500=%d, Nasdaq=%d)", len(symbols), len(sp500), len(nasdaq))
    return sorted(symbols)
