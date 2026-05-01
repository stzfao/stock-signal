import asyncio
import logging
from dataclasses import dataclass, field

import pandas as pd

from .clients import StockAnalysisClient, FMPClient
from .store import Store
from ..config import Config

log = logging.getLogger(__name__)


@dataclass
class SwarmResult:
    screener_df: pd.DataFrame
    symbols_refreshed: int = 0
    symbols_skipped: int = 0
    errors: dict[str, Exception] = field(default_factory=dict)


class Swarm:
    def __init__(self, config: Config, targets: set[str]):
        self._config = config
        self.targets = targets

    async def run(self) -> SwarmResult:
        cfg = self._config
        refreshed = 0
        skipped = 0
        errors: dict[str, Exception] = {}

        with Store(cfg.datastore.db_path, cfg.schedule.staleness_days) as store:
            async with (
                StockAnalysisClient(cfg.stockanalysis) as sa_client,
                FMPClient(cfg.fmp) as fmp_client,
            ):
                sa_task = asyncio.create_task(
                    sa_client.fetch_screener_data(self.targets)
                )

                ordered = sorted(self.targets)
                fmp_results = await asyncio.gather(
                    *[self._ingest_symbol(fmp_client, store, sym) for sym in ordered],
                    return_exceptions=True,
                )

                for sym, result in zip(ordered, fmp_results):
                    if isinstance(result, Exception):
                        log.error("FMP ingest failed for %s: %s", sym, result)
                        errors[sym] = result
                    elif result:
                        refreshed += 1
                    else:
                        skipped += 1

                screener_df = await sa_task

        log.info(
            "Swarm complete — refreshed=%d  skipped=%d  errors=%d",
            refreshed, skipped, len(errors),
        )
        return SwarmResult(
            screener_df=screener_df,
            symbols_refreshed=refreshed,
            symbols_skipped=skipped,
            errors=errors,
        )

    async def _ingest_symbol(self, client: FMPClient, store: Store, symbol: str) -> bool:
        """Fetch and store all FMP data for one symbol. Returns True if refreshed."""
        if not store.is_stale("prices", symbol):
            return False

        prices, income, balance, cashflow, earnings = await asyncio.gather(
            client.get_daily_prices(symbol),
            client.get_income_statement(symbol, period="annual", limit=5),
            client.get_balance_sheet(symbol, period="annual", limit=5),
            client.get_cash_flow(symbol, period="annual", limit=5),
            client.get_earnings(symbol, limit=40),
        )

        store.upsert_prices(symbol, prices)
        store.upsert_financials(symbol, income, balance, cashflow)
        store.upsert_earnings_surprises(symbol, earnings)

        log.debug("FMP ingested: %s", symbol)
        return True
