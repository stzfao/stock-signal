"""Pipeline orchestrator: universe → ingest → factors → score → rank → CSV."""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from .config import Config
from .core.store import Store
from .core.swarm import Swarm
from .core.universe import Universe
from .factors import (
    accruals,
    asset_growth,
    gross_profitability,
    momentum_12_1,
    net_issuance,
    piotroski_fscore,
    proximity_52wk_high,
    revenue_acceleration,
    sue,
)
from .scoring import FilterThresholds, composite_score, rank_top_decile

log = logging.getLogger(__name__)

# SA screener column → (threshold, fill_value)
# fill_value = what to assume when data is missing (benefit of doubt = pass the filter)
_SA_FILTERS: list[tuple[str, float, float]] = [
    ("market_cap",       300_000_000, 0.0),   # < $300M → exclude
    ("altman_z",         1.8,         1.8),   # distressed → exclude; missing = neutral
    ("analyst_count",    3.0,         3.0),   # < 3 analysts → micro-cap noise
    ("price",            1.0,         1.0),   # penny stocks
    ("interest_coverage",1.5,         1.5),   # can't service debt
    ("dollar_volume",    1_000_000,   1e9),   # illiquid; missing = assume liquid
]


async def run_us_pipeline(
    config: Config,
    scope: list[str] | None = None,
    symbols: list[str] | None = None,
) -> Path:
    """Universe → ingest → factors → score → rank → CSV.

    Args:
        config:  Loaded Config object.
        scope:   Universe scope list (Scope enum values). Defaults to SP500.
        symbols: Override universe with explicit symbols (testing / ad-hoc).

    Returns:
        Path to the ranked output CSV.
    """
    # ------------------------------------------------------------------ #
    # 1. Universe                                                          #
    # ------------------------------------------------------------------ #
    if symbols:
        targets: set[str] = set(symbols)
    else:
        u = Universe(config)
        await u.fetch_us_universe(scope or [])
        targets = u.targets

    log.info("Universe: %d symbols", len(targets))

    # ------------------------------------------------------------------ #
    # 2. Ingest — FMP (per symbol) + StockAnalysis (full universe)        #
    # ------------------------------------------------------------------ #
    result = await Swarm(config, targets).run()
    screener_df = result.screener_df
    log.info(
        "Ingest complete — FMP refreshed=%d  skipped=%d  errors=%d  SA rows=%d",
        result.symbols_refreshed, result.symbols_skipped,
        len(result.errors), len(screener_df),
    )

    # ------------------------------------------------------------------ #
    # 3. Hard pre-filters from SA screener                                #
    # ------------------------------------------------------------------ #
    if not screener_df.empty:
        filt = screener_df.reindex(sorted(targets))
        mask = pd.Series(True, index=filt.index)
        for col, threshold, fill in _SA_FILTERS:
            if col in filt.columns:
                mask &= filt[col].fillna(fill) >= threshold
        targets = set(filt.index[mask])
        log.info("After SA filters: %d symbols pass", len(targets))

    if not targets:
        raise RuntimeError("All symbols filtered out — check SA data / thresholds")

    # ------------------------------------------------------------------ #
    # 4. Load stored data                                                  #
    # ------------------------------------------------------------------ #
    symbols_list = sorted(targets)
    with Store(config.datastore.db_path, config.schedule.staleness_days) as store:
        prices_df   = store.load_prices(symbols_list)
        fin_df      = store.load_financials(symbols_list, period="FY")
        earnings_df = store.load_earnings_surprises(symbols_list)

    # ------------------------------------------------------------------ #
    # 5. Compute factors                                                   #
    # ------------------------------------------------------------------ #
    factors: dict[str, pd.Series] = {}

    if not prices_df.empty:
        factors["momentum"]           = momentum_12_1(prices_df)
        factors["proximity_52wk_high"]= proximity_52wk_high(prices_df)

    if not fin_df.empty:
        factors["fscore"]              = piotroski_fscore(fin_df)
        factors["gross_profitability"] = gross_profitability(fin_df)
        factors["accruals"]            = accruals(fin_df)
        factors["asset_growth"]        = asset_growth(fin_df)
        factors["net_issuance"]        = net_issuance(fin_df)
        factors["revenue_acceleration"]= revenue_acceleration(fin_df)

    if not earnings_df.empty:
        factors["sue"] = sue(earnings_df)

    # revisions — sourced from SA screener (eps_estimate_growth = EPS revision breadth proxy)
    # FMP /analyst-estimates is paywalled; SA provides this for free
    if not screener_df.empty and "eps_estimate_growth" in screener_df.columns:
        factors["revisions"] = screener_df["eps_estimate_growth"].reindex(symbols_list).dropna()

    if not factors:
        raise RuntimeError("No factors computed — check data ingestion")

    log.info("Computed %d factors: %s", len(factors), sorted(factors.keys()))
    for name, s in factors.items():
        log.debug("  %-22s %d symbols", name, s.notna().sum())

    # ------------------------------------------------------------------ #
    # 6. Score + rank                                                      #
    # ------------------------------------------------------------------ #
    scores = composite_score(factors)
    ranked = rank_top_decile(scores)

    # ------------------------------------------------------------------ #
    # 7. Write output                                                      #
    # ------------------------------------------------------------------ #
    config.datastore.output_dir.mkdir(parents=True, exist_ok=True)
    out = config.datastore.output_dir / f"us_ranked_{date.today().isoformat()}.csv"
    ranked.to_csv(out, index=False)
    log.info("Wrote %d ranked stocks → %s", len(ranked), out)
    return out
