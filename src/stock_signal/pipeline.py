"""Pipeline orchestrator: universe -> ingest -> factors -> score -> rank -> CSV."""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .core.clients import StockAnalysisClient
from .core.store import Store
from .core.swarm import Swarm
from .core.universe import Universe
from .factors import (
    accruals,
    asset_growth,
    gross_profitability,
    momentum_12_1,
    momentum_quality,
    net_issuance,
    piotroski_fscore,
    proximity_52wk_high,
    revenue_acceleration,
    sue,
    valuation_penalty,
)
from .core.scoring import composite_score, rank_top_decile, apply_earnings_override

log = logging.getLogger(__name__)

# SA screener column -> (threshold, fill_value)
# fill_value = what to assume when data is missing, to give
# the benefit of doubt. pass the filter in those cases.
_PRE_FILTERS: list[tuple[str, float, float]] = [
    ("market_cap",       300_000_000, 0.0),   # < $300M -> exclude
    ("altman_z",         1.8,         1.8),   # distressed -> exclude; missing = neutral
    ("analyst_count",    3.0,         3.0),   # < 3 analysts -> micro-cap noise
    ("price",            1.0,         1.0),   # penny stocks
    ("interest_coverage",1.5,         1.5),   # can't service debt
    ("dollar_volume",    1_000_000,   1e9),   # illiquid; missing = assume liquid
]


async def run_us_pipeline(
    config: Config,
    scope: list[str] | None = None,
    symbols: list[str] | None = None,
) -> Path:
    """Universe -> screen -> ingest -> factors -> score -> rank -> CSV.
    :param config: Loaded Config object.
    :param scope: Universe scope list (Scope enum values). Defaults to SP500.
    :param symbols: Override universe with explicit symbols (testing / ad-hoc).
    :returns: Path to the ranked output CSV.
    """
    # ------------------------------------------------------------------
    # 1. Universe: Fetch targets according to the universe(s) selected.
    # ------------------------------------------------------------------
    if symbols:
        targets: set[str] = set(symbols)
    else:
        u = Universe(config)
        await u.fetch_us_universe(scope or [])
        targets = u.targets

    log.info("Universe: %d symbols", len(targets))

    # ------------------------------------------------------------------
    # 2. SA screener data (pre-filter to reduce FMP targets)
    # ------------------------------------------------------------------
    screener_df = pd.DataFrame()
    async with StockAnalysisClient(config.stockanalysis) as sa_client:
        screener_df = await sa_client.fetch_screener_data(targets)
    log.info("SA screener: %d rows fetched", len(screener_df))

    # ------------------------------------------------------------------
    # 3. Hard pre-filters from SA screener
    # ------------------------------------------------------------------
    if screener_df.empty:
        raise RuntimeError("StockAnalysis screener returned empty.")
    
    pre_filter = screener_df.reindex(sorted(targets))
    mask = pd.Series(True, index=pre_filter.index)
    for col, threshold, fill in _PRE_FILTERS:
        if col in pre_filter.columns:
            mask &= pre_filter[col].fillna(fill) >= threshold
    targets = set(pre_filter.index[mask])
    log.info("After SA filters: %d symbols pass", len(targets))

    if not targets:
        raise RuntimeError("All symbols filtered out — check SA data / thresholds")

    # ------------------------------------------------------------------
    # 4. Ingest — FMP (per symbol, filtered targets only)
    # ------------------------------------------------------------------
    result = await Swarm(config, targets, screener_df).run()
    log.info(
        "FMP ingest complete — refreshed=%d  skipped=%d  errors=%d",
        result.symbols_refreshed, result.symbols_skipped, len(result.errors),
    )

    # ------------------------------------------------------------------
    # 5. Load stored data
    # ------------------------------------------------------------------
    symbols_list = sorted(targets)
    with Store(config.datastore.db_path, config.schedule) as store:
        prices_df    = store.load_prices(symbols_list)
        fin_df       = store.load_financials(symbols_list, period="FY")
        earnings_df  = store.load_earnings_surprises(symbols_list)

    # ------------------------------------------------------------------
    # 6. Compute factors
    # ------------------------------------------------------------------
    factors: dict[str, pd.Series] = {}

    if prices_df.empty:
        log.warning("No price data loaded — price-based factors skipped")
    else:
        factors["momentum"]            = momentum_12_1(prices_df)
        factors["proximity_52wk_high"] = proximity_52wk_high(prices_df)
        factors["momentum_quality"]    = momentum_quality(prices_df)

    if fin_df.empty:
        log.warning("No financials loaded — fundamental factors skipped")
    else:
        factors["fscore"]              = piotroski_fscore(fin_df)
        factors["gross_profitability"] = gross_profitability(fin_df)
        factors["accruals"]            = accruals(fin_df)
        factors["asset_growth"]        = asset_growth(fin_df)
        factors["net_issuance"]        = net_issuance(fin_df)
        factors["revenue_acceleration"]= revenue_acceleration(fin_df)

    if not earnings_df.empty:
        factors["sue"] = sue(earnings_df)

    # revisions: blend of 3 SA signals
    rev_cols = ["eps_estimate_growth", "rev_estimate_growth", "price_target_change"]
    rev_blend = screener_df[[c for c in rev_cols if c in screener_df.columns]].reindex(symbols_list).mean(axis=1).dropna()
    if not rev_blend.empty:
        factors["revisions"] = rev_blend

    # SA guard factors
    if "ma_200_pct" in screener_df.columns:
        factors["mean_reversion_risk"] = screener_df["ma_200_pct"].reindex(symbols_list).dropna()

    if {"pe_ratio", "pe_ratio_3y"}.issubset(screener_df.columns):
        factors["valuation_penalty"] = valuation_penalty(screener_df, symbols_list)

    if not factors:
        raise RuntimeError("No factors computed — check data ingestion")

    log.info("Computed %d factors: %s", len(factors), sorted(factors.keys()))
    for name, s in factors.items():
        log.debug("  %-22s  %d symbols", name, s.notna().sum())

    # ------------------------------------------------------------------
    # 7. Score + rank
    # ------------------------------------------------------------------
    scores = composite_score(factors)
    ranked = rank_top_decile(scores)

    # ------------------------------------------------------------------
    # 8. Enrich + write output
    # ------------------------------------------------------------------
    for col in ["sector", "ma_200_pct", "rsi", "next_earnings_date"]:
        if col in screener_df.columns:
            ranked[col] = ranked["symbol"].map(screener_df[col])

    ma      = pd.to_numeric(ranked.get("ma_200_pct"), errors="coerce")
    rsi_val = pd.to_numeric(ranked.get("rsi"), errors="coerce")
    ranked["entry_zone"] = np.select(
        [ma < -5, ma.between(-5, 15) & (rsi_val < 65), ma > 10 & (rsi_val >= 75), ma > 10 & (rsi_val < 75)],
        ["broken_trend", "pullback_entry", "extended", "trend_entry"],
        default="neutral",
    )
    ranked = apply_earnings_override(ranked)

    config.datastore.output_dir.mkdir(parents=True, exist_ok=True)
    out = config.datastore.output_dir / f"us_ranked_{date.today().isoformat()}.csv"
    ranked.to_csv(out, index=False)
    log.info("Wrote %d ranked stocks -> %s", len(ranked), out)
    return out
