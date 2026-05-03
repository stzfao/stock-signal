"""Pipeline orchestrator: universe → ingest → factors → score → rank → CSV."""

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
    # 2. SA screener data (pre-filter to reduce FMP targets)             #
    # ------------------------------------------------------------------ #
    screener_df = pd.DataFrame()
    async with StockAnalysisClient(config.stockanalysis) as sa_client:
        screener_df = await sa_client.fetch_screener_data(targets)
    log.info("SA screener: %d rows fetched", len(screener_df))

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
    # 4. Ingest — FMP (per symbol, filtered targets only)               #
    # ------------------------------------------------------------------ #
    result = await Swarm(config, targets, screener_df).run()
    log.info(
        "FMP ingest complete — refreshed=%d  skipped=%d  errors=%d",
        result.symbols_refreshed, result.symbols_skipped, len(result.errors),
    )

    # ------------------------------------------------------------------ #
    # 5. Load stored data                                                  #
    # ------------------------------------------------------------------ #
    symbols_list = sorted(targets)
    with Store(config.datastore.db_path, config.schedule) as store:
        prices_df    = store.load_prices(symbols_list)
        fin_df       = store.load_financials(symbols_list, period="FY")
        earnings_df  = store.load_earnings_surprises(symbols_list)

    # ------------------------------------------------------------------ #
    # 6. Compute factors                                                   #
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

    # revisions — blended from 3 SA screener signals
    if not screener_df.empty:
        rev_cols = ["eps_estimate_growth", "rev_estimate_growth", "price_target_change"]
        available = [c for c in rev_cols if c in screener_df.columns]
        if available:
            rev_blend = screener_df[available].reindex(symbols_list).mean(axis=1).dropna()
            if not rev_blend.empty:
                factors["revisions"] = rev_blend

    # guard factors from SA screener
    if not screener_df.empty:
        # mean_reversion_risk: % above SMA-200 (higher = more penalised via negative weight)
        if "ma_200_pct" in screener_df.columns:
            factors["mean_reversion_risk"] = screener_df["ma_200_pct"].reindex(symbols_list).dropna()

        # valuation_penalty: current PE / 3yr avg PE - 1 (stretch above own history)
        if "pe_ratio" in screener_df.columns and "pe_ratio_3y" in screener_df.columns:
            pe_now = screener_df["pe_ratio"].reindex(symbols_list)
            pe_3y = screener_df["pe_ratio_3y"].reindex(symbols_list)
            val_pen = (pe_now / pe_3y - 1).replace([float("inf"), float("-inf")], float("nan")).dropna()
            if not val_pen.empty:
                factors["valuation_penalty"] = val_pen

    # momentum_quality: Sharpe-like (1m return / 3m vol) from price data
    if not prices_df.empty:
        mq = momentum_quality(prices_df)
        if not mq.empty:
            factors["momentum_quality"] = mq

    if not factors:
        raise RuntimeError("No factors computed — check data ingestion")

    log.info("Computed %d factors: %s", len(factors), sorted(factors.keys()))
    for name, s in factors.items():
        log.debug("  %-22s %d symbols", name, s.notna().sum())

    # ------------------------------------------------------------------ #
    # 7. Score + rank                                                      #
    # ------------------------------------------------------------------ #
    scores = composite_score(factors)
    ranked = rank_top_decile(scores)

    # ------------------------------------------------------------------ #
    # 8. Enrich + write output                                             #
    # ------------------------------------------------------------------ #
    # Attach SA screener metadata for trade-readiness
    if not screener_df.empty:
        enrich_cols = {
            "sector": "sector",
            "ma_200_pct": "ma_200_pct",
            "rsi": "rsi",
            "next_earnings_date": "next_earnings_date",
        }
        for src, dst in enrich_cols.items():
            if src in screener_df.columns:
                ranked[dst] = ranked["symbol"].map(screener_df[src])

        # entry_zone: categorical signal based on trend position + momentum
        if "ma_200_pct" in ranked.columns and "rsi" in ranked.columns:
            ma = pd.to_numeric(ranked["ma_200_pct"], errors="coerce")
            rsi_val = pd.to_numeric(ranked["rsi"], errors="coerce")
            conditions = [
                (ma < -5),
                (ma.between(-5, 15)) & (rsi_val < 65),
                (ma > 10) & (rsi_val >= 75),
                (ma > 10) & (rsi_val < 75),
                ]
            choices = ["broken_trend", "pullback_entry", "extended", "trend_entry"]
            ranked["entry_zone"] = np.select(conditions, choices, default="neutral")

            # ── Earnings override ─────────────────────────────────────────
            if "next_earnings_date" in ranked.columns:
                earn = pd.to_datetime(ranked["next_earnings_date"], errors="coerce")
                today_ts = pd.Timestamp.now().normalize()
                days_to_earn = (earn - today_ts).dt.days

                # Hard block: ≤3 days to earnings — do not enter
                ranked.loc[days_to_earn <= 3, "entry_zone"] = "pre_earnings_avoid"

                # Soft flag: 4–21 days — preserve zone but flag it
                soft_mask = days_to_earn.between(4, 21) & ranked["entry_zone"].isin(
                    ["trend_entry", "pullback_entry"]
                )
                ranked.loc[soft_mask, "entry_zone"] = (
                        ranked.loc[soft_mask, "entry_zone"] + "_earn_risk"
                )
            # ─────────────────────────────────────────────────────────────

    config.datastore.output_dir.mkdir(parents=True, exist_ok=True)
    out = config.datastore.output_dir / f"us_ranked_{date.today().isoformat()}.csv"
    ranked.to_csv(out, index=False)
    log.info("Wrote %d ranked stocks → %s", len(ranked), out)
    return out
