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
    momentum_quality,
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
        sa = screener_df  # alias for readability

        # -- SA-sourced factors (precomputed, replaces FMP equivalents) ------
        # momentum 12-1: (1 + 12mo return) / (1 + 1mo return) - 1
        if {"ch_1y", "ch_1m"}.issubset(sa.columns):
            ch_1y = sa["ch_1y"].reindex(symbols_list) / 100
            ch_1m = sa["ch_1m"].reindex(symbols_list) / 100
            factors["momentum"] = ((1 + ch_1y) / (1 + ch_1m) - 1).dropna()

        if "position_in_range" in sa.columns:
            factors["proximity_52wk_high"] = (sa["position_in_range"] / 100).reindex(symbols_list).dropna()

        if "f_score" in sa.columns:
            factors["fscore"] = sa["f_score"].reindex(symbols_list).dropna()

        if "shares_yoy" in sa.columns:
            factors["net_issuance"] = (-sa["shares_yoy"]).reindex(symbols_list).dropna()  # buybacks = positive = good

        # revisions: blend of 3 SA signals
        rev_cols = [c for c in ["eps_estimate_growth", "rev_estimate_growth", "price_target_change"] if c in sa.columns]
        if rev_cols:
            rev_blend = sa[rev_cols].reindex(symbols_list).mean(axis=1).dropna()
            if not rev_blend.empty:
                factors["revisions"] = rev_blend

        # quality additions
        if "roic" in sa.columns:
            factors["roic"] = sa["roic"].reindex(symbols_list).dropna()

        if "fcf_yield" in sa.columns:
            factors["fcf_yield"] = sa["fcf_yield"].reindex(symbols_list).dropna()

        # guard factors
        if "ma_200_pct" in sa.columns:
            factors["mean_reversion_risk"] = sa["ma_200_pct"].reindex(symbols_list).dropna()

        if {"pe_ratio", "pe_ratio_3y"}.issubset(sa.columns):
            factors["valuation_penalty"] = valuation_penalty(sa, symbols_list)

        # -- FMP-computed factors (time-series data) --------
        if prices_df.empty:
            log.warning("No price data — momentum_quality skipped")
        else:
            factors["momentum_quality"] = momentum_quality(prices_df)

        if fin_df.empty:
            log.warning("No financials — FMP fundamental factors skipped")
        else:
            factors["gross_profitability"] = gross_profitability(fin_df)
            factors["accruals"]            = accruals(fin_df)
            factors["asset_growth"]        = asset_growth(fin_df)
            factors["revenue_acceleration"]= revenue_acceleration(fin_df)

        if not earnings_df.empty:
            factors["sue"] = sue(earnings_df)

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
        for col in ["sector", "ma_200_pct", "rsi", "next_earnings_date", "price"]:
            if col in screener_df.columns:
                ranked[col] = ranked["symbol"].map(screener_df[col])

        ma      = pd.to_numeric(ranked.get("ma_200_pct"), errors="coerce")
        rsi_val = pd.to_numeric(ranked.get("rsi"), errors="coerce")
        ranked["entry_zone"] = np.select(
            [ma < -5, ma.between(-5, 15) & (rsi_val < 65), ma > 10 & (rsi_val >= 75), ma > 10 & (rsi_val < 75)],
            ["broken_trend", "pullback_entry", "extended", "trend_entry"],
            default="neutral",
        )
        ranked.loc[ma.isna(), "entry_zone"] = "insufficient_data"
        ranked = apply_earnings_override(ranked)

        # -- days_in_zone from score history --
        days_df = store.load_days_in_top(lookback_days=30)
        if not days_df.empty:
            ranked = ranked.merge(days_df, on="symbol", how="left")
            ranked["days_in_top"] = ranked["days_in_top"].fillna(0).astype(int)
        else:
            ranked["days_in_top"] = 0

        # entry price bands
        ranked["entry_price_min"] = np.nan
        ranked["entry_price_max"] = np.nan
        price = pd.to_numeric(ranked.get("price"), errors="coerce")

        pullback = ranked["entry_zone"].str.startswith("pullback_entry")
        ranked.loc[pullback, "entry_price_min"] = (price[pullback] * 0.97).round(2)
        ranked.loc[pullback, "entry_price_max"] = (price[pullback] * 1.03).round(2)

        trend = ranked["entry_zone"].str.startswith("trend_entry")
        ranked.loc[trend, "entry_price_min"] = (price[trend] * 0.98).round(2)
        ranked.loc[trend, "entry_price_max"] = (price[trend] * 1.05).round(2)

        broken = ranked["entry_zone"] == "broken_trend"
        ranked.loc[broken, "entry_price_min"] = (price[broken] * 0.95).round(2)
        ranked.loc[broken, "entry_price_max"] = (price[broken] * 1.02).round(2)

        config.datastore.output_dir.mkdir(parents=True, exist_ok=True)
        universe = '-'.join(symbols) if symbols else '-'.join(scope)
        out = config.datastore.output_dir / f"us_ranked_{universe}_{date.today().isoformat()}.csv"
        ranked.to_csv(out, index=False)
        log.info("Wrote %d ranked stocks -> %s", len(ranked), out)

        store.save_score_history(date.today(), ranked)
        log.info("Saved %d scores to history", len(ranked))

    return out
