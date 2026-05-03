"""Scoring pipeline: winsorize, z-score, composite, filters, rank."""

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class FactorWeights:
    """Default weights for US-long track composite scoring.
    Weights auto-normalize to sum to 1.0 across active factors.
    """
    # tier 0 — original signals
    momentum: float = 0.20
    revisions: float = 0.20
    sue: float = 0.12
    fscore: float = 0.10
    gross_profitability: float = 0.08
    proximity_52wk_high: float = 0.08

    # tier 1 — new fundamental signals
    accruals: float = 0.07
    asset_growth: float = 0.05
    net_issuance: float = 0.05
    revenue_acceleration: float = 0.05

    # tier 2 — valuation & mean-reversion guards
    mean_reversion_risk: float = -0.08
    valuation_penalty: float = -0.07
    momentum_quality: float = 0.05


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip values at the given percentiles."""
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def cross_sectional_zscore(s: pd.Series) -> pd.Series:
    """Standardize: (x - mean) / std across all symbols."""
    mean = s.mean()
    std = s.std()
    if std == 0 or np.isnan(std):
        return s * 0.0
    return (s - mean) / std


def composite_score(
    factor_series: dict[str, pd.Series],
    weights: FactorWeights | None = None,
) -> pd.Series:
    """Winsorize, z-score, then compute weighted sum across factors.
    :param factor_series: Dict mapping factor name to Series indexed by symbol.
                       Keys must match FactorWeights field names.
    :param weights: Factor weights. Defaults to FactorWeights().
    :returns: Series indexed by symbol with composite z-score.
    """
    if weights is None:
        weights = FactorWeights()

    weight_map = asdict(weights)

    # weight_map = {
    #     "momentum": weights.momentum,
    #     "revisions": weights.revisions,
    #     "sue": weights.sue,
    #     "fscore": weights.fscore,
    #     "gross_profitability": weights.gross_profitability,
    #     "proximity_52wk_high": weights.proximity_52wk_high,
    #     "accruals": weights.accruals,
    #     "asset_growth": weights.asset_growth,
    #     "net_issuance": weights.net_issuance,
    #     "revenue_acceleration": weights.revenue_acceleration,
    #     "mean_reversion_risk": weights.mean_reversion_risk,
    #     "valuation_penalty": weights.valuation_penalty,
    #     "momentum_quality": weights.momentum_quality,
    # }

    # align all factors to a common index
    all_symbols: set[str] = set()
    for s in factor_series.values():
        all_symbols.update(s.index)
    index = pd.Index(sorted(all_symbols))

    # normalize weights across available factors
    # (abs sum preserves sign)
    active_weights = {
        name: weight_map[name]
        for name in factor_series
        if name in weight_map and weight_map[name] != 0
    }
    total_abs = sum(abs(w) for w in active_weights.values())
    if total_abs == 0:
        return pd.Series(dtype=float)
    normalized = {name: w / total_abs for name, w in active_weights.items()}

    composite = pd.Series(0.0, index=index)

    for name, raw in factor_series.items():
        w = normalized.get(name, 0.0)
        if w == 0:
            continue
        aligned = raw.reindex(index)
        winsorized = winsorize(aligned.dropna())
        zscored = cross_sectional_zscore(winsorized)
        composite = composite.add(zscored * w, fill_value=0.0)

    return composite


@dataclass
class FilterThresholds:
    """Hard exclusion thresholds."""

    min_market_cap: float = 300_000_000  # $300M
    min_adv: float = 1_000_000  # $1M average daily volume (dollar)
    min_altman_z: float = 1.8


def apply_hard_filters(
    scores: pd.Series,
    metrics: pd.DataFrame,
    thresholds: FilterThresholds | None = None,
) -> pd.Series:
    """Drop stocks failing hard exclusion criteria.
    :param scores: Composite scores indexed by symbol.
    :param metrics: DataFrame indexed by symbol with columns:
                 market_cap, avg_daily_volume, altman_z.
                 Missing columns are skipped (no filter applied).
    :param thresholds: Exclusion thresholds.
    :returns Filtered scores.
    """
    if thresholds is None:
        thresholds = FilterThresholds()

    mask = pd.Series(True, index=scores.index)

    if "market_cap" in metrics.columns:
        aligned = metrics["market_cap"].reindex(scores.index)
        mask &= aligned.fillna(0) >= thresholds.min_market_cap

    if "avg_daily_volume" in metrics.columns:
        aligned = metrics["avg_daily_volume"].reindex(scores.index)
        mask &= aligned.fillna(0) >= thresholds.min_adv

    if "altman_z" in metrics.columns:
        aligned = metrics["altman_z"].reindex(scores.index)
        # if altman_z is missing, don't exclude (benefit of doubt)
        mask &= aligned.fillna(thresholds.min_altman_z) >= thresholds.min_altman_z

    return scores[mask]


def apply_earnings_override(ranked: pd.DataFrame) -> pd.DataFrame:
    """Overrides entry_zone based on proximity to next earnings date.

    Hard block  (≤3 days)  : entry_zone → "pre_earnings_avoid"
    Soft flag   (4–21 days): appends "_earn_risk" to trend_entry / pullback_entry

    Requires columns: entry_zone, next_earnings_date.
    No-ops silently if either column is absent.
    """
    if "next_earnings_date" not in ranked.columns or "entry_zone" not in ranked.columns:
        return ranked

    earn = pd.to_datetime(ranked["next_earnings_date"], errors="coerce")
    days = (earn - pd.Timestamp.now().normalize()).dt.days

    ranked = ranked.copy()
    ranked.loc[days <= 3, "entry_zone"] = "pre_earnings_avoid"

    soft = days.between(4, 21) & ranked["entry_zone"].isin(["trend_entry", "pullback_entry"])
    ranked.loc[soft, "entry_zone"] = ranked.loc[soft, "entry_zone"] + "_earn_risk"

    return ranked


def rank_top_decile(scores: pd.Series, top_pct: float = 0.10) -> pd.DataFrame:
    """Extract top decile of scored symbols.
    :returns: DataFrame with columns [symbol, composite_score, rank],
            sorted by composite_score descending.
    """
    n = max(1, int(len(scores) * top_pct))
    top = scores.nlargest(n)
    df = pd.DataFrame({
        "symbol": top.index,
        "composite_score": top.values,
        "rank": range(1, len(top) + 1),
    })
    return df.reset_index(drop=True)
