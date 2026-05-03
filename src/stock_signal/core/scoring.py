"""Scoring pipeline: winsorize, z-score, composite, filters, rank."""

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class FactorWeights:
    """Factor weights for composite scoring. Signs encode direction: positive = higher is better,
    negative = higher is penalised. Weights are normalised by absolute sum inside composite_score().
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

    # tier 2 — quality additions (SA-sourced)
    roic: float = 0.05
    fcf_yield: float = 0.05

    # tier 3 — valuation & mean-reversion guards
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
    weight_sum = pd.Series(0.0, index=index)

    for name, raw in factor_series.items():
        w = normalized.get(name, 0.0)
        if w == 0:
            continue
        aligned = raw.reindex(index)
        winsorized = winsorize(aligned.dropna())
        zscored = cross_sectional_zscore(winsorized).reindex(index)

        has_data = zscored.notna()
        composite = composite.add(zscored.fillna(0.0) * w, fill_value=0.0)
        weight_sum += has_data.astype(float) * abs(w)

    # rescale by realised weight so all symbols are comparable
    nonzero = weight_sum > 0
    composite[nonzero] = composite[nonzero] / weight_sum[nonzero]
    composite[~nonzero] = np.nan

    return composite


def apply_earnings_override(ranked: pd.DataFrame) -> pd.DataFrame:
    """Overrides entry_zone based on proximity to next earnings date.

    Hard block  (<=3 days)  : entry_zone -> "pre_earnings_avoid"
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
