"""EPS revision breadth — analyst estimate revision direction proxy."""

import pandas as pd


def eps_revision_breadth(
    analyst_estimates: pd.DataFrame, window_days: int = 90
) -> pd.Series:
    """Fraction of estimate revisions that are upward over trailing window.

    FMP doesn't provide per-analyst data, so we use month-over-month changes
    in consensus estimated_eps_avg as a proxy for revision direction.

    Breadth = (# up revisions - # down revisions) / total revisions

    :param analyst_estimates: DataFrame with [symbol, date, estimated_eps_avg],
                          sorted by (symbol, date).
    :param window_days: Lookback window in calendar days.
    :returns: Series indexed by symbol, values in [-1, 1].
    """
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=window_days)
    result: dict[str, float] = {}

    for symbol, group in analyst_estimates.groupby("symbol"):
        group = group.sort_values("date")
        group = group.dropna(subset=["estimated_eps_avg"])

        # ensure date is comparable Timestamp type
        group = group.copy()
        group["date"] = pd.to_datetime(group["date"])
        window_data = group[group["date"] >= cutoff]
        if len(window_data) < 2:
            continue

        # compute changes between consecutive estimates
        changes = window_data["estimated_eps_avg"].diff().dropna()

        if len(changes) == 0:
            continue

        up = (changes > 0).sum()
        down = (changes < 0).sum()
        total = up + down

        if total == 0:
            result[str(symbol)] = 0.0
        else:
            result[str(symbol)] = (up - down) / total

    return pd.Series(result, dtype=float)
