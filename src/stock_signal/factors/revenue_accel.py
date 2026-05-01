"""Revenue Acceleration — 2nd derivative of revenue growth. Accelerating growth is bullish."""

import pandas as pd


def revenue_acceleration(financials: pd.DataFrame) -> pd.Series:
    """Compute revenue acceleration = revenue_growth_t - revenue_growth_t-1.

    Positive acceleration means growth is speeding up. This is the signal,
    NOT the growth level itself (which LSV 1994 showed is a glamour trap).

    Args:
        financials: DataFrame with columns [symbol, date, revenue],
                    at least 3 annual filings per symbol (need t, t-1, t-2).

    Returns:
        Series indexed by symbol. Higher = accelerating revenue = better.
    """
    df = financials.sort_values(["symbol", "date"])
    result: dict[str, float] = {}

    for symbol, group in df.groupby("symbol"):
        if len(group) < 3:
            continue

        rev_t = group.iloc[-1]["revenue"]
        rev_t1 = group.iloc[-2]["revenue"]
        rev_t2 = group.iloc[-3]["revenue"]

        if any(v is None or v == 0 for v in [rev_t, rev_t1, rev_t2]):
            continue

        growth_t = (rev_t / rev_t1) - 1.0
        growth_t1 = (rev_t1 / rev_t2) - 1.0

        # Acceleration = change in growth rate
        result[symbol] = growth_t - growth_t1

    return pd.Series(result, dtype=float)
