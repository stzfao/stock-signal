"""Revenue Acceleration — 2nd derivative of revenue growth. Accelerating growth is bullish."""

import pandas as pd
import numpy as np


def revenue_acceleration(financials: pd.DataFrame) -> pd.Series:
    """Compute revenue acceleration = revenue_growth_t - revenue_growth_t-1.

    Positive acceleration means growth is speeding up.
    Second derivative framing: Chan, L. K. C., Karceski, J., & Lakonishok,
    J. (1996). Momentum Strategies. Journal of Finance, 51(5), 1681-1713

    :param financials: DataFrame with columns [symbol, date, revenue],
                    at least 3 annual filings per symbol (need t, t-1, t-2).
    :returns: Series indexed by symbol. Higher = accelerating revenue = better.
    """
    df = financials.sort_values(["symbol", "date"])
    result: dict[str, float] = {}

    for symbol, group in df.groupby("symbol"):
        if len(group) < 3:
            continue

        rev_t = group["revenue"].iloc[-1]
        rev_t1 = group["revenue"].iloc[-2]
        rev_t2 = group["revenue"].iloc[-3]

        if any(pd.isna(v) or v == 0 for v in [rev_t, rev_t1, rev_t2]):
            continue

        # clipping so if rev_t1 is differential growth doesn't blow up
        growth_t  = np.clip((rev_t  / rev_t1) - 1.0, -2.0, 2.0)
        growth_t1 = np.clip((rev_t1 / rev_t2) - 1.0, -2.0, 2.0)

        # acceleration = change in growth rate within last period
        result[str(symbol)] = growth_t - growth_t1

    return pd.Series(result, dtype=float)
