"""Standardized Unexpected Earnings (SUE) — Post-Earnings Announcement Drift signal."""

import pandas as pd
import numpy as np

def sue(earnings_surprises: pd.DataFrame) -> pd.Series:
    """Compute SUE for most recent earnings per symbol.

    SUE = (actual - estimate) / std_dev(surprise over trailing 8 quarters)

    :param earnings_surprises: DataFrame with columns [symbol, date,
                            actual_earnings_result, estimated_earnings],
                            sorted by (symbol, date).
    :returns: Series indexed by symbol with most recent quarter's SUE.
    """
    result: dict[str, float] = {}

    for symbol, group in earnings_surprises.groupby("symbol"):
        group = group.sort_values("date")
        group = group.dropna(subset=["actual_earnings_result", "estimated_earnings"])

        if len(group) < 6:
            continue

        # compute surprise for each quarter
        surprise = group["actual_earnings_result"] - group["estimated_earnings"]

        # use trailing 8 quarters (or whatever is available)
        trailing = surprise.tail(8)
        std = trailing.std()

        if pd.isna(std) or std == 0:
            continue

        latest_surprise = surprise.iloc[-1]
        result[str(symbol)] = float(np.clip(latest_surprise / std, -5.0, 5.0))

    return pd.Series(result, dtype=float)
