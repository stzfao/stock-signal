"""Net Stock Issuance — Pontiff & Woodgate (2008). Share dilution predicts underperformance."""

import pandas as pd

def net_issuance(financials: pd.DataFrame) -> pd.Series:
    """Compute net issuance = (shares_t / shares_t-1) - 1.
    INVERTED: returns negated values so higher = better (buybacks = bullish).

    :param financials: DataFrame with columns [symbol, date, shares_outstanding],
                    at least 2 annual filings per symbol.
    :returns: Series indexed by symbol. Higher values = more buybacks = better.
    """
    df = financials.sort_values(["symbol", "date"])
    result: dict[str, float] = {}

    for symbol, group in df.groupby("symbol"):
        if len(group) < 2:
            continue

        curr = group["shares_outstanding"].iloc[-1]
        prev = group["shares_outstanding"].iloc[-2]

        if pd.isna(prev) or prev == 0 or pd.isna(curr):
            continue

        issuance = (curr / prev) - 1.0
        # negate: buybacks (so negative issuance) are bullish
        result[str(symbol)] = -issuance

    return pd.Series(result, dtype=float)
