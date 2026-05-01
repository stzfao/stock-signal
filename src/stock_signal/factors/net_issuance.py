"""Net Stock Issuance — Pontiff & Woodgate (2008). Share dilution predicts underperformance."""

import pandas as pd


def net_issuance(financials: pd.DataFrame) -> pd.Series:
    """Compute net issuance = (shares_t / shares_t-1) - 1.

    INVERTED: returns negated values so higher = better (buybacks = bullish).

    Args:
        financials: DataFrame with columns [symbol, date, shares_outstanding],
                    at least 2 annual filings per symbol.

    Returns:
        Series indexed by symbol. Higher values = more buybacks = better.
    """
    df = financials.sort_values(["symbol", "date"])
    result: dict[str, float] = {}

    for symbol, group in df.groupby("symbol"):
        if len(group) < 2:
            continue

        curr = group.iloc[-1]["shares_outstanding"]
        prev = group.iloc[-2]["shares_outstanding"]

        if prev is None or prev == 0 or curr is None:
            continue

        issuance = (curr / prev) - 1.0
        # Negate: buybacks (negative issuance) are bullish
        result[symbol] = -issuance

    return pd.Series(result, dtype=float)
