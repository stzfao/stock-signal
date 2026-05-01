"""Asset Growth factor — Cooper, Gulen & Schill (2008). High asset growth predicts underperformance."""

import pandas as pd


def asset_growth(financials: pd.DataFrame) -> pd.Series:
    """Compute YoY asset growth = (total_assets_t / total_assets_t-1) - 1.

    INVERTED: returns negated values so higher = better (low asset growth = bullish).

    Args:
        financials: DataFrame with columns [symbol, date, total_assets],
                    at least 2 annual filings per symbol.

    Returns:
        Series indexed by symbol. Higher values = lower asset growth = better.
    """
    df = financials.sort_values(["symbol", "date"])
    result: dict[str, float] = {}

    for symbol, group in df.groupby("symbol"):
        if len(group) < 2:
            continue

        curr = group.iloc[-1]["total_assets"]
        prev = group.iloc[-2]["total_assets"]

        if prev is None or prev == 0 or curr is None:
            continue

        growth = (curr / prev) - 1.0
        # Negate: low asset growth is bullish
        result[symbol] = -growth

    return pd.Series(result, dtype=float)
