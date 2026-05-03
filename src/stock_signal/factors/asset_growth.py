"""Asset Growth factor — Cooper, Gulen & Schill (2008). High asset growth predicts underperformance."""

import pandas as pd
import numpy as np

def asset_growth(financials: pd.DataFrame) -> pd.Series:
    """Compute YoY asset growth = (total_assets_t / total_assets_t-1) - 1.

    INVERTED: returns negated values so higher = better (low asset growth = bullish).

    :param financials: DataFrame with columns [symbol, date, total_assets],
                    at least 2 annual filings per symbol.
    :returns: Series indexed by symbol. Higher values = lower asset growth = better.
    """
    df = financials.sort_values(["symbol", "date"])
    result: dict[str, float] = {}

    for symbol, group in df.groupby("symbol"):
        if len(group) < 2:
            continue

        curr = group["total_assets"].iloc[-1]
        prev = group["total_assets"].iloc[-2]

        if pd.isna(prev) or prev == 0 or pd.isna(curr):
            continue

        # a company that doubles assets in a year via acquisition
        # will be heavily penalised (-1.0); but a company going
        # millions -> billions will score -9999. clip it.
        growth = np.clip((curr / prev) - 1.0, -2.0, 2.0)

        # negate: low asset growth is bullish
        result[str(symbol)] = -growth

    return pd.Series(result, dtype=float)
