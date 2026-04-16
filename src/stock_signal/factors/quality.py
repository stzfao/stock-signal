"""Quality factor: Novy-Marx gross profitability."""

import pandas as pd


def gross_profitability(financials: pd.DataFrame) -> pd.Series:
    """Gross profitability = (Revenue - COGS) / Total Assets.

    Uses the most recent annual filing per symbol.

    Args:
        financials: DataFrame with columns [symbol, date, revenue,
                    cost_of_revenue, total_assets], period='annual'.

    Returns:
        Series indexed by symbol.
    """
    # Take most recent filing per symbol
    latest = financials.sort_values("date").groupby("symbol").last()

    numerator = latest["revenue"] - latest["cost_of_revenue"]
    denominator = latest["total_assets"].replace(0, float("nan"))

    result = numerator / denominator
    return result.dropna()
