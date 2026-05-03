"""Quality factor: Novy-Marx gross profitability."""

import pandas as pd


def gross_profitability(financials: pd.DataFrame) -> pd.Series:
    """Gross profitability = (Revenue - COGS) / Total Assets.
    Uses the most recent annual filing per symbol.

    NOTE: symbols with no cost_of_revenue (e.g. pure service firms) are
    excluded by the downstream dropna(). This is intentional — gross
    profitability is undefined without a cost structure to measure against.

    :param financials: DataFrame with columns [symbol, date, revenue,
                    cost_of_revenue, total_assets], period='annual'.
    :returns: Series indexed by symbol.
    """
    # take most recent filing per symbol
    latest = financials.sort_values("date").groupby("symbol").last()

    numerator = latest["revenue"] - latest["cost_of_revenue"]
    denominator = latest["total_assets"].replace(0, float("nan"))

    result = numerator / denominator
    return result.dropna()
