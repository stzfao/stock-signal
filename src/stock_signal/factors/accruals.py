"""Accruals factor — Sloan (1996). High accruals predict underperformance."""

import pandas as pd


def accruals(financials: pd.DataFrame) -> pd.Series:
    """Compute accruals = (net_income - operating_cash_flow) / total_assets.

    INVERTED: returns negated values so higher = better (low accruals = bullish).

    :param financials: DataFrame with columns [symbol, date, net_income,
                    operating_cash_flow, total_assets], period='FY'.
    :returns: Series indexed by symbol. Higher values = lower accruals = better.
    """
    latest = financials.sort_values("date").groupby("symbol").last()

    ni = latest["net_income"]
    cfo = latest["operating_cash_flow"]
    ta = latest["total_assets"].replace(0, float("nan"))

    raw = (ni - cfo) / ta

    # negate: low accruals (cash earnings >> reported) is bullish
    return (-raw).dropna()
