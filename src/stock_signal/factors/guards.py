"""Guard factors sourced from SA screener data."""

import pandas as pd


def valuation_penalty(screener_df: pd.DataFrame, symbols: list[str]) -> pd.Series:
    """PE ratio relative to own 3-year average — penalises stocks stretched above history.

    :param screener_df: SA screener DataFrame indexed by symbol.
    :param symbols: Symbols to compute for (already-filtered target list).
    :returns: Series indexed by symbol: (pe_now / pe_3y) - 1. Positive = expensive vs history.
    """
    pe_now = screener_df["pe_ratio"].reindex(symbols)
    pe_3y  = screener_df["pe_ratio_3y"].reindex(symbols)
    result = (pe_now / pe_3y - 1).replace([float("inf"), float("-inf")], float("nan"))
    return result.dropna()
