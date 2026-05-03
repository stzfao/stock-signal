"""Momentum factors: 12-1 momentum and 52-week-high proximity."""

import pandas as pd
import numpy as np


def momentum_12_1(prices: pd.DataFrame) -> pd.Series:
    """12-month return skipping the most recent month (Jegadeesh-Titman).

    :param prices: DataFrame with columns [symbol, date, adj_close],
                sorted by (symbol, date), covering at least 13 months.
    :returns: Series indexed by symbol with 12-1 momentum value.
    """
    # use date-based lookback
    price_by_sym: dict[str, float] = {}
    price_12m_by_sym: dict[str, float] = {}

    for symbol, group in prices.groupby("symbol"):
        # 252 + ~1 month buffer for the skip period
        if len(group) < 275:
            continue
        # price 1 month ago (skip last ~21 trading days)
        price_1m_val = group["adj_close"].iloc[-22]
        # price 12 months ago (~252 trading days)
        price_12m_val = group["adj_close"].iloc[-252]

        if price_12m_val > 0 and price_1m_val > 0 and not np.isnan(price_1m_val):
            price_by_sym[str(symbol)] = price_1m_val
            price_12m_by_sym[str(symbol)] = price_12m_val

    p1m = pd.Series(price_by_sym, dtype=float)
    p12m = pd.Series(price_12m_by_sym, dtype=float)

    return (p1m - p12m) / p12m


def proximity_52wk_high(prices: pd.DataFrame) -> pd.Series:
    """Current price as fraction of 52-week high (George-Hwang).

    :param prices: DataFrame with columns [symbol, date, adj_close],
                sorted by (symbol, date), covering at least 252 days.
    :returns: Series indexed by symbol, values in (0, 1].
    """
    result: dict[str, float] = {}
    for symbol, group in prices.groupby("symbol"):
        if len(group) < 252:
            continue
        last_252 = group["adj_close"].iloc[-252:]
        high_52wk = last_252.max()
        current = group["adj_close"].iloc[-1]
        if high_52wk > 0 and not np.isnan(current):
            result[str(symbol)] = current / high_52wk

    return pd.Series(result, dtype=float)
