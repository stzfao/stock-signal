"""Momentum quality factor: Sharpe-like return/volatility signal."""

import numpy as np
import pandas as pd


def momentum_quality(prices_df: pd.DataFrame) -> pd.Series:
    """Sharpe-like ratio: 1-month return / 3-month annualised volatility.
    Rewards smooth, low-vol momentum; punishes spike-then-fade patterns.

    :param prices_df: DataFrame with columns [symbol, date, close], sorted by date.
    :returns: Series indexed by symbol with momentum quality scores.
    """
    results: dict[str, float] = {}

    for symbol, grp in prices_df.groupby("symbol"):
        grp = grp.sort_values("date")
        close = grp["close"].dropna().reset_index(drop=True)

        # need ~3 months of data
        if len(close) < 63:
            continue

        # 1-month return (last ~21 trading days)
        ret_1m = close.iloc[-1] / close.iloc[-21] - 1

        # 3-month daily returns -> annualised vol
        daily_rets = close.iloc[-63:].pct_change().dropna()

        # meaningful volume data
        if len(daily_rets) < 40:
            continue

        vol_3m = daily_rets.std() * np.sqrt(252)

        if vol_3m == 0 or np.isnan(vol_3m):
            continue

        # composite score handles negative 1m return with small 3m volume,
        # but clip extreme values so we don't see blow-up names.
        results[str(symbol)] = np.clip(ret_1m / vol_3m, -3.0, 3.0)

    return pd.Series(results, dtype=float)
