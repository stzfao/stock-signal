"""Piotroski F-Score: 9 binary signals for fundamental quality."""

import pandas as pd


def piotroski_fscore(financials: pd.DataFrame) -> pd.Series:
    """Compute Piotroski F-Score (0-9) from the two most recent annual filings.

    The 9 binary signals:
      Profitability (4):
        1. ROA > 0
        2. Operating cash flow > 0
        3. Delta ROA > 0 (improving)
        4. Accrual: CFO > net income (cash quality)
      Leverage/Liquidity (3):
        5. Delta long-term debt ratio < 0 (deleveraging)
        6. Delta current ratio > 0 (improving liquidity)
        7. No new share issuance (shares_outstanding didn't increase)
      Efficiency (2):
        8. Delta gross margin > 0
        9. Delta asset turnover > 0

    Args:
        financials: DataFrame with at least 2 annual filings per symbol.
                    Required columns: symbol, date, net_income, total_assets,
                    operating_cash_flow, long_term_debt, total_current_assets,
                    total_current_liabilities, shares_outstanding,
                    gross_profit_ratio, asset_turnover.

    Returns:
        Series indexed by symbol, integer values 0-9.
    """
    # Sort and take the last 2 filings per symbol
    df = financials.sort_values(["symbol", "date"])
    groups = df.groupby("symbol")

    scores: dict[str, int] = {}

    for symbol, group in groups:
        if len(group) < 2:
            continue

        curr = group.iloc[-1]
        prev = group.iloc[-2]

        score = 0

        # --- Profitability (4 points) ---
        roa_curr = _safe_div(curr["net_income"], curr["total_assets"])
        roa_prev = _safe_div(prev["net_income"], prev["total_assets"])

        # 1. ROA > 0
        if roa_curr is not None and roa_curr > 0:
            score += 1

        # 2. Operating cash flow > 0
        if _pos(curr.get("operating_cash_flow")):
            score += 1

        # 3. Delta ROA > 0
        if roa_curr is not None and roa_prev is not None and roa_curr > roa_prev:
            score += 1

        # 4. Accrual: CFO > Net Income
        cfo = curr.get("operating_cash_flow")
        ni = curr.get("net_income")
        if cfo is not None and ni is not None and cfo > ni:
            score += 1

        # --- Leverage / Liquidity (3 points) ---

        # 5. Delta leverage < 0 (long-term debt / total assets)
        lev_curr = _safe_div(curr.get("long_term_debt"), curr.get("total_assets"))
        lev_prev = _safe_div(prev.get("long_term_debt"), prev.get("total_assets"))
        if lev_curr is not None and lev_prev is not None and lev_curr < lev_prev:
            score += 1

        # 6. Delta current ratio > 0
        cr_curr = _safe_div(curr.get("total_current_assets"), curr.get("total_current_liabilities"))
        cr_prev = _safe_div(prev.get("total_current_assets"), prev.get("total_current_liabilities"))
        if cr_curr is not None and cr_prev is not None and cr_curr > cr_prev:
            score += 1

        # 7. No dilution (shares outstanding didn't increase)
        sh_curr = curr.get("shares_outstanding")
        sh_prev = prev.get("shares_outstanding")
        if sh_curr is not None and sh_prev is not None and sh_curr <= sh_prev:
            score += 1

        # --- Efficiency (2 points) ---

        # 8. Delta gross margin > 0
        gm_curr = curr.get("gross_profit_ratio")
        gm_prev = prev.get("gross_profit_ratio")
        if gm_curr is not None and gm_prev is not None and gm_curr > gm_prev:
            score += 1

        # 9. Delta asset turnover > 0
        at_curr = curr.get("asset_turnover")
        at_prev = prev.get("asset_turnover")
        if at_curr is not None and at_prev is not None and at_curr > at_prev:
            score += 1

        scores[symbol] = score

    return pd.Series(scores, dtype=int)


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _pos(val: float | None) -> bool:
    return val is not None and val > 0
