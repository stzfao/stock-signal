"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Generate 300 days of synthetic price data for 3 symbols."""
    np.random.seed(42)
    symbols = ["AAPL", "MSFT", "GOOGL"]
    rows = []
    for sym in symbols:
        base = 100 + np.random.randn() * 20
        for i in range(300):
            price = base + np.cumsum(np.random.randn(1))[0] * 0.5 + i * 0.05
            rows.append({
                "symbol": sym,
                "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
                "open": price * 0.99,
                "high": price * 1.02,
                "low": price * 0.98,
                "close": price,
                "adj_close": price,
                "volume": int(1e6 + np.random.randint(0, 5e5)),
            })
            base = price
    return pd.DataFrame(rows)


@pytest.fixture
def sample_financials() -> pd.DataFrame:
    """Generate 2 years of annual financials for 3 symbols."""
    rows = []
    for sym in ["AAPL", "MSFT", "GOOGL"]:
        for year, date_str in [(2022, "2022-12-31"), (2023, "2023-12-31")]:
            rev = 100e9 + np.random.rand() * 50e9
            cogs = rev * (0.4 + np.random.rand() * 0.2)
            rows.append({
                "symbol": sym,
                "date": pd.Timestamp(date_str).date(),
                "period": "annual",
                "revenue": rev,
                "cost_of_revenue": cogs,
                "gross_profit": rev - cogs,
                "net_income": rev * 0.15,
                "total_assets": rev * 2,
                "total_current_assets": rev * 0.5,
                "total_current_liabilities": rev * 0.3,
                "long_term_debt": rev * 0.2 if year == 2023 else rev * 0.25,
                "total_stockholders_equity": rev * 0.8,
                "shares_outstanding": 1e9,
                "operating_cash_flow": rev * 0.2,
                "capital_expenditure": -rev * 0.05,
                "gross_profit_ratio": (rev - cogs) / rev,
                "asset_turnover": rev / (rev * 2),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_earnings() -> pd.DataFrame:
    """Generate 8 quarters of earnings surprises for 3 symbols."""
    rows = []
    for sym in ["AAPL", "MSFT", "GOOGL"]:
        for q in range(8):
            est = 1.0 + np.random.rand() * 0.5
            actual = est + np.random.randn() * 0.1
            rows.append({
                "symbol": sym,
                "date": pd.Timestamp("2022-03-31") + pd.DateOffset(months=3 * q),
                "actual_earnings_result": actual,
                "estimated_earnings": est,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_estimates() -> pd.DataFrame:
    """Generate monthly analyst estimates for 3 symbols."""
    rows = []
    for sym in ["AAPL", "MSFT", "GOOGL"]:
        base_eps = 5.0 + np.random.rand()
        for m in range(12):
            # Simulate upward drift for AAPL, downward for GOOGL
            drift = 0.05 if sym == "AAPL" else (-0.03 if sym == "GOOGL" else 0.01)
            eps = base_eps + drift * m + np.random.randn() * 0.02
            rows.append({
                "symbol": sym,
                "date": pd.Timestamp("2023-01-15") + pd.DateOffset(months=m),
                "estimated_eps_avg": eps,
                "estimated_eps_high": eps * 1.1,
                "estimated_eps_low": eps * 0.9,
                "number_analysts_estimated": 20 + np.random.randint(0, 10),
            })
    return pd.DataFrame(rows)
