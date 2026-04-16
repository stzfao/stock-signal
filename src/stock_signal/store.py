"""DuckDB storage layer — schema creation, upserts, and query helpers."""

from datetime import datetime, timedelta

import duckdb
import pandas as pd

from stock_signal.config import DB_PATH, STALENESS_DAYS


def get_connection() -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH))
    _init_schema(conn)
    return conn


def _init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            symbol       VARCHAR NOT NULL,
            date         DATE NOT NULL,
            open         DOUBLE,
            high         DOUBLE,
            low          DOUBLE,
            close        DOUBLE,
            adj_close    DOUBLE,
            volume       BIGINT,
            fetched_at   TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS financials (
            symbol                      VARCHAR NOT NULL,
            date                        DATE NOT NULL,
            period                      VARCHAR NOT NULL,
            revenue                     DOUBLE,
            cost_of_revenue             DOUBLE,
            gross_profit                DOUBLE,
            net_income                  DOUBLE,
            total_assets                DOUBLE,
            total_current_assets        DOUBLE,
            total_current_liabilities   DOUBLE,
            long_term_debt              DOUBLE,
            total_stockholders_equity   DOUBLE,
            shares_outstanding          DOUBLE,
            operating_cash_flow         DOUBLE,
            capital_expenditure         DOUBLE,
            gross_profit_ratio          DOUBLE,
            asset_turnover              DOUBLE,
            fetched_at                  TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (symbol, date, period)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS earnings_surprises (
            symbol                  VARCHAR NOT NULL,
            date                    DATE NOT NULL,
            actual_earnings_result  DOUBLE,
            estimated_earnings      DOUBLE,
            fetched_at              TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyst_estimates (
            symbol                      VARCHAR NOT NULL,
            date                        DATE NOT NULL,
            estimated_eps_avg           DOUBLE,
            estimated_eps_high          DOUBLE,
            estimated_eps_low           DOUBLE,
            number_analysts_estimated   BIGINT,
            fetched_at                  TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (symbol, date)
        )
    """)


# -- Staleness check --

def is_stale(conn: duckdb.DuckDBPyConnection, table: str, symbol: str) -> bool:
    """Return True if symbol has no data or data is older than STALENESS_DAYS."""
    cutoff = datetime.now() - timedelta(days=STALENESS_DAYS)
    result = conn.execute(
        f"SELECT MAX(fetched_at) FROM {table} WHERE symbol = ?", [symbol]
    ).fetchone()
    if result is None or result[0] is None:
        return True
    return result[0] < cutoff


# -- Upsert helpers --

def upsert_prices(conn: duckdb.DuckDBPyConnection, symbol: str, rows: list[dict]) -> int:
    """Insert or replace price rows. Returns count inserted."""
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    col_map = {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adjClose": "adj_close",
        "volume": "volume",
    }
    df = df.rename(columns=col_map)
    keep = [c for c in col_map.values() if c in df.columns]
    df = df[keep]
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["fetched_at"] = datetime.now()
    conn.execute("DELETE FROM prices WHERE symbol = ?", [symbol])
    conn.execute("INSERT INTO prices SELECT * FROM df")
    return len(df)


def upsert_financials(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    income: list[dict],
    balance: list[dict],
    cashflow: list[dict],
) -> int:
    """Merge income + balance + cashflow into financials table."""
    if not income:
        return 0

    def to_df(rows: list[dict], prefix: str = "") -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    inc_df = to_df(income)
    bal_df = to_df(balance)
    cf_df = to_df(cashflow)

    # Merge on (date, period)
    merge_keys = ["date", "period"]
    merged = inc_df[merge_keys + [
        "revenue", "costOfRevenue", "grossProfit", "netIncome", "grossProfitRatio",
    ]].copy() if all(k in inc_df.columns for k in merge_keys) else pd.DataFrame()

    if merged.empty:
        return 0

    # Balance sheet fields
    bal_fields = {
        "totalAssets": "total_assets",
        "totalCurrentAssets": "total_current_assets",
        "totalCurrentLiabilities": "total_current_liabilities",
        "longTermDebt": "long_term_debt",
        "totalStockholdersEquity": "total_stockholders_equity",
        "weightedAverageShsOut": "shares_outstanding",
    }
    if not bal_df.empty and "date" in bal_df.columns:
        bal_subset = bal_df[merge_keys + [k for k in bal_fields if k in bal_df.columns]]
        merged = merged.merge(bal_subset, on=merge_keys, how="left")

    # Cash flow fields
    cf_fields = {
        "operatingCashFlow": "operating_cash_flow",
        "capitalExpenditure": "capital_expenditure",
    }
    if not cf_df.empty and "date" in cf_df.columns:
        cf_subset = cf_df[merge_keys + [k for k in cf_fields if k in cf_df.columns]]
        merged = merged.merge(cf_subset, on=merge_keys, how="left")

    # Rename all FMP camelCase to snake_case
    rename_map = {
        "costOfRevenue": "cost_of_revenue",
        "grossProfit": "gross_profit",
        "netIncome": "net_income",
        "grossProfitRatio": "gross_profit_ratio",
        **bal_fields,
        **cf_fields,
    }
    merged = merged.rename(columns=rename_map)

    # Compute asset_turnover if possible
    if "revenue" in merged.columns and "total_assets" in merged.columns:
        merged["asset_turnover"] = merged["revenue"] / merged["total_assets"].replace(0, float("nan"))
    else:
        merged["asset_turnover"] = None

    merged["symbol"] = symbol
    merged["fetched_at"] = datetime.now()

    # Select columns matching the table schema
    table_cols = [
        "symbol", "date", "period", "revenue", "cost_of_revenue", "gross_profit",
        "net_income", "total_assets", "total_current_assets", "total_current_liabilities",
        "long_term_debt", "total_stockholders_equity", "shares_outstanding",
        "operating_cash_flow", "capital_expenditure", "gross_profit_ratio",
        "asset_turnover", "fetched_at",
    ]
    for col in table_cols:
        if col not in merged.columns:
            merged[col] = None
    merged = merged[table_cols]

    conn.execute("DELETE FROM financials WHERE symbol = ?", [symbol])
    conn.execute("INSERT INTO financials SELECT * FROM merged")
    return len(merged)


def upsert_earnings_surprises(
    conn: duckdb.DuckDBPyConnection, symbol: str, rows: list[dict]
) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    rename = {
        "date": "date",
        "actualEarningResult": "actual_earnings_result",
        "estimatedEarning": "estimated_earnings",
    }
    df = df.rename(columns=rename)
    keep = [v for v in rename.values() if v in df.columns]
    df = df[keep]
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["fetched_at"] = datetime.now()
    conn.execute("DELETE FROM earnings_surprises WHERE symbol = ?", [symbol])
    conn.execute("INSERT INTO earnings_surprises SELECT * FROM df")
    return len(df)


def upsert_analyst_estimates(
    conn: duckdb.DuckDBPyConnection, symbol: str, rows: list[dict]
) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    rename = {
        "date": "date",
        "estimatedEpsAvg": "estimated_eps_avg",
        "estimatedEpsHigh": "estimated_eps_high",
        "estimatedEpsLow": "estimated_eps_low",
        "numberAnalystsEstimatedEps": "number_analysts_estimated",
    }
    df = df.rename(columns=rename)
    keep = [v for v in rename.values() if v in df.columns]
    df = df[keep]
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["fetched_at"] = datetime.now()
    conn.execute("DELETE FROM analyst_estimates WHERE symbol = ?", [symbol])
    conn.execute("INSERT INTO analyst_estimates SELECT * FROM df")
    return len(df)


# -- Query helpers --

def load_prices(conn: duckdb.DuckDBPyConnection, symbols: list[str] | None = None) -> pd.DataFrame:
    if symbols:
        placeholders = ", ".join(["?"] * len(symbols))
        return conn.execute(
            f"SELECT * FROM prices WHERE symbol IN ({placeholders}) ORDER BY symbol, date",
            symbols,
        ).df()
    return conn.execute("SELECT * FROM prices ORDER BY symbol, date").df()


def load_financials(
    conn: duckdb.DuckDBPyConnection, symbols: list[str] | None = None, period: str = "annual"
) -> pd.DataFrame:
    if symbols:
        placeholders = ", ".join(["?"] * len(symbols))
        return conn.execute(
            f"SELECT * FROM financials WHERE symbol IN ({placeholders}) AND period = ? ORDER BY symbol, date",
            symbols + [period],
        ).df()
    return conn.execute(
        "SELECT * FROM financials WHERE period = ? ORDER BY symbol, date", [period]
    ).df()


def load_earnings_surprises(
    conn: duckdb.DuckDBPyConnection, symbols: list[str] | None = None
) -> pd.DataFrame:
    if symbols:
        placeholders = ", ".join(["?"] * len(symbols))
        return conn.execute(
            f"SELECT * FROM earnings_surprises WHERE symbol IN ({placeholders}) ORDER BY symbol, date",
            symbols,
        ).df()
    return conn.execute("SELECT * FROM earnings_surprises ORDER BY symbol, date").df()


def load_analyst_estimates(
    conn: duckdb.DuckDBPyConnection, symbols: list[str] | None = None
) -> pd.DataFrame:
    if symbols:
        placeholders = ", ".join(["?"] * len(symbols))
        return conn.execute(
            f"SELECT * FROM analyst_estimates WHERE symbol IN ({placeholders}) ORDER BY symbol, date",
            symbols,
        ).df()
    return conn.execute("SELECT * FROM analyst_estimates ORDER BY symbol, date").df()
