"""DuckDB storage layer — class-based Store with SQL loaded from queries.sql."""

import re
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

_SQL_PATH = Path(__file__).parent.parent / "sql" / "queries.sql"

_INC_RENAMES: dict[str, str] = {
    "costOfRevenue":              "cost_of_revenue",
    "grossProfit":                "gross_profit",
    "operatingIncome":            "operating_income",
    "ebitda":                     "ebitda",
    "netIncome":                  "net_income",
    "interestExpense":            "interest_expense",
    "depreciationAndAmortization":"depreciation_amortization",
    "epsdiluted":                 "eps_diluted",
    "weightedAverageShsOut":      "shares_outstanding",
}
_BAL_RENAMES: dict[str, str] = {
    "totalAssets":             "total_assets",
    "totalCurrentAssets":      "total_current_assets",
    "totalCurrentLiabilities": "total_current_liabilities",
    "cashAndCashEquivalents":  "cash_and_equivalents",
    "totalDebt":               "total_debt",
    "longTermDebt":            "long_term_debt",
    "totalStockholdersEquity": "total_stockholders_equity",
}
_CF_RENAMES: dict[str, str] = {
    "operatingCashFlow":      "operating_cash_flow",
    "capitalExpenditure":     "capital_expenditure",
    "freeCashFlow":           "free_cash_flow",
    "stockBasedCompensation": "stock_based_compensation",
}


def _load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    name: str | None = None
    buf: list[str] = []
    for line in path.read_text().splitlines():
        m = re.match(r"--\s*name:\s*(\S+)", line)
        if m:
            if name:
                queries[name] = "\n".join(buf).strip()
            name, buf = m.group(1), []
        elif name is not None:
            buf.append(line)
    if name:
        queries[name] = "\n".join(buf).strip()
    return queries


class Store:
    _queries: dict[str, str] = {}  # class-level cache, loaded once

    def __init__(self, db_path: Path, staleness_days: int) -> None:
        self._db_path = db_path
        self._staleness_days = staleness_days
        self._conn: duckdb.DuckDBPyConnection | None = None
        if not Store._queries:
            Store._queries = _load_queries(_SQL_PATH)

    def __enter__(self) -> "Store":
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._db_path))
        for name in ("create-prices-table", "create-financials-table",
                     "create-earnings-surprises-table", "create-analyst-estimates-table"):
            self._conn.execute(Store._queries[name])
        return self

    def __exit__(self, *_: object) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        assert self._conn is not None, "Store used outside context manager"
        return self._conn

    def _q(self, name: str, **fmt: str) -> str:
        sql = Store._queries[name]
        return sql.format(**fmt) if fmt else sql

    # ------------------------------------------------------------------ #
    # Staleness                                                            #
    # ------------------------------------------------------------------ #

    def is_stale(self, table: str, symbol: str) -> bool:
        cutoff = datetime.now() - timedelta(days=self._staleness_days)
        row = self.conn.execute(self._q("stale-check", table=table), [symbol]).fetchone()
        return row is None or row[0] is None or row[0] < cutoff

    # ------------------------------------------------------------------ #
    # Upserts                                                              #
    # ------------------------------------------------------------------ #

    def _upsert(self, table: str, symbol: str, df: pd.DataFrame) -> int:
        """Delete-then-insert a DataFrame for one symbol. Uses DuckDB's BY NAME insert."""
        if df.empty:
            return 0
        table_cols = {r[0] for r in self.conn.execute(f"DESCRIBE {table}").fetchall()}
        keep = [c for c in df.columns if c in table_cols]
        _df = df[keep].assign(symbol=symbol, fetched_at=datetime.now())
        self.conn.execute(self._q("delete-symbol", table=table), [symbol])
        self.conn.register("_df", _df)
        self.conn.execute(self._q("insert-df", table=table))
        self.conn.unregister("_df")
        return len(_df)

    def upsert_prices(self, symbol: str, rows: list[dict]) -> int:
        if not rows:
            return 0
        df = (
            pd.DataFrame(rows)
            .rename(columns={"adjClose": "adj_close"})
            .assign(adj_close=lambda d: d.get("adj_close", d["close"]))
        )
        df["date"] = pd.to_datetime(df["date"]).dt.date
        keep = ["date", "open", "high", "low", "close", "adj_close", "volume"]
        df = df[[c for c in keep if c in df.columns]].drop_duplicates(subset=["date"], keep="last")
        return self._upsert("prices", symbol, df)

    def upsert_financials(
        self, symbol: str, income: list[dict], balance: list[dict], cashflow: list[dict]
    ) -> int:
        if not income:
            return 0

        def _df(rows: list[dict], renames: dict[str, str]) -> pd.DataFrame:
            if not rows:
                return pd.DataFrame()
            d = pd.DataFrame(rows).rename(columns=renames)
            d["date"] = pd.to_datetime(d["date"]).dt.date
            return d

        keys = ["date", "period"]
        inc = _df(income, _INC_RENAMES)
        bal = _df(balance, _BAL_RENAMES)
        cf  = _df(cashflow, _CF_RENAMES)

        if not all(k in inc.columns for k in keys):
            return 0

        merged = inc.copy()
        for other in (bal, cf):
            if not other.empty and "date" in other.columns:
                cols = keys + [c for c in other.columns if c not in keys and c not in merged.columns]
                merged = merged.merge(other[cols], on=keys, how="left")

        if "gross_profit" in merged.columns and "revenue" in merged.columns:
            merged["gross_profit_ratio"] = merged["gross_profit"] / merged["revenue"].replace(0, float("nan"))
        if "revenue" in merged.columns and "total_assets" in merged.columns:
            merged["asset_turnover"] = merged["revenue"] / merged["total_assets"].replace(0, float("nan"))

        merged = merged.drop_duplicates(subset=keys, keep="last")
        return self._upsert("financials", symbol, merged)

    def upsert_earnings_surprises(self, symbol: str, rows: list[dict]) -> int:
        if not rows:
            return 0
        df = (
            pd.DataFrame(rows)
            .rename(columns={"epsActual": "actual_earnings_result", "epsEstimated": "estimated_earnings"})
            .dropna(subset=["actual_earnings_result"])
        )
        if df.empty:
            return 0
        df["date"] = pd.to_datetime(df["date"]).dt.date
        keep = ["date", "actual_earnings_result", "estimated_earnings"]
        return self._upsert("earnings_surprises", symbol, df[[c for c in keep if c in df.columns]])

    def upsert_analyst_estimates(self, symbol: str, rows: list[dict]) -> int:
        if not rows:
            return 0
        df = pd.DataFrame(rows).rename(columns={
            "estimatedEpsAvg":           "estimated_eps_avg",
            "estimatedEpsHigh":          "estimated_eps_high",
            "estimatedEpsLow":           "estimated_eps_low",
            "numberAnalystsEstimatedEps":"number_analysts_estimated",
        })
        df["date"] = pd.to_datetime(df["date"]).dt.date
        keep = ["date", "estimated_eps_avg", "estimated_eps_high", "estimated_eps_low", "number_analysts_estimated"]
        return self._upsert("analyst_estimates", symbol, df[[c for c in keep if c in df.columns]])

    # ------------------------------------------------------------------ #
    # Loaders                                                              #
    # ------------------------------------------------------------------ #

    def load_prices(self, symbols: list[str] | None = None) -> pd.DataFrame:
        if symbols:
            return self.conn.execute(self._q("load-prices-by-symbol"), [symbols]).df()
        return self.conn.execute(self._q("load-prices")).df()

    def load_financials(self, symbols: list[str] | None = None, period: str = "annual") -> pd.DataFrame:
        if symbols:
            return self.conn.execute(self._q("load-financials-by-symbol"), [symbols, period]).df()
        return self.conn.execute(self._q("load-financials"), [period]).df()

    def load_earnings_surprises(self, symbols: list[str] | None = None) -> pd.DataFrame:
        if symbols:
            return self.conn.execute(self._q("load-earnings-surprises-by-symbol"), [symbols]).df()
        return self.conn.execute(self._q("load-earnings-surprises")).df()

    def load_analyst_estimates(self, symbols: list[str] | None = None) -> pd.DataFrame:
        if symbols:
            return self.conn.execute(self._q("load-analyst-estimates-by-symbol"), [symbols]).df()
        return self.conn.execute(self._q("load-analyst-estimates")).df()
