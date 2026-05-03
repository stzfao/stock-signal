-- name: create-prices-table
CREATE TABLE IF NOT EXISTS prices (
    symbol      VARCHAR   NOT NULL,
    date        DATE      NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    adj_close   DOUBLE,
    volume      BIGINT,
    fetched_at  TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (symbol, date)
);

-- name: create-financials-table
CREATE TABLE IF NOT EXISTS financials (
    symbol                    VARCHAR NOT NULL,
    date                      DATE    NOT NULL,
    period                    VARCHAR NOT NULL,
    -- Income statement
    revenue                   DOUBLE,
    cost_of_revenue           DOUBLE,
    gross_profit              DOUBLE,
    operating_income          DOUBLE,
    ebitda                    DOUBLE,
    net_income                DOUBLE,
    interest_expense          DOUBLE,
    depreciation_amortization DOUBLE,
    eps_diluted               DOUBLE,
    -- Balance sheet
    total_assets              DOUBLE,
    total_current_assets      DOUBLE,
    total_current_liabilities DOUBLE,
    cash_and_equivalents      DOUBLE,
    total_debt                DOUBLE,
    long_term_debt            DOUBLE,
    total_stockholders_equity DOUBLE,
    shares_outstanding        DOUBLE,
    -- Cash flow
    operating_cash_flow       DOUBLE,
    capital_expenditure       DOUBLE,
    free_cash_flow            DOUBLE,
    stock_based_compensation  DOUBLE,
    -- Derived
    gross_profit_ratio        DOUBLE,
    asset_turnover            DOUBLE,
    fetched_at                TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (symbol, date, period)
);

-- name: create-earnings-surprises-table
CREATE TABLE IF NOT EXISTS earnings_surprises (
    symbol                 VARCHAR NOT NULL,
    date                   DATE    NOT NULL,
    actual_earnings_result DOUBLE,
    estimated_earnings     DOUBLE,
    fetched_at             TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (symbol, date)
);

-- name: create-analyst-estimates-table
CREATE TABLE IF NOT EXISTS analyst_estimates (
    symbol                    VARCHAR NOT NULL,
    date                      DATE    NOT NULL,
    estimated_eps_avg         DOUBLE,
    estimated_eps_high        DOUBLE,
    estimated_eps_low         DOUBLE,
    number_analysts_estimated BIGINT,
    fetched_at                TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (symbol, date)
);

-- name: stale-check
SELECT MAX(fetched_at) FROM {table} WHERE symbol = ?;

-- name: delete-symbol
DELETE FROM {table} WHERE symbol = ?;

-- name: insert-df
INSERT INTO {table} BY NAME SELECT * FROM _df;

-- name: load-prices
SELECT * FROM prices ORDER BY symbol, date;

-- name: load-prices-by-symbol
SELECT * FROM prices WHERE symbol = ANY(?) ORDER BY symbol, date;

-- name: load-financials
SELECT * FROM financials WHERE period = ? ORDER BY symbol, date;

-- name: load-financials-by-symbol
SELECT * FROM financials WHERE symbol = ANY(?) AND period = ? ORDER BY symbol, date;

-- name: load-earnings-surprises
SELECT * FROM earnings_surprises ORDER BY symbol, date;

-- name: load-earnings-surprises-by-symbol
SELECT * FROM earnings_surprises WHERE symbol = ANY(?) ORDER BY symbol, date;

-- name: load-analyst-estimates
SELECT * FROM analyst_estimates ORDER BY symbol, date;

-- name: load-analyst-estimates-by-symbol
SELECT * FROM analyst_estimates WHERE symbol = ANY(?) ORDER BY symbol, date;

-- name: create-score-history-table
CREATE TABLE IF NOT EXISTS score_history (
    run_date        DATE      NOT NULL,
    symbol          VARCHAR   NOT NULL,
    composite_score DOUBLE,
    rank            INTEGER,
    entry_zone      VARCHAR,
    PRIMARY KEY (run_date, symbol)
);

-- name: save-score-history
INSERT OR REPLACE INTO score_history BY NAME SELECT * FROM _df;

-- name: load-score-history-latest
SELECT sh.symbol, sh.days_in_top
FROM (
    SELECT symbol, COUNT(*) AS days_in_top
    FROM score_history
    WHERE run_date >= ?
    GROUP BY symbol
) sh;

-- name: load-previous-run
SELECT symbol, composite_score, rank, entry_zone
FROM score_history
WHERE run_date = (SELECT MAX(run_date) FROM score_history WHERE run_date < ?);
