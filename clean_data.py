import argparse
import duckdb

REQUIRED_NOT_NULL_COLS = [
    "minute_timestamp",
    "open_interest_mean",
    "open_interest_last",
    "underlying_price_mean",
    "underlying_price_last",
    "delta_mean",
    "delta_last",
    "gamma_mean",
    "gamma_last",
    "vega_mean",
    "vega_last",
    "theta_mean",
    "theta_last",
    "rho_mean",
    "rho_last",
    "bid_price_mean",
    "bid_price_last",
    "bid_amount_mean",
    "bid_amount_last",
    "bid_iv_mean",
    "bid_iv_last",
    "ask_price_mean",
    "ask_price_last",
    "ask_amount_mean",
    "ask_amount_last",
    "ask_iv_mean",
    "ask_iv_last",
    "symbol",
    "type",
    "strike_price",
    "expiration",
    "minute",
]

MICROSECONDS_PER_MINUTE = 60 * 1_000_000
MAX_DAYS_TO_EXPIRY = 25
MAX_MICROSECONDS_TO_EXPIRY = MAX_DAYS_TO_EXPIRY * 24 * 3600 * 1_000_000


def build_query(source: str, limit: int | None) -> str:
    not_null_clause = " AND ".join(f"{col} IS NOT NULL" for col in REQUIRED_NOT_NULL_COLS)
    limit_clause = f"LIMIT {limit}" if limit is not None else ""

    return f"""
    WITH base AS (
        SELECT * FROM read_parquet('{source}') {limit_clause}
    ),
    no_nulls AS (
        SELECT * FROM base
        WHERE {not_null_clause}
    ),
    spread_and_expiry_filtered AS (
        SELECT * FROM no_nulls
        WHERE bid_price_last < ask_price_last
          AND bid_price_mean < ask_price_mean
          AND (expiration - minute_timestamp * {MICROSECONDS_PER_MINUTE})
              BETWEEN 0 AND {MAX_MICROSECONDS_TO_EXPIRY}
          AND type = 'call'
    )
    SELECT DISTINCT * FROM spread_and_expiry_filtered
    """


def run(source: str, output: str, limit: int | None):
    con = duckdb.connect()
    query = build_query(source, limit)

    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    raw_count_query = f"SELECT count(*) FROM (SELECT * FROM read_parquet('{source}') {limit_clause})"
    raw_count = con.execute(raw_count_query).fetchone()[0]

    con.execute(f"COPY ({query}) TO '{output}' (FORMAT PARQUET)")
    clean_count = con.execute(f"SELECT count(*) FROM read_parquet('{output}')").fetchone()[0]

    print(f"source rows considered : {raw_count}")
    print(f"rows after cleaning    : {clean_count}")
    print(f"rows removed           : {raw_count - clean_count}")
    print(f"output written to      : {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./BTC_2025-01.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Only read the first N raw rows (for testing)")
    args = parser.parse_args()

    run(args.input, args.output, args.limit)
