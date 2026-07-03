import argparse
import duckdb

from clean_data import REQUIRED_NOT_NULL_COLS, MICROSECONDS_PER_MINUTE, MAX_MICROSECONDS_TO_EXPIRY


def validate(path: str):
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{path}')").fetchdf()
    print(f"rows in output: {len(df)}")

    checks = {}

    null_counts = df[REQUIRED_NOT_NULL_COLS].isnull().sum()
    checks["no nulls in required columns"] = bool((null_counts == 0).all())
    if not checks["no nulls in required columns"]:
        print(null_counts[null_counts > 0])

    checks["bid_price_last < ask_price_last everywhere"] = bool((df["bid_price_last"] < df["ask_price_last"]).all())
    checks["bid_price_mean < ask_price_mean everywhere"] = bool((df["bid_price_mean"] < df["ask_price_mean"]).all())

    diff_us = df["expiration"] - df["minute_timestamp"] * MICROSECONDS_PER_MINUTE
    checks["expiry diff >= 0 everywhere"] = bool((diff_us >= 0).all())
    checks["expiry diff <= 30 days everywhere"] = bool((diff_us <= MAX_MICROSECONDS_TO_EXPIRY).all())

    checks["no duplicate rows"] = bool(len(df) == len(df.drop_duplicates()))

    print()
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    validate(args.path)
