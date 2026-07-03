import argparse
import polars as pl

from clean_data import (
    REQUIRED_NOT_NULL_COLS,
    MICROSECONDS_PER_MINUTE,
    MAX_MICROSECONDS_TO_EXPIRY,
    GREEK_BOUNDS,
)


def validate(path: str):
    lf = pl.scan_parquet(path)

    diff_us = pl.col("expiration") - pl.col("minute_timestamp") * MICROSECONDS_PER_MINUTE
    null_count_by_col = {c: pl.col(c).is_null().sum() for c in REQUIRED_NOT_NULL_COLS}

    # One boolean per greek bound: True iff every row (mean & last) respects it.
    greek_bound_exprs = {}
    for greek, (lo, hi) in GREEK_BOUNDS.items():
        for suffix in ("mean", "last"):
            col = f"{greek}_{suffix}"
            if lo is not None:
                greek_bound_exprs[f"greek__{col}_ge_lo"] = (pl.col(col) >= lo).all()
            if hi is not None:
                greek_bound_exprs[f"greek__{col}_le_hi"] = (pl.col(col) <= hi).all()

    # Single lazy pass: every check reduces to a scalar, so polars only ever
    # streams column chunks through, never materializing the full table.
    summary = lf.select(
        row_count=pl.len(),
        distinct_row_count=pl.struct(pl.all()).n_unique(),
        bid_lt_ask_last=(pl.col("bid_price_last") < pl.col("ask_price_last")).all(),
        bid_lt_ask_mean=(pl.col("bid_price_mean") < pl.col("ask_price_mean")).all(),
        expiry_diff_ge_0=(diff_us >= 0).all(),
        expiry_diff_le_max=(diff_us <= MAX_MICROSECONDS_TO_EXPIRY).all(),
        type_is_call=(pl.col("type") == "call").all(),
        **null_count_by_col,
        **greek_bound_exprs,
    ).collect().row(0, named=True)

    print(f"rows in output: {summary['row_count']}")

    null_counts = {c: summary[c] for c in REQUIRED_NOT_NULL_COLS}
    checks = {
        "no nulls in required columns": all(n == 0 for n in null_counts.values()),
        "bid_price_last < ask_price_last everywhere": summary["bid_lt_ask_last"],
        "bid_price_mean < ask_price_mean everywhere": summary["bid_lt_ask_mean"],
        "expiry diff >= 0 everywhere": summary["expiry_diff_ge_0"],
        "expiry diff <= 30 days everywhere": summary["expiry_diff_le_max"],
        "no duplicate rows": summary["distinct_row_count"] == summary["row_count"],
        "type == 'call' everywhere": summary["type_is_call"],
        "all greeks within sign/definition bounds": all(
            summary[k] for k in greek_bound_exprs
        ),
    }

    if not checks["no nulls in required columns"]:
        for col, count in null_counts.items():
            if count:
                print(f"  {col}: {count} nulls")

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
