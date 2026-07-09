"""Stage 1: build the scored, backtest-ready 0-3 DTE call candidate table.

Reads data/full_data_clean.parquet (already cleaned by clean_data.py: no
nulls, no crossed markets, calls only within the underlying pipeline's own
25-day expiry window, greek sign bounds enforced). This script narrows that
further to the strategy's actual tradable universe (0-3 DTE) and computes the
option-selection score, so trade_engine.py and minute_snapshot.py never touch
the raw 5.7M-row source.

Run:
    python src/preprocessing.py
"""
import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

MICROSECONDS_PER_MINUTE = 60_000_000
MIN_DTE_DAYS = 0
MAX_DTE_DAYS = 3


@dataclass
class ScoreWeights:
    """Configurable weight terms for score_i = w_theta*theta_carry
    + w_vega*vega_carry - w_gamma*gamma_risk - w_spread*spread_cost.

    Any explicit value overrides auto-calibration; leave as None to have it
    set from the data (1 / IQR of that component), so each weighted term has
    a comparable (25th-75th percentile) spread before being summed.
    """
    w_theta: Optional[float] = None
    w_vega: Optional[float] = None
    w_gamma: Optional[float] = None
    w_spread: Optional[float] = None


def _iqr_weight(series: pd.Series) -> float:
    q25, q75 = series.quantile([0.25, 0.75])
    iqr = q75 - q25
    return 1.0 / iqr if iqr > 0 else 1.0


def calibrate_weights(df: pd.DataFrame, overrides: ScoreWeights) -> ScoreWeights:
    """Fill in any weight left as None with 1/IQR of its raw component, so
    every term contributes on the same order of magnitude by construction."""
    return ScoreWeights(
        w_theta=overrides.w_theta if overrides.w_theta is not None else _iqr_weight(df["theta_carry"]),
        w_vega=overrides.w_vega if overrides.w_vega is not None else _iqr_weight(df["vega_carry"]),
        w_gamma=overrides.w_gamma if overrides.w_gamma is not None else _iqr_weight(df["gamma_risk"]),
        w_spread=overrides.w_spread if overrides.w_spread is not None else _iqr_weight(df["spread_cost"]),
    )


def load_candidates(source: str) -> pd.DataFrame:
    con = duckdb.connect()
    query = f"""
        SELECT *,
               (expiration - minute_timestamp * {MICROSECONDS_PER_MINUTE}) / 1e6 / 86400.0 AS dte_days
        FROM read_parquet('{source}')
        WHERE type = 'call'
          AND bid_price_last > 0
          AND (expiration - minute_timestamp * {MICROSECONDS_PER_MINUTE}) / 1e6 / 86400.0
              BETWEEN {MIN_DTE_DAYS} AND {MAX_DTE_DAYS}
    """
    return con.execute(query).df()


def compute_score_components(df: pd.DataFrame) -> pd.DataFrame:
    mid = (df["bid_price_last"] + df["ask_price_last"]) / 2.0

    df["spread_cost"] = (df["ask_price_last"] - df["bid_price_last"]) / mid
    # Short call: daily theta income is the negative of the (negative) raw theta.
    df["theta_carry"] = -df["theta_last"]
    # Short-vega benefit scales with the size of the vega exposure being sold.
    df["vega_carry"] = df["vega_last"]
    # Expected variance of the underlying over the option's remaining life,
    # S^2 * sigma^2 * T -- standard proxy for "how much gamma can hurt".
    iv_decimal = df["mark_iv_last"] / 100.0
    expected_variance = (df["underlying_price_last"] * iv_decimal) ** 2 * (df["dte_days"] / 365.0)
    df["gamma_risk"] = df["gamma_last"].abs() * expected_variance
    return df


def compute_score(df: pd.DataFrame, weights: ScoreWeights) -> pd.DataFrame:
    df["score"] = (
        weights.w_theta * df["theta_carry"]
        + weights.w_vega * df["vega_carry"]
        - weights.w_gamma * df["gamma_risk"]
        - weights.w_spread * df["spread_cost"]
    )
    return df


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """A small number of (minute, symbol) pairs appear twice with slightly
    different greek/price values (sub-tick snapshot artifacts, not exact
    duplicates so the source's own SELECT DISTINCT doesn't catch them).
    Keep one deterministically -- last by row order -- and report the count."""
    before = len(df)
    df = df.drop_duplicates(["minute_timestamp", "symbol"], keep="last")
    return df, before - len(df)


def validate(df: pd.DataFrame) -> None:
    assert not df.duplicated(["minute_timestamp", "symbol"]).any(), \
        "duplicate (minute, symbol) rows -- breaks asof/lookup alignment downstream"
    assert df["score"].notna().all(), "NaN score -- breaks argmax/threshold comparisons downstream"
    assert (df.groupby("symbol")["strike_price"].nunique() == 1).all(), \
        "a symbol's strike_price changed across rows"
    assert (df.groupby("symbol")["expiration"].nunique() == 1).all(), \
        "a symbol's expiration changed across rows"


def run(source: str, output: str, meta_output: str, weights: ScoreWeights) -> None:
    raw_count_query = f"SELECT count(*) FROM read_parquet('{source}')"
    raw_count = duckdb.connect().execute(raw_count_query).fetchone()[0]

    df = load_candidates(source)
    df = compute_score_components(df)

    calibrated = calibrate_weights(df, weights)
    df = compute_score(df, calibrated)

    df = df.sort_values(["minute_timestamp", "symbol"]).reset_index(drop=True)
    df, n_duplicates_removed = deduplicate(df)

    df["symbol_id"], symbol_categories = pd.factorize(df["symbol"])
    df = df.sort_values(["minute_timestamp", "symbol_id"]).reset_index(drop=True)

    validate(df)

    keep_cols = [
        "minute_timestamp", "symbol", "symbol_id", "strike_price", "expiration",
        "dte_days", "bid_price_last", "ask_price_last", "mark_price_last",
        "underlying_price_last", "delta_last", "gamma_last", "vega_last",
        "theta_last", "mark_iv_last",
        "spread_cost", "theta_carry", "vega_carry", "gamma_risk", "score",
    ]
    df[keep_cols].to_parquet(output, index=False)

    meta = {
        "weights": asdict(calibrated),
        "raw_rows_considered": int(raw_count),
        "candidate_rows": int(len(df)),
        "duplicate_minute_symbol_rows_removed": int(n_duplicates_removed),
        "distinct_minutes": int(df["minute_timestamp"].nunique()),
        "distinct_symbols": int(df["symbol_id"].nunique()),
    }
    with open(meta_output, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"raw rows considered      : {raw_count}")
    print(f"candidate rows (0-3 DTE calls): {len(df)}")
    print(f"duplicate rows removed   : {n_duplicates_removed}")
    print(f"distinct minutes         : {df['minute_timestamp'].nunique()}")
    print(f"distinct symbols         : {df['symbol_id'].nunique()}")
    print(f"calibrated weights       : {asdict(calibrated)}")
    print(f"output written to        : {output}")
    print(f"metadata written to      : {meta_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./data/full_data_clean.parquet")
    parser.add_argument("--output", default="./data/candidates.parquet")
    parser.add_argument("--meta-output", default="./data/candidates_meta.json")
    parser.add_argument("--w-theta", type=float, default=None)
    parser.add_argument("--w-vega", type=float, default=None)
    parser.add_argument("--w-gamma", type=float, default=None)
    parser.add_argument("--w-spread", type=float, default=None)
    args = parser.parse_args()

    run(
        source=args.input,
        output=args.output,
        meta_output=args.meta_output,
        weights=ScoreWeights(
            w_theta=args.w_theta, w_vega=args.w_vega,
            w_gamma=args.w_gamma, w_spread=args.w_spread,
        ),
    )
