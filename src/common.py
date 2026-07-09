"""Shared helpers used by both trade_engine.py and minute_snapshot.py.

Kept in one place so both stages read stale/gappy per-symbol market data via
the *identical* asof mechanism -- diverging this logic between the two files
would let them silently disagree on what a position's delta "was" at a given
minute.
"""
import numpy as np
import pandas as pd

MICROSECONDS_PER_MINUTE = 60_000_000


def build_symbol_series(cand: pd.DataFrame) -> dict:
    """Per-symbol sorted arrays -- O(1) asof lookups instead of a per-minute
    dict keyed by (minute, symbol), which is both larger (1.7M entries) and
    breaks outright on the gaps individual instruments have in this data.
    """
    cand_sorted = cand.sort_values(["symbol_id", "minute_timestamp"])
    series = {}
    for sid, g in cand_sorted.groupby("symbol_id", sort=False):
        series[sid] = {
            "minutes": g["minute_timestamp"].to_numpy(),
            "delta": g["delta_last"].to_numpy(),
            "gamma": g["gamma_last"].to_numpy(),
            "vega": g["vega_last"].to_numpy(),
            "theta": g["theta_last"].to_numpy(),
            "ask": g["ask_price_last"].to_numpy(),
        }
    return series


def build_spot_series(cand: pd.DataFrame, unique_minutes: np.ndarray) -> np.ndarray:
    """Underlying price per global minute. In practice this series is missing
    at most a handful of minutes (a minute with zero eligible call candidates
    at all) -- a plain ffill/bfill is sufficient, no asof machinery needed."""
    spot = (
        cand.groupby("minute_timestamp")["underlying_price_last"]
        .first()
        .reindex(unique_minutes)
        .ffill()
        .bfill()
    )
    return spot.to_numpy()


def asof_values(sym: dict, target_minutes: np.ndarray, field: str) -> np.ndarray:
    """Last known value at or before each target minute (array form)."""
    idx = np.searchsorted(sym["minutes"], target_minutes, side="right") - 1
    idx = np.clip(idx, 0, len(sym["minutes"]) - 1)
    return sym[field][idx]


def asof_value(sym: dict, target_minute: int, field: str) -> float:
    """Last known value at or before a single target minute (scalar form)."""
    idx = np.searchsorted(sym["minutes"], target_minute, side="right") - 1
    idx = max(idx, 0)
    return sym[field][idx]
