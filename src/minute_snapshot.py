"""Stage 3: minute-level NAV / greeks / PnL reconstruction.

Reads data/candidates.parquet plus outputs/trades.csv and
outputs/hedge_trades.csv (both produced by trade_engine.py). No sequential
per-minute decision loop here -- position intervals and hedge events are
already known, so every series below is built with vectorized scatter
(np.add.at / slice assignment) and cumsum/diff, looping only over the
(small) trade and hedge-trade counts rather than the 44,628 minutes.

Run:
    python src/minute_snapshot.py
"""
import argparse

import numpy as np
import pandas as pd

from common import build_symbol_series, build_spot_series, asof_values

INITIAL_CAPITAL = 1_000_000.0


def generate_minute_snapshot(cand: pd.DataFrame, trades_df: pd.DataFrame,
                              hedge_trades_df: pd.DataFrame,
                              initial_capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    unique_minutes = np.unique(cand["minute_timestamp"].to_numpy())
    n = len(unique_minutes)

    symbol_series = build_symbol_series(cand)
    spot_arr = build_spot_series(cand, unique_minutes)

    net_delta = np.zeros(n)
    net_gamma = np.zeros(n)
    net_vega  = np.zeros(n)
    net_theta = np.zeros(n)
    unrealized_pnl = np.zeros(n)
    fee_jump      = np.zeros(n)   # option fees, sparse jumps at entry minutes
    realized_jump = np.zeros(n)   # option pnl, sparse jumps at exit minutes

    for t in trades_df.itertuples():
        entry_g = np.searchsorted(unique_minutes, t.entry_time, side="left")
        exit_g  = np.searchsorted(unique_minutes, t.exit_time, side="left")  # exclusive
        exit_g  = max(exit_g, entry_g + 1)  # a same-minute entry+exit still marks one minute
        target_minutes = unique_minutes[entry_g:exit_g]

        s = symbol_series[t.symbol_id]
        delta_vals = asof_values(s, target_minutes, "delta")
        gamma_vals = asof_values(s, target_minutes, "gamma")
        vega_vals  = asof_values(s, target_minutes, "vega")
        theta_vals = asof_values(s, target_minutes, "theta")
        ask_vals   = asof_values(s, target_minutes, "ask")   # BTC-quoted

        # BTC-quoted option prices -> USD, using spot at the time of each cash
        # flow, so this leg is addable to the (already-USD) hedge PnL/fees.
        entry_premium_usd = t.entry_execution_price * t.spot_entry
        ask_vals_usd = ask_vals * spot_arr[entry_g:exit_g]

        size = -t.quantity
        net_delta[entry_g:exit_g] += t.quantity * delta_vals
        net_gamma[entry_g:exit_g] += t.quantity * gamma_vals
        net_vega[entry_g:exit_g]  += t.quantity * vega_vals
        net_theta[entry_g:exit_g] += t.quantity * theta_vals
        unrealized_pnl[entry_g:exit_g] += size * (entry_premium_usd - ask_vals_usd)

        fee_jump[entry_g] += t.option_entry_fee
        exit_idx = min(np.searchsorted(unique_minutes, t.exit_time, side="left"), n - 1)
        realized_jump[exit_idx] += t.option_gross_pnl

    cumulative_option_fees = np.cumsum(fee_jump)
    realized_option_pnl    = np.cumsum(realized_jump)

    hedge_delta_jump = np.zeros(n)
    hedge_fee_jump   = np.zeros(n)
    if not hedge_trades_df.empty:
        h_idx = np.searchsorted(unique_minutes, hedge_trades_df["timestamp"].to_numpy())
        h_idx = np.clip(h_idx, 0, n - 1)
        np.add.at(hedge_delta_jump, h_idx, hedge_trades_df["hedge_trade_size"].to_numpy())
        np.add.at(hedge_fee_jump, h_idx, hedge_trades_df["hedge_fee"].to_numpy())
    hedge_units_arr       = np.cumsum(hedge_delta_jump)
    cumulative_hedge_fees = np.cumsum(hedge_fee_jump)

    # Hedge PnL accrues on the *previous* interval's position -- vectorized via diff/cumsum.
    hedge_pnl_increment = np.zeros(n)
    hedge_pnl_increment[1:] = hedge_units_arr[:-1] * np.diff(spot_arr)
    cumulative_hedge_pnl = np.cumsum(hedge_pnl_increment)

    nav = (initial_capital + realized_option_pnl + unrealized_pnl
           + cumulative_hedge_pnl - cumulative_option_fees - cumulative_hedge_fees)

    return pd.DataFrame({
        "minute_timestamp": unique_minutes, "nav": nav,
        "net_delta_before_hedge": net_delta,
        "net_delta_after_hedge": net_delta + hedge_units_arr,
        "net_gamma": net_gamma, "net_vega": net_vega, "net_theta": net_theta,
        "realized_option_pnl": realized_option_pnl,
        "unrealized_option_pnl": unrealized_pnl,
        "cumulative_hedge_pnl": cumulative_hedge_pnl,
        "cumulative_option_fees": cumulative_option_fees,
        "cumulative_hedge_fees": cumulative_hedge_fees,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="./data/candidates.parquet")
    parser.add_argument("--trades", default="./outputs/trades.csv")
    parser.add_argument("--hedge-trades", default="./outputs/hedge_trades.csv")
    parser.add_argument("--output", default="./outputs/minute_snapshot.csv")
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)
    args = parser.parse_args()

    cand = pd.read_parquet(args.candidates)
    trades_df = pd.read_csv(args.trades)
    hedge_trades_df = pd.read_csv(args.hedge_trades)

    snapshot = generate_minute_snapshot(cand, trades_df, hedge_trades_df, args.initial_capital)
    snapshot.to_csv(args.output, index=False)

    print(f"minute snapshot rows : {len(snapshot)}")
    print(f"final NAV            : {snapshot['nav'].iloc[-1]:.2f}")
    print(f"output written to    : {args.output}")
