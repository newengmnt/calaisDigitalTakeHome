"""Stage 2: sequential entry/exit/hedge decision loop.

Reads data/candidates.parquet (built by preprocessing.py). Produces
outputs/trades.csv and outputs/hedge_trades.csv only -- no per-minute
snapshot bookkeeping lives here, that's minute_snapshot.py's job, computed
afterward from the position intervals this stage produces.

Run:
    python src/trade_engine.py
"""
import argparse
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from common import MICROSECONDS_PER_MINUTE, build_symbol_series, build_spot_series, asof_value
from pnl import call_intrinsic, short_call_gross_pnl, short_call_net_pnl
from greeks import position_delta, target_hedge_position
from execution import option_fee, hedge_fee, round_to_lot, UNDERLYING_LOT_SIZE

MIN_SCORE           = 0.0
HEDGE_THRESHOLD     = 0.02      # BTC delta band before rebalancing
MAX_OPEN_POSITIONS  = 5
OPTION_LOT_SIZE     = 1.0


@dataclass
class Position:
    symbol_id: int
    symbol: str
    strike: float
    expiry: float          # microseconds, matches the source `expiration` column
    entry_time: int
    entry_premium: float   # BTC-quoted price, as executed (= bid at entry)
    entry_spot: float      # USD/BTC spot at entry -- needed to convert the BTC premium to USD
    entry_dte: float
    entry_mark: float
    entry_delta: float
    entry_gamma: float
    entry_vega: float
    entry_theta: float
    entry_bid: float
    entry_ask: float
    qty: float              # negative: short position, magnitude = contracts held


def compute_eligible_mask(cand: pd.DataFrame, last_minute: int) -> np.ndarray:
    """A candidate is only entry-eligible if its expiry falls at or before the
    dataset's last observed minute. Entering anything that expires beyond
    that would leave a position the loop can never settle -- it would sit in
    open_positions forever, silently distorting hedge state and NAV with no
    corresponding trades.csv row. Excluding it here means it's simply never
    entered in the first place."""
    last_minute_us = last_minute * MICROSECONDS_PER_MINUTE
    return (cand["expiration"] <= last_minute_us).to_numpy()


def generate_trades(cand: pd.DataFrame, *,
                     hedge_threshold: float = HEDGE_THRESHOLD,
                     max_open_positions: int = MAX_OPEN_POSITIONS,
                     option_lot_size: float = OPTION_LOT_SIZE,
                     min_score: float = MIN_SCORE,
                     underlying_lot_size: float = UNDERLYING_LOT_SIZE):
    cand = cand.sort_values(["minute_timestamp", "symbol_id"]).reset_index(drop=True)
    unique_minutes = np.unique(cand["minute_timestamp"].to_numpy())
    last_minute = unique_minutes.max()

    eligible = compute_eligible_mask(cand, last_minute)
    scored = cand["score"].to_numpy().copy()
    scored[~eligible] = -np.inf   # never selected by the per-minute argmax below

    minute_ts  = cand["minute_timestamp"].to_numpy()
    symbol_id  = cand["symbol_id"].to_numpy()
    strike     = cand["strike_price"].to_numpy()
    expiry     = cand["expiration"].to_numpy()
    bid        = cand["bid_price_last"].to_numpy()
    ask        = cand["ask_price_last"].to_numpy()
    delta      = cand["delta_last"].to_numpy()
    gamma      = cand["gamma_last"].to_numpy()
    vega       = cand["vega_last"].to_numpy()
    theta      = cand["theta_last"].to_numpy()
    mark       = cand["mark_price_last"].to_numpy()
    dte_days   = cand["dte_days"].to_numpy()
    symbol_index = pd.Series(cand["symbol"].to_numpy(), index=symbol_id).groupby(level=0).first()

    best_idx_by_minute = (
        pd.DataFrame({"minute_timestamp": minute_ts, "score": scored})
        .groupby("minute_timestamp")["score"].idxmax()
        .to_dict()
    )

    symbol_series = build_symbol_series(cand)
    spot_arr = build_spot_series(cand, unique_minutes)

    open_positions: Dict[int, Position] = {}
    hedge_units = 0.0
    trades, hedge_trades = [], []

    for i, m in enumerate(unique_minutes):
        spot = spot_arr[i]

        # 1. Expiry settlement -- DTE computed analytically from the cached
        #    expiry, no per-minute market-data lookup needed for the exit
        #    trigger itself (only the settlement value needs `spot`).
        expired = [
            sid for sid, pos in open_positions.items()
            if (pos.expiry - m * MICROSECONDS_PER_MINUTE) / 1e6 / 86400.0 <= 0.0
        ]
        for sid in expired:
            pos = open_positions.pop(sid)
            size = -pos.qty
            # Option prices in this data are BTC-quoted (bid/ask ~1e-4 to
            # ~0.2), but strike/spot are USD. Convert the BTC premium to USD
            # using spot at the time it was received, so it's addable to the
            # (already-USD) intrinsic settlement value and to hedge PnL/fees,
            # which the spec's own formula (size*spot*rate) treats as USD.
            entry_premium_usd = pos.entry_premium * pos.entry_spot
            settle_price_usd = call_intrinsic(spot, pos.strike)
            entry_fee = option_fee(entry_premium_usd, size)
            gross = short_call_gross_pnl(entry_premium_usd, settle_price_usd, size)
            trades.append(dict(
                entry_time=pos.entry_time, exit_time=m,
                symbol=pos.symbol, symbol_id=pos.symbol_id,
                expiry=pos.expiry, dte_entry=pos.entry_dte, strike=pos.strike,
                spot_entry=pos.entry_spot, spot_exit=spot,
                delta_entry=pos.entry_delta, gamma_entry=pos.entry_gamma,
                vega_entry=pos.entry_vega, theta_entry=pos.entry_theta,
                bid_entry=pos.entry_bid, ask_entry=pos.entry_ask, mark_entry=pos.entry_mark,
                entry_execution_price=pos.entry_premium,           # BTC-quoted, as executed
                exit_execution_price=settle_price_usd / spot if spot > 0 else 0.0,  # BTC-equivalent intrinsic
                quantity=pos.qty, side="sell",
                option_entry_fee=entry_fee, option_exit_fee=0.0,
                option_gross_pnl=gross, option_net_pnl=short_call_net_pnl(gross, entry_fee),   # USD
                exit_reason="expiry_settlement",
            ))

        # 2. Entry -- at most one new position per minute: the single best
        #    eligible candidate, skipped if its symbol is already held.
        if len(open_positions) < max_open_positions:
            row = best_idx_by_minute.get(m)
            if row is not None and scored[row] >= min_score:
                sid = symbol_id[row]
                if sid not in open_positions:
                    open_positions[sid] = Position(
                        symbol_id=sid, symbol=symbol_index.loc[sid],
                        strike=strike[row], expiry=expiry[row], entry_time=m,
                        entry_premium=bid[row], entry_spot=spot,
                        entry_dte=dte_days[row], entry_mark=mark[row],
                        entry_delta=delta[row],
                        entry_gamma=gamma[row], entry_vega=vega[row],
                        entry_theta=theta[row], entry_bid=bid[row],
                        entry_ask=ask[row], qty=-option_lot_size,
                    )

        # 3. Net delta -- asof lookup per open position (handles gaps; no
        #    staleness cap for now).
        net_delta = 0.0
        for sid, pos in open_positions.items():
            d = asof_value(symbol_series[sid], m, "delta")
            net_delta += position_delta(pos.qty, d)

        # 4. Threshold-based hedge rebalance, rounded to the tradable lot.
        target_hedge = target_hedge_position(net_delta)
        raw_trade_size = target_hedge - hedge_units
        if abs(raw_trade_size) > hedge_threshold:
            trade_size = round_to_lot(raw_trade_size, lot=underlying_lot_size)
            if trade_size != 0.0:
                fee = hedge_fee(trade_size, spot)
                hedge_trades.append(dict(
                    timestamp=m, spot_price=spot,
                    previous_hedge_position=hedge_units,
                    target_hedge_position=target_hedge,
                    hedge_trade_size=trade_size,
                    hedge_trade_notional=abs(trade_size) * spot,
                    hedge_fee=fee, side="buy" if trade_size > 0 else "sell",
                    reason_for_rebalance="threshold",
                ))
                hedge_units += trade_size

    return pd.DataFrame(trades), pd.DataFrame(hedge_trades)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="./data/candidates.parquet")
    parser.add_argument("--trades-output", default="./outputs/trades.csv")
    parser.add_argument("--hedge-trades-output", default="./outputs/hedge_trades.csv")
    parser.add_argument("--hedge-threshold", type=float, default=HEDGE_THRESHOLD)
    parser.add_argument("--max-open-positions", type=int, default=MAX_OPEN_POSITIONS)
    parser.add_argument("--option-lot-size", type=float, default=OPTION_LOT_SIZE)
    parser.add_argument("--min-score", type=float, default=MIN_SCORE)
    parser.add_argument("--underlying-lot-size", type=float, default=UNDERLYING_LOT_SIZE)
    args = parser.parse_args()

    cand = pd.read_parquet(args.candidates)
    trades_df, hedge_trades_df = generate_trades(
        cand, hedge_threshold=args.hedge_threshold,
        max_open_positions=args.max_open_positions,
        option_lot_size=args.option_lot_size, min_score=args.min_score,
        underlying_lot_size=args.underlying_lot_size,
    )

    trades_df.to_csv(args.trades_output, index=False)
    hedge_trades_df.to_csv(args.hedge_trades_output, index=False)

    print(f"option trades generated : {len(trades_df)}")
    print(f"hedge trades generated  : {len(hedge_trades_df)}")
    print(f"trades written to       : {args.trades_output}")
    print(f"hedge trades written to : {args.hedge_trades_output}")
