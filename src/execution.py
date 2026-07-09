"""Fees, spreads, and lot-size rounding. Pure functions -- no state, no I/O."""
import numpy as np

OPTION_FEE_RATE     = 0.0005
HEDGE_FEE_RATE      = 0.0004
UNDERLYING_LOT_SIZE = 0.001


def mid_price(bid: float, ask: float) -> float:
    return (bid + ask) / 2.0


def relative_spread(bid: float, ask: float) -> float:
    mid = mid_price(bid, ask)
    return (ask - bid) / mid if mid > 0 else float("inf")


def option_fee(premium: float, size: float, rate: float = OPTION_FEE_RATE) -> float:
    """size is contracts (positive); fee is always a positive cost."""
    return abs(size) * premium * rate


def hedge_fee(trade_size: float, spot: float, rate: float = HEDGE_FEE_RATE) -> float:
    """trade_size can be signed (buy or sell); fee is always a positive cost."""
    return abs(trade_size) * spot * rate


def round_to_lot(size: float, lot: float = UNDERLYING_LOT_SIZE) -> float:
    """Round a hedge trade size to the nearest tradable lot.

    Nearest-multiple, not ceiling -- rounding up every trade would
    systematically over-hedge across thousands of rebalances.
    """
    return np.round(size / lot) * lot
