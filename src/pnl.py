"""Short-call payoff and PnL. Pure functions -- no state, no I/O."""


def call_intrinsic(spot: float, strike: float) -> float:
    """European call intrinsic/settlement value. Never negative."""
    return max(spot - strike, 0.0)


def short_call_gross_pnl(entry_premium: float, settle_price: float, size: float) -> float:
    """PnL for a short call of `size` contracts (size > 0), before fees.

    size * (premium received - value paid to close/settle). Profits when the
    option is worth less at exit than the premium collected at entry.
    """
    return size * (entry_premium - settle_price)


def short_call_net_pnl(gross_pnl: float, entry_fee: float, exit_fee: float = 0.0) -> float:
    return gross_pnl - entry_fee - exit_fee
