"""Position-level greeks and hedge targeting. Pure functions -- no state, no I/O.

Convention: qty is negative for a short position (see Position.qty in
trade_engine.py), so these never re-negate -- the sign is already carried by
qty and multiplying it through is what flips a raw (always-positive-for-calls)
delta/gamma/vega into the correct short-book sign.
"""


def position_delta(qty: float, option_delta: float) -> float:
    return qty * option_delta


def position_gamma(qty: float, option_gamma: float) -> float:
    return qty * option_gamma


def position_vega(qty: float, option_vega: float) -> float:
    return qty * option_vega


def position_theta(qty: float, option_theta: float) -> float:
    return qty * option_theta


def net_greek(qtys, greek_values) -> float:
    """Book-level aggregation: sum of qty*greek across open positions."""
    return sum(q * g for q, g in zip(qtys, greek_values))


def target_hedge_position(net_option_delta: float) -> float:
    """Underlying exposure needed to neutralize the book's option delta."""
    return -net_option_delta
