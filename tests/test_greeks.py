"""Greeks sign aggregation: a short call must flip the raw (always
delta/gamma/vega >= 0, theta <= 0 for calls) greeks to the short-book
convention -- short gamma, short vega, long theta (spec section 5.3)."""
from greeks import position_delta, position_gamma, position_vega, position_theta, net_greek


def test_short_call_is_short_delta():
    assert position_delta(qty=-1, option_delta=0.30) < 0


def test_short_call_is_short_gamma():
    assert position_gamma(qty=-1, option_gamma=0.02) < 0


def test_short_call_is_short_vega():
    assert position_vega(qty=-1, option_vega=15.0) < 0


def test_short_call_is_long_theta():
    # raw call theta is negative; short position flips it positive (income).
    assert position_theta(qty=-1, option_theta=-400.0) > 0


def test_long_call_book_has_opposite_signs_to_short():
    assert position_delta(qty=1, option_delta=0.30) > 0
    assert position_theta(qty=1, option_theta=-400.0) < 0


def test_net_greek_sums_signed_contributions_across_book():
    # book: short call (delta 0.3) + short call (delta 0.1) -> net -0.4
    net_delta = net_greek(qtys=[-1, -1], greek_values=[0.3, 0.1])
    assert net_delta == -0.4
