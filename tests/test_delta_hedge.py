"""Delta hedge sign/calculation: the spec's own worked example (section 11).

Short 1 call with call_delta = +0.20:
  option position delta = -0.20 BTC equivalent
  target underlying hedge = +0.20 BTC equivalent
"""
from greeks import position_delta, target_hedge_position, net_greek
from execution import round_to_lot


def test_position_delta_sign_matches_spec_example():
    pos_delta = position_delta(qty=-1, option_delta=0.20)
    assert pos_delta == -0.20


def test_target_hedge_offsets_position_delta():
    pos_delta = position_delta(qty=-1, option_delta=0.20)
    target = target_hedge_position(pos_delta)
    assert target == 0.20


def test_target_hedge_aggregates_across_book():
    # Two short calls, deltas 0.20 and 0.35 -> net option delta -0.55,
    # target hedge +0.55.
    net_delta = net_greek(qtys=[-1, -1], greek_values=[0.20, 0.35])
    assert net_delta == -0.55
    assert target_hedge_position(net_delta) == 0.55


def test_hedge_trade_size_rounds_to_tradable_lot():
    # raw delta target implies a trade of 0.5783 BTC -> nearest 0.001 lot.
    assert round_to_lot(0.5783) == 0.578
    assert round_to_lot(-0.5786) == -0.579
