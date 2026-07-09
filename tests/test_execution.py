"""Transaction fees: option fee = 5 bps, hedge fee = 4 bps (spec section 4.4),
plus lot-size rounding for the underlying hedge."""
import pytest

from execution import option_fee, hedge_fee, round_to_lot, relative_spread


def test_option_fee_is_five_bps_of_premium_times_size():
    fee = option_fee(premium=100.0, size=2, rate=0.0005)
    assert fee == 0.10  # 2 * 100 * 0.0005


def test_hedge_fee_is_four_bps_of_notional():
    fee = hedge_fee(trade_size=0.5, spot=100_000.0, rate=0.0004)
    assert fee == 20.0  # 0.5 * 100_000 * 0.0004


def test_fees_are_always_positive_regardless_of_side():
    buy_fee = hedge_fee(trade_size=0.5, spot=100_000.0)
    sell_fee = hedge_fee(trade_size=-0.5, spot=100_000.0)
    assert buy_fee == sell_fee > 0


def test_relative_spread_matches_spec_formula():
    # spread=0.05, mid=0.5 -> relative_spread = 0.1
    assert relative_spread(bid=0.475, ask=0.525) == pytest.approx(0.1)


def test_round_to_lot_snaps_to_nearest_underlying_increment():
    assert round_to_lot(0.02134, lot=0.001) == 0.021
    assert round_to_lot(0.0, lot=0.001) == 0.0
