"""Payoff sign: call intrinsic/settlement value is never negative, and is
exactly zero when out-of-the-money (spec section 5.1 / 11)."""
from pnl import call_intrinsic


def test_call_intrinsic_zero_when_otm():
    assert call_intrinsic(spot=100, strike=110) == 0.0


def test_call_intrinsic_positive_when_itm():
    assert call_intrinsic(spot=120, strike=110) == 10.0


def test_call_intrinsic_zero_at_the_money():
    assert call_intrinsic(spot=110, strike=110) == 0.0


def test_call_intrinsic_never_negative():
    for spot, strike in [(50, 110), (0, 1), (109.999, 110), (110.001, 110)]:
        assert call_intrinsic(spot, strike) >= 0.0
