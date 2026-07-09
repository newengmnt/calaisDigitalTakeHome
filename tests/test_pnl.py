"""Short-call PnL: the spec's own worked example (section 11).

Strike K = 110, entry premium = 3, size = 1 (short 1 call):
  S_T = 100 -> PnL = +3 before fees
  S_T = 120 -> PnL = 3 - 10 = -7 before fees
"""
from pnl import call_intrinsic, short_call_gross_pnl, short_call_net_pnl


def test_short_call_pnl_otm_is_full_premium():
    settle = call_intrinsic(spot=100, strike=110)
    pnl = short_call_gross_pnl(entry_premium=3, settle_price=settle, size=1)
    assert pnl == 3.0


def test_short_call_pnl_itm_loses_intrinsic_minus_premium():
    settle = call_intrinsic(spot=120, strike=110)
    pnl = short_call_gross_pnl(entry_premium=3, settle_price=settle, size=1)
    assert pnl == -7.0


def test_short_call_net_pnl_subtracts_fees():
    gross = short_call_gross_pnl(entry_premium=3, settle_price=0.0, size=1)
    net = short_call_net_pnl(gross, entry_fee=0.15, exit_fee=0.0)
    assert net == gross - 0.15


def test_short_call_pnl_scales_with_size():
    settle = call_intrinsic(spot=100, strike=110)
    pnl_1x = short_call_gross_pnl(entry_premium=3, settle_price=settle, size=1)
    pnl_5x = short_call_gross_pnl(entry_premium=3, settle_price=settle, size=5)
    assert pnl_5x == 5 * pnl_1x
