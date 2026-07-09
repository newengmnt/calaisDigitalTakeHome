"""Integration test: replay the spec's own worked example (section 11)
through the real generate_trades() sequential loop, not just the extracted
pure functions in isolation -- confirms the production code path reproduces
the hand-checked numbers.

Entry spot is pinned to 1.0 so the BTC->USD premium conversion is a no-op,
making the numbers match the spec's example (which doesn't model that
conversion) exactly: strike 110, premium 3, S_T=100 -> +3, S_T=120 -> -7.
"""
import pandas as pd

from trade_engine import generate_trades

MICROSECONDS_PER_MINUTE = 60_000_000


def _make_candidates(settle_underlying_price: float) -> pd.DataFrame:
    expiry_us = 1 * MICROSECONDS_PER_MINUTE
    rows = [
        # entry minute: dte > 0, real quote, positive score -> gets selected
        dict(minute_timestamp=0, symbol_id=0, symbol="TEST-110-C",
             strike_price=110.0, expiration=expiry_us,
             bid_price_last=3.0, ask_price_last=3.1, mark_price_last=3.05,
             underlying_price_last=1.0, delta_last=0.5, gamma_last=0.01,
             vega_last=10.0, theta_last=-50.0, dte_days=1.0 / 1440, score=1.0),
        # settlement minute: dte <= 0 -> triggers expiry settlement using this
        # minute's underlying_price_last as spot. Low score -> no re-entry.
        dict(minute_timestamp=1, symbol_id=0, symbol="TEST-110-C",
             strike_price=110.0, expiration=expiry_us,
             bid_price_last=0.001, ask_price_last=0.002, mark_price_last=0.0015,
             underlying_price_last=settle_underlying_price, delta_last=0.01,
             gamma_last=0.001, vega_last=1.0, theta_last=-1.0,
             dte_days=0.0, score=-999.0),
    ]
    return pd.DataFrame(rows)


def test_generate_trades_otm_matches_spec_example():
    cand = _make_candidates(settle_underlying_price=100.0)
    trades_df, _ = generate_trades(cand)

    assert len(trades_df) == 1
    trade = trades_df.iloc[0]
    assert trade["exit_reason"] == "expiry_settlement"
    assert trade["quantity"] == -1.0          # short 1 contract
    assert trade["option_gross_pnl"] == 3.0   # OTM -> keep full premium


def test_generate_trades_itm_matches_spec_example():
    cand = _make_candidates(settle_underlying_price=120.0)
    trades_df, _ = generate_trades(cand)

    assert len(trades_df) == 1
    trade = trades_df.iloc[0]
    assert trade["option_gross_pnl"] == -7.0  # premium 3 - intrinsic 10
