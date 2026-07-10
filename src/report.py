"""Stage 4: post-trade analysis, risk metrics, execution analysis, and plots.

Reads outputs/minute_snapshot.csv (primary input) plus outputs/trades.csv,
outputs/hedge_trades.csv, and data/candidates.parquet (only to rebuild the
spot series via common.build_spot_series -- minute_snapshot.csv itself
carries no spot column by design, see conversation notes).

Produces, all under outputs/:
  daily_pnl.csv, greeks_timeseries.csv, risk_summary.json, execution_summary.csv,
  nav_curve.png, drawdown_curve.png, options_pnl.png, hedge_pnl.png,
  greek_pnl_decomposition.png, overall_pnl.png

Run:
    python src/report.py
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import build_spot_series
from execution import option_fee

INITIAL_CAPITAL = 1_000_000.0
TRADING_DAYS_PER_YEAR = 365
DT_DAYS_PER_MINUTE = 1.0 / 1440.0

# ---- palette (validated default, references/palette.md) ----
SURFACE      = "#fcfcfb"
INK_PRIMARY  = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED    = "#898781"
GRIDLINE     = "#e1e0d9"
BASELINE     = "#c3c2b7"
BLUE   = "#2a78d6"   # categorical slot 1 -- "without exec costs" / gross / delta
AQUA   = "#1baf7a"   # categorical slot 2 -- "with exec costs" / net / gamma
YELLOW = "#eda100"   # categorical slot 3 -- theta
GREEN  = "#008300"   # categorical slot 4 -- vega/IV residual
CRITICAL_RED = "#d03b3b"  # status: drawdown


def _style_axis(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left", pad=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)


def _save(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def to_datetime_col(minute_col: pd.Series) -> pd.Series:
    return pd.to_datetime(minute_col.astype("int64"), unit="m", utc=True)


def enrich_inputs(snapshot: pd.DataFrame, trades: pd.DataFrame, hedge_trades: pd.DataFrame,
                   cand: pd.DataFrame):
    """Add spot/datetime columns to already-loaded (in-memory) frames -- the
    part of load_inputs that doesn't require touching disk, so callers that
    already have the DataFrames (e.g. main.py's minimal-output path) don't
    need to round-trip through CSV files just to get here."""
    snapshot = snapshot.sort_values("minute_timestamp").reset_index(drop=True)
    unique_minutes = snapshot["minute_timestamp"].to_numpy()
    snapshot["spot"] = build_spot_series(cand, unique_minutes)
    snapshot["datetime"] = to_datetime_col(snapshot["minute_timestamp"])
    snapshot["date"] = snapshot["datetime"].dt.date

    if len(trades):
        trades["entry_datetime"] = to_datetime_col(trades["entry_time"])
        trades["exit_datetime"] = to_datetime_col(trades["exit_time"])
    if len(hedge_trades):
        hedge_trades["datetime"] = to_datetime_col(hedge_trades["timestamp"])

    return snapshot, trades, hedge_trades


def load_inputs(snapshot_path: str, trades_path: str, hedge_trades_path: str,
                 candidates_path: str):
    snapshot = pd.read_csv(snapshot_path)
    trades = pd.read_csv(trades_path)
    hedge_trades = pd.read_csv(hedge_trades_path)
    cand = pd.read_parquet(candidates_path)
    return enrich_inputs(snapshot, trades, hedge_trades, cand)


# ---------------------------------------------------------------------------
# Greek PnL decomposition: dV ~= Delta*dS + 0.5*Gamma*dS^2 + Vega*dIV + Theta*dt
# ---------------------------------------------------------------------------
def compute_greek_pnl_decomposition(snapshot: pd.DataFrame) -> pd.DataFrame:
    df = snapshot
    dS = df["spot"].diff().fillna(0.0)

    # Start-of-interval (lagged) greeks earn that interval's PnL.
    delta_lag = df["net_delta_before_hedge"].shift(1).fillna(0.0)
    gamma_lag = df["net_gamma"].shift(1).fillna(0.0)
    theta_lag = df["net_theta"].shift(1).fillna(0.0)

    delta_pnl = delta_lag * dS
    gamma_pnl = 0.5 * gamma_lag * dS ** 2
    theta_pnl = theta_lag * DT_DAYS_PER_MINUTE

    total_option_pnl = df["realized_option_pnl"] + df["unrealized_option_pnl"]
    actual_pnl_change = total_option_pnl.diff().fillna(total_option_pnl.iloc[0])

    # No portfolio-level IV series is persisted, so vega/IV is the residual:
    # whatever delta/gamma/theta don't explain. This also absorbs the
    # settlement-minute gap (last quoted ask vs actual intrinsic value at
    # expiry), which is a real but separate effect -- not a pure vega move.
    vega_iv_and_residual_pnl = actual_pnl_change - delta_pnl - gamma_pnl - theta_pnl

    return pd.DataFrame({
        "minute_timestamp": df["minute_timestamp"],
        "datetime": df["datetime"],
        "delta_pnl_cum": delta_pnl.cumsum(),
        "gamma_pnl_cum": gamma_pnl.cumsum(),
        "theta_pnl_cum": theta_pnl.cumsum(),
        "vega_iv_and_residual_pnl_cum": vega_iv_and_residual_pnl.cumsum(),
        "actual_option_pnl_cum": total_option_pnl,
    })


# ---------------------------------------------------------------------------
# daily_pnl.csv
# ---------------------------------------------------------------------------
def compute_open_positions_per_day(day_end_minutes: np.ndarray, trades: pd.DataFrame) -> np.ndarray:
    if trades.empty:
        return np.zeros(len(day_end_minutes), dtype=int)
    entry = trades["entry_time"].to_numpy()
    exit_ = trades["exit_time"].to_numpy()
    return np.array([int(((entry <= t) & (exit_ > t)).sum()) for t in day_end_minutes])


def build_daily_pnl(snapshot: pd.DataFrame, trades: pd.DataFrame,
                     initial_capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    daily = snapshot.groupby("date").last().reset_index()

    prev_nav = daily["nav"].shift(1).fillna(initial_capital)
    daily["daily_return"] = daily["nav"] / prev_nav - 1.0
    daily["cumulative_pnl"] = daily["nav"] - initial_capital

    total_option_pnl = daily["realized_option_pnl"] + daily["unrealized_option_pnl"]
    daily["option_pnl"] = total_option_pnl.diff().fillna(total_option_pnl.iloc[0])
    daily["hedge_pnl"] = daily["cumulative_hedge_pnl"].diff().fillna(daily["cumulative_hedge_pnl"].iloc[0])
    daily["option_fees"] = daily["cumulative_option_fees"].diff().fillna(daily["cumulative_option_fees"].iloc[0])
    daily["hedge_fees"] = daily["cumulative_hedge_fees"].diff().fillna(daily["cumulative_hedge_fees"].iloc[0])
    daily["total_pnl"] = daily["option_pnl"] + daily["hedge_pnl"] - daily["option_fees"] - daily["hedge_fees"]

    daily["open_positions"] = compute_open_positions_per_day(daily["minute_timestamp"].to_numpy(), trades)

    cols = ["date", "nav", "daily_return", "option_pnl", "hedge_pnl", "option_fees",
            "hedge_fees", "total_pnl", "cumulative_pnl", "net_delta_before_hedge",
            "net_delta_after_hedge", "net_gamma", "net_vega", "net_theta", "open_positions"]
    return daily[cols]


def build_greeks_timeseries(snapshot: pd.DataFrame) -> pd.DataFrame:
    cols = ["minute_timestamp", "datetime", "spot", "net_delta_before_hedge",
            "net_delta_after_hedge", "net_gamma", "net_vega", "net_theta"]
    return snapshot[cols].copy()


# ---------------------------------------------------------------------------
# risk_summary.json
# ---------------------------------------------------------------------------
def collect_data_quality_warnings(trades: pd.DataFrame, hedge_trades: pd.DataFrame,
                                    candidates_meta_path: str) -> list:
    warnings = []
    if os.path.exists(candidates_meta_path):
        with open(candidates_meta_path) as f:
            meta = json.load(f)
        n_dup = meta.get("duplicate_minute_symbol_rows_removed", 0)
        if n_dup:
            warnings.append(
                f"{n_dup} duplicate (minute, symbol) rows removed during preprocessing "
                "(sub-tick snapshot artifacts, not exact duplicates)."
            )
    if len(trades) > 0:
        ratio = len(hedge_trades) / len(trades)
        if ratio > 50:
            warnings.append(
                f"{len(hedge_trades)} hedge trades vs {len(trades)} option trades "
                f"({ratio:.0f}x) -- hedge threshold may be too tight relative to position size."
            )
    warnings.append(
        "Option prices in the source data are BTC-quoted; PnL is converted to USD using "
        "spot at the time of each cash flow. Greek-PnL decomposition's vega/IV component "
        "is a residual (no portfolio IV series persisted) and also absorbs the settlement-"
        "minute gap between last quoted ask and actual intrinsic value at expiry."
    )
    return warnings


def compute_risk_summary(daily: pd.DataFrame, trades: pd.DataFrame, hedge_trades: pd.DataFrame,
                          snapshot: pd.DataFrame, initial_capital: float,
                          data_quality_warnings: list) -> dict:
    nav = daily["nav"].to_numpy()
    daily_returns = daily["daily_return"].to_numpy()

    total_return = float(nav[-1] / initial_capital - 1.0)
    n_days = max(len(daily), 1)
    annualized_return = float((1 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1)

    running_max = np.maximum.accumulate(nav)
    drawdown = nav / running_max - 1.0
    max_drawdown = float(drawdown.min())

    std = daily_returns.std()
    sharpe = float(daily_returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else 0.0
    calmar = float(annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    win_rate = float((trades["option_net_pnl"] > 0).mean()) if len(trades) else 0.0
    total_option_fees = float(trades["option_entry_fee"].sum() + trades["option_exit_fee"].sum()) if len(trades) else 0.0
    total_hedge_fees = float(hedge_trades["hedge_fee"].sum()) if len(hedge_trades) else 0.0

    if len(trades):
        mid = (trades["bid_entry"] + trades["ask_entry"]) / 2
        average_relative_spread = float(((trades["ask_entry"] - trades["bid_entry"]) / mid).mean())
    else:
        average_relative_spread = 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "calmar": calmar,
        "num_option_trades": int(len(trades)),
        "num_hedge_trades": int(len(hedge_trades)),
        "win_rate": win_rate,
        "worst_day_pnl": float(daily["total_pnl"].min()),
        "best_day_pnl": float(daily["total_pnl"].max()),
        "total_option_fees": total_option_fees,
        "total_hedge_fees": total_hedge_fees,
        "average_relative_spread": average_relative_spread,
        "max_abs_net_delta_before_hedge": float(snapshot["net_delta_before_hedge"].abs().max()),
        "max_abs_net_delta_after_hedge": float(snapshot["net_delta_after_hedge"].abs().max()),
        "max_abs_gamma": float(snapshot["net_gamma"].abs().max()),
        "max_abs_vega": float(snapshot["net_vega"].abs().max()),
        "average_theta": float(snapshot["net_theta"].mean()),
        "data_quality_warnings": data_quality_warnings,
    }


# ---------------------------------------------------------------------------
# execution_summary.csv
# ---------------------------------------------------------------------------
def _recompute_pnl_at_entry_price(trades: pd.DataFrame, entry_price: pd.Series):
    """Settlement value doesn't depend on how the option was entered, so only
    the entry-side premium/fee change under a different execution price."""
    size = -trades["quantity"]
    entry_premium_usd = entry_price * trades["spot_entry"]
    settle_price_usd = trades["exit_execution_price"] * trades["spot_exit"]
    gross = size * (entry_premium_usd - settle_price_usd)
    fee = option_fee(entry_premium_usd, size)
    return gross, fee, gross - fee


def build_execution_summary(trades: pd.DataFrame, hedge_trades: pd.DataFrame,
                             initial_capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame([{}])

    mid_entry = (trades["bid_entry"] + trades["ask_entry"]) / 2
    mark_entry = trades["mark_entry"]

    net_actual = trades["option_net_pnl"].sum()
    _, _, net_mid = _recompute_pnl_at_entry_price(trades, mid_entry)
    _, _, net_mark = _recompute_pnl_at_entry_price(trades, mark_entry)

    relative_spread = (trades["ask_entry"] - trades["bid_entry"]) / mid_entry
    option_notional = (trades["bid_entry"] * trades["spot_entry"] * (-trades["quantity"])).sum()
    hedge_notional = hedge_trades["hedge_trade_notional"].sum() if len(hedge_trades) else 0.0

    summary = {
        "num_option_trades": len(trades),
        "num_hedge_trades": len(hedge_trades),
        "average_relative_spread": relative_spread.mean(),
        "average_relative_spread_pct_of_mid": relative_spread.mean() * 100,
        "option_turnover_notional_usd": option_notional,
        "option_turnover_pct_of_capital": option_notional / initial_capital * 100,
        "hedge_turnover_notional_usd": hedge_notional,
        "hedge_turnover_pct_of_capital": hedge_notional / initial_capital * 100,
        "total_option_fees": trades["option_entry_fee"].sum() + trades["option_exit_fee"].sum(),
        "total_hedge_fees": hedge_trades["hedge_fee"].sum() if len(hedge_trades) else 0.0,
        "option_net_pnl_actual_bid_execution": net_actual,
        "option_net_pnl_at_mid_execution": net_mid.sum(),
        "option_net_pnl_at_mark_execution": net_mark.sum(),
        "pnl_improvement_if_executed_at_mid": net_mid.sum() - net_actual,
        "pnl_improvement_if_executed_at_mark": net_mark.sum() - net_actual,
    }
    return pd.DataFrame([summary])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_nav_curve(daily: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(pd.to_datetime(daily["date"]), daily["nav"], color=BLUE, linewidth=2)
    _style_axis(ax, "NAV Curve", "NAV (USD)")
    _save(fig, path)


def plot_drawdown_curve(daily: pd.DataFrame, path: str):
    nav = daily["nav"].to_numpy()
    running_max = np.maximum.accumulate(nav)
    drawdown = nav / running_max - 1.0
    dates = pd.to_datetime(daily["date"])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(dates, drawdown * 100, color=CRITICAL_RED, linewidth=2)
    ax.fill_between(dates, drawdown * 100, 0, color=CRITICAL_RED, alpha=0.15)
    _style_axis(ax, "Drawdown Curve", "Drawdown (%)")
    _save(fig, path)


def plot_options_pnl(snapshot: pd.DataFrame, path: str):
    dt = snapshot["datetime"]
    gross = snapshot["realized_option_pnl"] + snapshot["unrealized_option_pnl"]
    net = gross - snapshot["cumulative_option_fees"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(dt, gross, color=BLUE, linewidth=2, label="Without exec costs (gross)")
    ax.plot(dt, net, color=AQUA, linewidth=2, label="With exec costs (net of fees)")
    ax.axhline(0, color=BASELINE, linewidth=1)
    _style_axis(ax, "Option PnL: Gross vs Net of Fees", "PnL (USD)")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=8, loc="best")
    _save(fig, path)


def plot_hedge_pnl(snapshot: pd.DataFrame, path: str):
    dt = snapshot["datetime"]
    gross = snapshot["cumulative_hedge_pnl"]
    net = gross - snapshot["cumulative_hedge_fees"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(dt, gross, color=BLUE, linewidth=2, label="Without exec costs (gross)")
    ax.plot(dt, net, color=AQUA, linewidth=2, label="With exec costs (net of fees)")
    ax.axhline(0, color=BASELINE, linewidth=1)
    _style_axis(ax, "Hedge PnL: Gross vs Net of Fees", "PnL (USD)")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=8, loc="best")
    _save(fig, path)


def plot_greek_pnl_decomposition(greek_pnl: pd.DataFrame, path: str):
    dt = greek_pnl["datetime"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(dt, greek_pnl["delta_pnl_cum"], color=BLUE, linewidth=1.6, label="Delta")
    ax.plot(dt, greek_pnl["gamma_pnl_cum"], color=AQUA, linewidth=1.6, label="Gamma")
    ax.plot(dt, greek_pnl["theta_pnl_cum"], color=YELLOW, linewidth=1.6, label="Theta")
    ax.plot(dt, greek_pnl["vega_iv_and_residual_pnl_cum"], color=GREEN, linewidth=1.6,
            label="Vega/IV + residual")
    ax.plot(dt, greek_pnl["actual_option_pnl_cum"], color=INK_SECONDARY, linewidth=1.2,
            linestyle="--", label="Actual (total)")
    ax.axhline(0, color=BASELINE, linewidth=1)
    _style_axis(ax, "Option PnL Decomposition by Greek", "Cumulative PnL (USD)")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=8, loc="best", ncol=2)
    _save(fig, path)


def plot_overall_pnl(snapshot: pd.DataFrame, initial_capital: float, path: str):
    dt = snapshot["datetime"]
    overall = snapshot["nav"] - initial_capital

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(dt, overall, color=BLUE, linewidth=2)
    ax.axhline(0, color=BASELINE, linewidth=1)
    _style_axis(ax, "Overall Strategy PnL", "Cumulative PnL (USD)")
    _save(fig, path)


# ---------------------------------------------------------------------------
def run(snapshot_path: str, trades_path: str, hedge_trades_path: str,
        candidates_path: str, candidates_meta_path: str, output_dir: str,
        initial_capital: float = INITIAL_CAPITAL):
    os.makedirs(output_dir, exist_ok=True)

    snapshot, trades, hedge_trades = load_inputs(
        snapshot_path, trades_path, hedge_trades_path, candidates_path)

    daily = build_daily_pnl(snapshot, trades, initial_capital)
    greeks_ts = build_greeks_timeseries(snapshot)
    greek_pnl = compute_greek_pnl_decomposition(snapshot)
    warnings = collect_data_quality_warnings(trades, hedge_trades, candidates_meta_path)
    risk_summary = compute_risk_summary(daily, trades, hedge_trades, snapshot, initial_capital, warnings)
    execution_summary = build_execution_summary(trades, hedge_trades, initial_capital)

    daily.to_csv(os.path.join(output_dir, "daily_pnl.csv"), index=False)
    greeks_ts.to_csv(os.path.join(output_dir, "greeks_timeseries.csv"), index=False)
    execution_summary.to_csv(os.path.join(output_dir, "execution_summary.csv"), index=False)
    with open(os.path.join(output_dir, "risk_summary.json"), "w") as f:
        json.dump(risk_summary, f, indent=2)

    plot_nav_curve(daily, os.path.join(output_dir, "nav_curve.png"))
    plot_drawdown_curve(daily, os.path.join(output_dir, "drawdown_curve.png"))
    plot_options_pnl(snapshot, os.path.join(output_dir, "options_pnl.png"))
    plot_hedge_pnl(snapshot, os.path.join(output_dir, "hedge_pnl.png"))
    plot_greek_pnl_decomposition(greek_pnl, os.path.join(output_dir, "greek_pnl_decomposition.png"))
    plot_overall_pnl(snapshot, initial_capital, os.path.join(output_dir, "overall_pnl.png"))

    print("risk_summary.json:")
    print(json.dumps(risk_summary, indent=2))
    print(f"\nall outputs written to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default="./outputs/minute_snapshot.csv")
    parser.add_argument("--trades", default="./outputs/trades.csv")
    parser.add_argument("--hedge-trades", default="./outputs/hedge_trades.csv")
    parser.add_argument("--candidates", default="./data/candidates.parquet")
    parser.add_argument("--candidates-meta", default="./data/candidates_meta.json")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)
    args = parser.parse_args()

    run(
        snapshot_path=args.snapshot, trades_path=args.trades,
        hedge_trades_path=args.hedge_trades, candidates_path=args.candidates,
        candidates_meta_path=args.candidates_meta, output_dir=args.output_dir,
        initial_capital=args.initial_capital,
    )
