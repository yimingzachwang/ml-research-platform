#!/usr/bin/env python3
"""Equal-Weight Buy-and-Hold (EWBH) Benchmark — 20-ETF Canonical Universe.

Constructs a passive benchmark portfolio using the identical universe, date
range, and market data as the canonical ML experiment:

    showcase_20etf_ridge_rar5d_top5_rt30_rb5

Portfolio rules:
    - Equal initial weights (1/N per asset) at inception
    - One allocation at inception — shares never rebalanced
    - No forecasting, no ranking, no asset selection
    - No signal generation

Implementation note — why a standalone script is required:
    The production backtesting engine (src/portfolio/portfolio_backtest.py)
    implements TARGET-WEIGHT REBALANCING.  The daily portfolio return is
    computed as (w_lagged × r).sum() where w_lagged holds the SAME target
    weights on every period.  There is no share-quantity tracking and no
    mechanism for weights to drift naturally with realized returns.

    True buy-and-hold requires weight drift: after the initial allocation,
    each asset's weight changes every day as its price changes relative to
    the portfolio.  The correct formulation is:

        V_i(t) = V_i(0) × Π(1 + r_i(s))  for s = 1..t
        V(t)   = Σ V_i(t)
        equity(t) = (1/N) × Σ (P_i(t) / P_i(t_0))

    This is equivalent to the arithmetic mean of N individually normalized
    price series, which is what this script computes.

    The production engine is not modified.

Usage:
    python benchmarks/ewbh_benchmark.py

Files created (benchmarks/ewbh/):
    metrics.json          — performance metrics
    returns.parquet       — daily EWBH portfolio returns
    equity_curve.parquet  — portfolio equity curve (starts at 1.0)
    comparison.json       — side-by-side vs canonical ML experiment

No production files, configs, or canonical results are modified.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

# ── Repository path bootstrap ─────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.backtesting.metrics import compute_metrics
from src.portfolio.alignment import align_prices, load_universe

# ── Canonical universe and date range ─────────────────────────────────────────
# Identical to showcase_20etf_ridge_rar5d_top5_rt30_rb5.yaml
UNIVERSE: list[str] = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "TLT", "IEF", "AGG", "HYG", "LQD",
    "GLD", "DBC", "VNQ",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU",
]
START_DATE = "2013-01-01"
END_DATE   = "2024-12-31"

# ── Paths ─────────────────────────────────────────────────────────────────────
_ML_DIR = (
    _REPO
    / "results/experiments"
    / "showcase_20etf_ridge_rar5d_top5_rt30_rb5"
    / "showcase_20etf_ridge_rar5d_top5_rt30_rb5"
)
OUTPUT_DIR = _REPO / "benchmarks" / "ewbh"


# ── Portfolio computation ─────────────────────────────────────────────────────

def compute_ewbh(prices: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute equal-weight buy-and-hold portfolio returns and equity curve.

    Methodology:
        At inception (first row of prices), invest 1/N of the portfolio in
        each of N assets.  Shares are never adjusted.  The portfolio value
        at each date is the arithmetic mean of each asset's price relative
        to its inception price — this is mathematically equivalent to tracking
        each sub-portfolio separately and summing with equal initial weights.

        equity(t) = (1/N) Σ_i [ P_i(t) / P_i(t_0) ]
        r(t)      = equity(t) / equity(t-1) - 1

    Returns:
        (daily_returns, equity_curve) — both indexed by prices.index.
        First return is NaN (no prior day available at inception).
    """
    normalized = prices / prices.iloc[0]          # each series starts at 1.0
    equity     = normalized.mean(axis=1)           # equal-weight average
    returns    = equity.pct_change()               # first row is NaN
    return returns, equity


# ── Additional metrics not in the standard module ─────────────────────────────

def _sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio (downside semi-deviation in denominator)."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    rf_pp = risk_free_rate / periods_per_year
    excess = r - rf_pp
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("nan")
    semi_dev = math.sqrt((downside**2).mean()) * math.sqrt(periods_per_year)
    if semi_dev == 0.0:
        return float("nan")
    ann_ret = float((1.0 + r).prod() ** (periods_per_year / len(r)) - 1.0)
    return float((ann_ret - risk_free_rate) / semi_dev)


def _extend_metrics(returns: pd.Series, equity: pd.Series) -> dict[str, float]:
    """Compute full metric set (standard + Sortino + total return + final value)."""
    r = returns.dropna()
    metrics = compute_metrics(r)
    metrics["sortino_ratio"]        = _sortino_ratio(r)
    metrics["total_return"]         = float(equity.dropna().iloc[-1] - 1.0)
    metrics["final_portfolio_value"] = float(equity.dropna().iloc[-1])
    return metrics


# ── ML reference metrics ──────────────────────────────────────────────────────

def _load_ml_reference() -> dict[str, float]:
    """Load canonical ML experiment metrics and augment with total return."""
    with open(_ML_DIR / "metrics.json") as f:
        ml = json.load(f)
    # Total return and final value are not stored in metrics.json;
    # derive them from the saved equity curve.
    eq = pd.read_parquet(_ML_DIR / "equity_curve.parquet")
    # equity_curve.parquet may have columns [equity_curve] or similar
    eq_series = eq.iloc[:, 0].dropna()
    ml["total_return"]          = float(eq_series.iloc[-1] - 1.0)
    ml["final_portfolio_value"] = float(eq_series.iloc[-1])
    # Compute Sortino from net returns
    ret = pd.read_parquet(_ML_DIR / "returns.parquet")
    ret_series = ret.iloc[:, 0].dropna()
    ml["sortino_ratio"] = _sortino_ratio(ret_series)
    return ml


# ── Reporting ─────────────────────────────────────────────────────────────────

_DISPLAY_ROWS: list[tuple[str, str, str]] = [
    ("Total Return",             "total_return",           "{:>+.2%}"),
    ("CAGR (Annualized Return)", "annualized_return",      "{:>+.2%}"),
    ("Annualized Volatility",    "annualized_volatility",  "{:>.2%}"),
    ("Sharpe Ratio (rf=0%)",     "sharpe_ratio",           "{:>.4f}"),
    ("Sortino Ratio (rf=0%)",    "sortino_ratio",          "{:>.4f}"),
    ("Calmar Ratio",             "calmar_ratio",           "{:>.4f}"),
    ("Maximum Drawdown",         "max_drawdown",           "{:>+.2%}"),
    ("Hit Rate",                 "hit_rate",               "{:>.2%}"),
    ("Final Portfolio Value",    "final_portfolio_value",  "${:>.4f}"),
]


def _fmt(val: float | None, fmt: str) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return fmt.format(val)


def print_report(
    bh: dict[str, float],
    ml: dict[str, float],
    n_assets: int,
    inception: str,
    end: str,
    n_days: int,
) -> None:
    """Print full benchmark report to stdout."""
    w_col = 32
    v_col = 18
    sep = "-" * (w_col + v_col * 2 + 2)

    print()
    print("=" * (w_col + v_col * 2 + 2))
    print("  EQUAL-WEIGHT BUY-AND-HOLD BENCHMARK")
    print(f"  Universe : {n_assets} ETFs (canonical showcase universe)")
    print(f"  Period   : {inception}  →  {end}  ({n_days} trading days)")
    print("=" * (w_col + v_col * 2 + 2))
    print()
    print(f"  {'Metric':<{w_col}}{'EWBH':>{v_col}}{'ML (in-sample)':>{v_col}}")
    print(f"  {sep}")
    for label, key, fmt in _DISPLAY_ROWS:
        bh_str = _fmt(bh.get(key), fmt)
        ml_str = _fmt(ml.get(key), fmt)
        print(f"  {label:<{w_col}}{bh_str:>{v_col}}{ml_str:>{v_col}}")
    print(f"  {sep}")
    print()
    print("  Notes:")
    print("  - EWBH: gross returns (no transaction costs).")
    print("    A one-time ~5 bps cost at inception would reduce total return")
    print("    by < 0.01% — negligible over a 12-year horizon.")
    print("  - ML strategy metrics are from the FULL-PERIOD BACKTEST, which")
    print("    is IN-SAMPLE: the model was trained on the entire 2013-2024")
    print("    history before the backtest was run.  These metrics overstate")
    print("    live performance; see walk-forward validation for OOS results.")
    print("  - Sharpe and Sortino use rf = 0% to match the canonical config.")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Load and align prices ─────────────────────────────────────────────────
    print(f"Loading {len(UNIVERSE)} ETFs from canonical data source...")
    raw = load_universe(UNIVERSE)
    prices = align_prices(raw)

    # 2. Filter to canonical date range ────────────────────────────────────────
    prices = prices.loc[START_DATE:END_DATE]
    inception = prices.index[0].strftime("%Y-%m-%d")
    end_date  = prices.index[-1].strftime("%Y-%m-%d")
    n_days    = len(prices)
    n_assets  = prices.shape[1]
    print(f"Aligned: {n_days} trading days × {n_assets} assets")
    print(f"Inception: {inception}   End: {end_date}")
    print(f"Missing assets: {set(UNIVERSE) - set(prices.columns) or 'none'}\n")

    # 3. Compute buy-and-hold portfolio ────────────────────────────────────────
    bh_returns, bh_equity = compute_ewbh(prices)

    # 4. Compute full metric set ───────────────────────────────────────────────
    bh_metrics = _extend_metrics(bh_returns, bh_equity)

    # 5. Load ML reference ─────────────────────────────────────────────────────
    ml_metrics = _load_ml_reference()

    # 6. Save outputs ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(bh_metrics, f, indent=2)

    bh_returns.to_frame("net_return").to_parquet(OUTPUT_DIR / "returns.parquet")
    bh_equity.to_frame("equity").to_parquet(OUTPUT_DIR / "equity_curve.parquet")

    comparison = {
        "ewbh":       {k: round(v, 6) if isinstance(v, float) and not math.isnan(v) else v
                       for k, v in bh_metrics.items()},
        "ml_backtest": {k: round(v, 6) if isinstance(v, float) and not math.isnan(v) else v
                        for k, v in ml_metrics.items()},
        "metadata": {
            "universe":          UNIVERSE,
            "n_assets":          n_assets,
            "inception":         inception,
            "end_date":          end_date,
            "n_trading_days":    n_days,
            "ewbh_gross":        True,
            "ml_is_in_sample":   True,
            "risk_free_rate":    0.0,
        },
    }
    with open(OUTPUT_DIR / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # 7. Print report ──────────────────────────────────────────────────────────
    print_report(bh_metrics, ml_metrics, n_assets, inception, end_date, n_days)
    print(f"Results saved to: {OUTPUT_DIR.relative_to(_REPO)}/")
    print(f"  metrics.json        — EWBH performance metrics")
    print(f"  returns.parquet     — daily returns series")
    print(f"  equity_curve.parquet — equity curve (starts at 1.0)")
    print(f"  comparison.json     — full side-by-side comparison")


if __name__ == "__main__":
    main()
