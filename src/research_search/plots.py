"""Phase 6A: Leaderboard plots.

Produces five required visualisations:
  1. leaderboard_bar.png      — Robustness score bar chart (all candidates)
  2. sharpe_vs_drawdown.png   — OOS Sharpe vs max drawdown scatter
  3. split_stability.png      — Split Sharpe mean ± std per candidate
  4. equity_curve_top3.png    — Equity curve comparison (top 3 by robustness)
  5. drawdown_top3.png        — Drawdown comparison (top 3 by robustness)

All plots skip gracefully when data or dependencies are missing.
Requires matplotlib and, for equity/drawdown, pandas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .leaderboard import LeaderboardRow

# ---------------------------------------------------------------------------
# Colour palette (consistent across plots)
# ---------------------------------------------------------------------------

_BASELINE_COLOR = "#2c3e50"  # dark navy for baseline
_RIDGE_COLOR = "#2980b9"     # blue
_LASSO_COLOR = "#27ae60"     # green
_ELASTIC_COLOR = "#e67e22"   # orange
_NOT_RUN_COLOR = "#bdc3c7"   # grey for missing

_MODEL_COLORS = {
    "ridge": _RIDGE_COLOR,
    "lasso": _LASSO_COLOR,
    "elasticnet": _ELASTIC_COLOR,
}

_TOP3_COLORS = ["#c0392b", "#2980b9", "#27ae60"]  # red, blue, green


def _bar_color(row: LeaderboardRow) -> str:
    if row.is_baseline:
        return _BASELINE_COLOR
    if row.status != "completed":
        return _NOT_RUN_COLOR
    return _MODEL_COLORS.get(row.model_id, "#7f8c8d")


# ---------------------------------------------------------------------------
# 1. Leaderboard bar chart
# ---------------------------------------------------------------------------


def plot_leaderboard_bar(
    rows: list[LeaderboardRow],
    output_path: Path,
) -> Path | None:
    """Horizontal bar chart of robustness scores, grouped by model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return None

    completed = [r for r in rows if r.status == "completed" and r.robustness_score is not None]
    if not completed:
        return None

    # Sort completed by robustness score desc for plot
    completed_sorted = sorted(completed, key=lambda r: r.robustness_score or 0.0, reverse=True)

    labels = [r.candidate_id for r in completed_sorted]
    scores = [r.robustness_score or 0.0 for r in completed_sorted]
    colors = [_bar_color(r) for r in completed_sorted]

    fig_height = max(6, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    bars = ax.barh(labels, scores, color=colors, edgecolor="white", linewidth=0.5)

    # Mark best candidates
    for i, row in enumerate(completed_sorted):
        suffix = ""
        if row.best_robustness:
            suffix += " ★"
        if row.best_sharpe:
            suffix += " ◆"
        if row.is_baseline:
            suffix += " [baseline]"
        if suffix:
            ax.text(
                scores[i] + 0.005,
                i,
                suffix.strip(),
                va="center",
                fontsize=8,
                color="#2c3e50",
            )

    ax.axvline(0, color="#7f8c8d", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Phase 6A Robustness Score")
    ax.set_title("Phase 6A Strategy Search — Robustness Leaderboard", fontsize=13)
    ax.invert_yaxis()

    # Legend
    legend_handles = [
        mpatches.Patch(color=_RIDGE_COLOR, label="RidgeRegression"),
        mpatches.Patch(color=_LASSO_COLOR, label="LassoRegression"),
        mpatches.Patch(color=_ELASTIC_COLOR, label="ElasticNetRegression"),
        mpatches.Patch(color=_BASELINE_COLOR, label="Baseline"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    ax.text(
        0.01, -0.04,
        "★ = best robustness   ◆ = best OOS Sharpe (reference only)",
        transform=ax.transAxes, fontsize=8, color="#7f8c8d",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 2. Sharpe vs max drawdown scatter
# ---------------------------------------------------------------------------


def plot_sharpe_vs_drawdown(
    rows: list[LeaderboardRow],
    output_path: Path,
) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    plottable = [
        r for r in rows
        if r.status == "completed"
        and r.oos_sharpe is not None
        and r.max_drawdown_pct is not None
    ]
    if not plottable:
        return None

    fig, ax = plt.subplots(figsize=(9, 6))

    for r in plottable:
        color = _bar_color(r)
        marker = "D" if r.is_baseline else ("*" if r.best_robustness else "o")
        size = 120 if (r.best_robustness or r.best_sharpe or r.is_baseline) else 60
        ax.scatter(
            r.max_drawdown_pct,
            r.oos_sharpe,
            c=color,
            s=size,
            marker=marker,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
        )
        if r.best_robustness or r.best_sharpe or r.is_baseline:
            ax.annotate(
                r.candidate_id,
                (r.max_drawdown_pct, r.oos_sharpe),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
            )

    ax.axhline(0, color="#7f8c8d", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("Mean OOS Sharpe")
    ax.set_title("Phase 6A: OOS Sharpe vs Max Drawdown", fontsize=12)
    ax.text(
        0.01, 0.01,
        "★ best robustness   ◆ best Sharpe   ◇ baseline   "
        "Top-right = better",
        transform=ax.transAxes, fontsize=7, color="#7f8c8d",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 3. Split stability plot
# ---------------------------------------------------------------------------


def plot_split_stability(
    rows: list[LeaderboardRow],
    output_path: Path,
) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    plottable = [
        r for r in rows
        if r.status == "completed"
        and r.split_sharpe_mean is not None
        and r.split_sharpe_std is not None
    ]
    if not plottable:
        return None

    sorted_rows = sorted(plottable, key=lambda r: r.split_sharpe_mean or 0.0, reverse=True)
    means = [r.split_sharpe_mean for r in sorted_rows]
    stds = [r.split_sharpe_std for r in sorted_rows]
    labels = [r.candidate_id for r in sorted_rows]
    colors = [_bar_color(r) for r in sorted_rows]

    fig_height = max(6, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    xs = list(range(len(sorted_rows)))
    ax.barh(xs, means, xerr=stds, color=colors, align="center",
            error_kw={"elinewidth": 0.8, "ecolor": "#7f8c8d", "capsize": 3},
            edgecolor="white", linewidth=0.4)
    ax.set_yticks(xs)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="#7f8c8d", linewidth=0.8, linestyle="--")
    ax.set_xlabel("OOS Sharpe (mean ± std across splits)")
    ax.set_title("Phase 6A: Split Sharpe Stability", fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 4 & 5. Equity curve + drawdown for top 3 robustness candidates
# ---------------------------------------------------------------------------


def _load_parquet(path: Path) -> "Any | None":
    """Load a parquet file; return None if unavailable."""
    if not path.exists():
        return None
    try:
        import pandas as pd
        return pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return None


def _top3_candidates(rows: list[LeaderboardRow]) -> list[LeaderboardRow]:
    completed = [r for r in rows if r.status == "completed" and r.robustness_score is not None]
    completed_sorted = sorted(completed, key=lambda r: r.robustness_score or 0.0, reverse=True)
    return completed_sorted[:3]


def plot_equity_curves_top3(
    rows: list[LeaderboardRow],
    results_base: Path,
    output_path: Path,
) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    from src.orchestration.utils.filesystem import equity_curve_path

    top3 = _top3_candidates(rows)
    if not top3:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    any_plotted = False

    for i, row in enumerate(top3):
        eq_path = results_base / row.candidate_id / "equity_curve.parquet"
        df = _load_parquet(eq_path)
        if df is None:
            continue
        # Normalise to starting value of 1.0
        col = df.columns[0] if len(df.columns) > 0 else None
        if col is None:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        series = series / series.iloc[0]
        label = f"{row.candidate_id}"
        if row.best_robustness:
            label += " ★"
        ax.plot(series.index, series.values, color=_TOP3_COLORS[i], label=label, linewidth=1.4)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return None

    ax.axhline(1.0, color="#7f8c8d", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Portfolio Value (normalised)")
    ax.set_title("Phase 6A: Equity Curves — Top 3 Robustness Candidates", fontsize=12)
    ax.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_drawdown_top3(
    rows: list[LeaderboardRow],
    results_base: Path,
    output_path: Path,
) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    top3 = _top3_candidates(rows)
    if not top3:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    any_plotted = False

    for i, row in enumerate(top3):
        eq_path = results_base / row.candidate_id / "equity_curve.parquet"
        df = _load_parquet(eq_path)
        if df is None:
            continue
        col = df.columns[0] if len(df.columns) > 0 else None
        if col is None:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        running_max = series.cummax()
        drawdown = (series - running_max) / running_max * 100.0
        label = f"{row.candidate_id}"
        if row.best_robustness:
            label += " ★"
        ax.fill_between(
            drawdown.index, drawdown.values, 0,
            color=_TOP3_COLORS[i], alpha=0.4, label=label,
        )
        ax.plot(drawdown.index, drawdown.values, color=_TOP3_COLORS[i], linewidth=0.8)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return None

    ax.axhline(0, color="#7f8c8d", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Phase 6A: Drawdowns — Top 3 Robustness Candidates", fontsize=12)
    ax.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# High-level: generate all 5 plots
# ---------------------------------------------------------------------------


def generate_all_plots(
    rows: list[LeaderboardRow],
    plots_dir: Path,
    results_base: Path | None = None,
) -> dict[str, Path | None]:
    """Generate all 5 required plots.  Returns map of stem → path (or None)."""
    from src.orchestration.utils.filesystem import experiments_root

    rb = results_base or experiments_root()

    results = {
        "leaderboard_bar": plot_leaderboard_bar(
            rows, plots_dir / "leaderboard_bar.png"
        ),
        "sharpe_vs_drawdown_scatter": plot_sharpe_vs_drawdown(
            rows, plots_dir / "sharpe_vs_drawdown_scatter.png"
        ),
        "split_stability": plot_split_stability(
            rows, plots_dir / "split_stability.png"
        ),
        "equity_curve_top3": plot_equity_curves_top3(
            rows, rb, plots_dir / "equity_curve_top3.png"
        ),
        "drawdown_top3": plot_drawdown_top3(
            rows, rb, plots_dir / "drawdown_top3.png"
        ),
    }
    return results
