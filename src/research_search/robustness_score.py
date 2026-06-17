"""Phase 6A: Robustness scoring formula.

Transparent, deterministic composite score that rewards out-of-sample Sharpe
and penalises drawdown, split instability, catastrophic splits, and missing
diagnostics.  Each component is returned individually for full auditability.

This is a Phase 6A–specific formula distinct from the existing
``_robustness_score`` in ``src.orchestration.api.research_api``.  That formula
is used by the general ``rank_experiment_variants`` tool.  This one is richer
(split-stability aware) and is only used by the leaderboard.

Formula
-------
score = (
      W_OOS_SHARPE × clamp(mean_oos_sharpe, −3, 3)
    + W_DRAWDOWN   × clamp(max_drawdown_pct / 100, −1, 0)   # dd is negative
    − W_SPLIT_STD  × clamp(std_oos_sharpe, 0, 3)
    − W_CATASTRO   × (1 if worst_split_sharpe < CATASTROPHIC_THRESHOLD else 0)
    − W_NEG_SPLITS × (n_negative_splits / n_splits)
    − W_MISSING    × missing_diagnostics_count
)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Weights (must sum to 1.0 for positive terms + 1.0 for penalty caps)
# ---------------------------------------------------------------------------

W_OOS_SHARPE: float = 0.40
W_DRAWDOWN: float = 0.25
W_SPLIT_STD: float = 0.15
W_CATASTRO: float = 0.10
W_NEG_SPLITS: float = 0.10
W_MISSING: float = 0.02  # per missing diagnostic

CATASTROPHIC_THRESHOLD: float = -0.5  # worst_split_sharpe below this = catastrophic

FORMULA_DESCRIPTION: str = (
    "Phase 6A Robustness Score = "
    f"{W_OOS_SHARPE}×clamp(mean_oos_sharpe,−3,3) "
    f"+ {W_DRAWDOWN}×clamp(max_drawdown_pct/100,−1,0) "
    f"− {W_SPLIT_STD}×clamp(std_oos_sharpe,0,3) "
    f"− {W_CATASTRO}×(1 if worst_split_sharpe<{CATASTROPHIC_THRESHOLD}) "
    f"− {W_NEG_SPLITS}×(n_negative_splits/n_splits) "
    f"− {W_MISSING}×missing_diagnostics_count"
)

WEIGHTS: dict[str, float] = {
    "oos_sharpe": W_OOS_SHARPE,
    "drawdown": W_DRAWDOWN,
    "split_std": W_SPLIT_STD,
    "catastrophic": W_CATASTRO,
    "negative_splits": W_NEG_SPLITS,
    "missing_diagnostics": W_MISSING,
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class RobustnessComponents:
    """Full breakdown of Phase 6A robustness score.

    All components are individually inspectable for transparency.
    ``total`` equals the sum of positive contributions minus all penalties.
    """
    oos_sharpe_contribution: float
    drawdown_contribution: float     # <= 0 (penalty)
    split_instability_penalty: float  # >= 0
    catastrophic_split_penalty: float  # 0 or W_CATASTRO
    negative_split_penalty: float     # >= 0
    missing_diagnostics_penalty: float  # >= 0
    total: float
    inputs_used: dict[str, Any]
    weights: dict[str, float] = field(default_factory=lambda: dict(WEIGHTS))

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "oos_sharpe_contribution": self.oos_sharpe_contribution,
            "drawdown_contribution": self.drawdown_contribution,
            "split_instability_penalty": self.split_instability_penalty,
            "catastrophic_split_penalty": self.catastrophic_split_penalty,
            "negative_split_penalty": self.negative_split_penalty,
            "missing_diagnostics_penalty": self.missing_diagnostics_penalty,
            "inputs_used": self.inputs_used,
            "weights": self.weights,
            "formula": FORMULA_DESCRIPTION,
        }


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def compute_robustness_score(
    oos_sharpe: float | None = None,
    max_drawdown_pct: float | None = None,
    std_oos_sharpe: float | None = None,
    worst_split_sharpe: float | None = None,
    n_negative_splits: int | None = None,
    n_splits: int | None = None,
    missing_diagnostics: int = 0,
) -> RobustnessComponents | None:
    """Compute the Phase 6A robustness score.

    Parameters
    ----------
    oos_sharpe:
        Mean OOS Sharpe across walk-forward splits.
    max_drawdown_pct:
        Max drawdown as a percentage (expected negative, e.g. −25.0 for −25 %).
        If supplied as a "−31.20%" string use ``_parse_pct_input`` first.
    std_oos_sharpe:
        Standard deviation of OOS Sharpe across splits.
    worst_split_sharpe:
        Minimum OOS Sharpe across all splits (most negative).
    n_negative_splits:
        Number of splits with negative OOS Sharpe.
    n_splits:
        Total number of walk-forward splits.
    missing_diagnostics:
        Count of expected diagnostic fields that were not found on disk.

    Returns None when all primary metric inputs (oos_sharpe, max_drawdown_pct,
    std_oos_sharpe) are None (no data available).
    """
    if oos_sharpe is None and max_drawdown_pct is None and std_oos_sharpe is None:
        return None

    # OOS Sharpe contribution
    oos_contrib = 0.0
    if oos_sharpe is not None:
        oos_contrib = W_OOS_SHARPE * max(-3.0, min(3.0, float(oos_sharpe)))

    # Drawdown contribution (negative term — lower drawdown → less negative)
    dd_contrib = 0.0
    if max_drawdown_pct is not None:
        dd_float = float(max_drawdown_pct)
        dd_contrib = W_DRAWDOWN * max(-1.0, min(0.0, dd_float / 100.0))

    # Split instability penalty
    split_std_penalty = 0.0
    if std_oos_sharpe is not None:
        split_std_penalty = W_SPLIT_STD * max(0.0, min(3.0, float(std_oos_sharpe)))

    # Catastrophic split penalty
    catastro_penalty = 0.0
    if worst_split_sharpe is not None and float(worst_split_sharpe) < CATASTROPHIC_THRESHOLD:
        catastro_penalty = W_CATASTRO

    # Negative-split proportion penalty
    neg_penalty = 0.0
    if (
        n_negative_splits is not None
        and n_splits is not None
        and int(n_splits) > 0
    ):
        neg_penalty = W_NEG_SPLITS * (int(n_negative_splits) / int(n_splits))

    # Missing-diagnostics penalty
    miss_penalty = W_MISSING * max(0, int(missing_diagnostics))

    total = (
        oos_contrib
        + dd_contrib
        - split_std_penalty
        - catastro_penalty
        - neg_penalty
        - miss_penalty
    )

    return RobustnessComponents(
        oos_sharpe_contribution=round(oos_contrib, 4),
        drawdown_contribution=round(dd_contrib, 4),
        split_instability_penalty=round(split_std_penalty, 4),
        catastrophic_split_penalty=round(catastro_penalty, 4),
        negative_split_penalty=round(neg_penalty, 4),
        missing_diagnostics_penalty=round(miss_penalty, 4),
        total=round(total, 4),
        inputs_used={
            "oos_sharpe": oos_sharpe,
            "max_drawdown_pct": max_drawdown_pct,
            "std_oos_sharpe": std_oos_sharpe,
            "worst_split_sharpe": worst_split_sharpe,
            "n_negative_splits": n_negative_splits,
            "n_splits": n_splits,
            "missing_diagnostics": missing_diagnostics,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_pct_input(value: Any) -> float | None:
    """Parse a percentage value from either a float or a '−31.20%' string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if s.endswith("%"):
            try:
                return float(s[:-1])
            except ValueError:
                return None
    return None
