"""Phase 6A: Leaderboard construction and ranking.

Reads persisted artefacts for each search candidate and builds a ranked
leaderboard.  Ranking is by Phase 6A robustness_score, not raw Sharpe alone.
Best Sharpe and best robustness candidates are flagged separately.

Governance:
  - Failed experiments are preserved (status="failed"), never hidden.
  - Not-yet-run experiments appear as status="not_run".
  - No experiment is ever selected by Sharpe alone.
  - Baseline is always included.
  - All metric values are read from disk, never invented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .robustness_score import (
    RobustnessComponents,
    compute_robustness_score,
    parse_pct_input,
)
from .search_space import BASE_EXPERIMENT, SearchCandidate, SearchSpace

# ---------------------------------------------------------------------------
# Leaderboard row schema
# ---------------------------------------------------------------------------


@dataclass
class LeaderboardRow:
    rank: int
    candidate_id: str
    model_id: str
    model_type: str
    feature_set_id: str
    feature_set_name: str
    cost_bps: int
    base_experiment: str
    # Performance metrics (None = not run or missing)
    oos_sharpe: float | None        # mean OOS Sharpe across splits
    annual_return_pct: float | None
    max_drawdown_pct: float | None  # float, e.g. −25.0
    hit_rate_pct: float | None
    # Split diagnostics
    split_sharpe_mean: float | None
    split_sharpe_std: float | None
    worst_split_sharpe: float | None
    n_negative_splits: int | None
    n_splits: int | None
    # Scoring
    robustness_score: float | None
    robustness_components: RobustnessComponents | None
    # Quality
    status: str  # "completed" | "not_run" | "failed"
    warnings: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    missing_metrics: list[str] = field(default_factory=list)
    # Leaderboard flags
    is_baseline: bool = False
    best_sharpe: bool = False
    best_robustness: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            "model_id": self.model_id,
            "model_type": self.model_type,
            "feature_set_id": self.feature_set_id,
            "feature_set_name": self.feature_set_name,
            "cost_bps": self.cost_bps,
            "base_experiment": self.base_experiment,
            "oos_sharpe": self.oos_sharpe,
            "annual_return_pct": self.annual_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "hit_rate_pct": self.hit_rate_pct,
            "split_sharpe_mean": self.split_sharpe_mean,
            "split_sharpe_std": self.split_sharpe_std,
            "worst_split_sharpe": self.worst_split_sharpe,
            "n_negative_splits": self.n_negative_splits,
            "n_splits": self.n_splits,
            "robustness_score": self.robustness_score,
            "status": self.status,
            "warnings": self.warnings,
            "failure_modes": self.failure_modes,
            "missing_metrics": self.missing_metrics,
            "is_baseline": self.is_baseline,
            "best_sharpe": self.best_sharpe,
            "best_robustness": self.best_robustness,
        }
        if self.robustness_components is not None:
            d["robustness_components"] = self.robustness_components.to_dict()
        return d

    def to_csv_row(self) -> dict[str, Any]:
        """Flat dict suitable for csv.DictWriter."""
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            "model_id": self.model_id,
            "feature_set_id": self.feature_set_id,
            "feature_set_name": self.feature_set_name,
            "cost_bps": self.cost_bps,
            "oos_sharpe": _fmt(self.oos_sharpe),
            "annual_return_pct": _fmt(self.annual_return_pct),
            "max_drawdown_pct": _fmt(self.max_drawdown_pct),
            "hit_rate_pct": _fmt(self.hit_rate_pct),
            "split_sharpe_mean": _fmt(self.split_sharpe_mean),
            "split_sharpe_std": _fmt(self.split_sharpe_std),
            "worst_split_sharpe": _fmt(self.worst_split_sharpe),
            "n_negative_splits": self.n_negative_splits,
            "n_splits": self.n_splits,
            "robustness_score": _fmt(self.robustness_score),
            "status": self.status,
            "is_baseline": self.is_baseline,
            "best_sharpe": self.best_sharpe,
            "best_robustness": self.best_robustness,
            "failure_modes": "; ".join(self.failure_modes),
            "warnings": "; ".join(self.warnings),
        }


def _fmt(v: float | None) -> str:
    if v is None:
        return ""
    return f"{v:.4f}"


def _frac_to_pct(v: float | None) -> float | None:
    """Convert a fractional metric (e.g. -0.27) to percent form (-27.0).

    Only converts when the absolute value is <= 1.0 to avoid double-converting
    values already stored in percent form.  Returns None when input is None.
    """
    if v is None:
        return None
    return v * 100.0 if abs(v) <= 1.0 else v


# ---------------------------------------------------------------------------
# Artefact loading helpers
# ---------------------------------------------------------------------------


def _load_json_safe(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _load_metrics(experiment_name: str, results_base: Path) -> dict[str, Any]:
    """Load metrics.json for the experiment."""
    path = results_base / experiment_name / "metrics.json"
    return _load_json_safe(path) or {}


def _load_split_metrics(experiment_name: str, results_base: Path) -> dict[str, Any]:
    """Load diagnostics/split_metrics.json."""
    path = results_base / experiment_name / "diagnostics" / "split_metrics.json"
    return _load_json_safe(path) or {}


def _load_metadata(experiment_name: str, results_base: Path) -> dict[str, Any]:
    path = results_base / experiment_name / "metadata.json"
    return _load_json_safe(path) or {}


def _detect_failure_modes(experiment_name: str, results_base: Path) -> list[str]:
    """Simple heuristic failure-mode detection from artefacts."""
    try:
        from src.orchestration.context.failure_mode_detector import detect_failure_modes
        from src.orchestration.context.context_builder import build_context
        ctx = build_context(experiment_name, base=results_base)
        return [fm.get("name") for fm in (ctx.failure_modes or []) if fm.get("name")]
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# Single candidate result loading
# ---------------------------------------------------------------------------


def _load_candidate_metrics(
    candidate: SearchCandidate,
    results_base: Path,
) -> tuple[str, dict[str, Any], dict[str, Any], list[str]]:
    """Return (status, metrics, split_summary, failure_modes) for one candidate.

    status: "completed" | "not_run"
    """
    exp_dir = results_base / candidate.candidate_id
    if not exp_dir.exists():
        return "not_run", {}, {}, []

    metrics = _load_metrics(candidate.candidate_id, results_base)
    if not metrics:
        return "not_run", {}, {}, []

    split_raw = _load_split_metrics(candidate.candidate_id, results_base)
    summary = split_raw.get("summary", {})
    splits = split_raw.get("splits", [])

    sharpe_vals = [
        s.get("sharpe_ratio") for s in splits if s.get("sharpe_ratio") is not None
    ]
    split_summary = {
        "mean_oos_sharpe": summary.get("mean_sharpe"),
        "std_oos_sharpe": summary.get("std_sharpe"),
        "n_splits": summary.get("n_splits", len(splits)),
        "n_negative_sharpe_splits": sum(1 for v in sharpe_vals if v < 0),
        "worst_split_sharpe": min(sharpe_vals) if sharpe_vals else None,
        "per_split_sharpes": sharpe_vals,
    }

    failure_modes = _detect_failure_modes(candidate.candidate_id, results_base)

    return "completed", metrics, split_summary, failure_modes


# ---------------------------------------------------------------------------
# LeaderboardRow construction
# ---------------------------------------------------------------------------


def _build_row(
    candidate: SearchCandidate,
    results_base: Path,
) -> LeaderboardRow:
    status, metrics, split_summary, failure_modes = _load_candidate_metrics(
        candidate, results_base
    )

    raw_sharpe = metrics.get("sharpe_ratio")
    # metrics.json stores fractional values (e.g. -0.27, 0.048, 0.481).
    # Fall back to the fraction key and convert to percent so the robustness
    # formula and leaderboard display use the expected %-scale numbers.
    raw_return = metrics.get("annualized_return_pct") or _frac_to_pct(
        metrics.get("annualized_return")
    )
    raw_dd = metrics.get("max_drawdown_pct") or _frac_to_pct(
        metrics.get("max_drawdown")
    )
    raw_hit = metrics.get("hit_rate_pct") or _frac_to_pct(
        metrics.get("hit_rate")
    )

    dd_float = parse_pct_input(raw_dd)
    return_float = parse_pct_input(raw_return)
    hit_float = parse_pct_input(raw_hit)

    oos_sharpe = split_summary.get("mean_oos_sharpe")
    std_oos = split_summary.get("std_oos_sharpe")
    n_splits = split_summary.get("n_splits")
    n_neg = split_summary.get("n_negative_sharpe_splits")
    worst = split_summary.get("worst_split_sharpe")

    # Count missing diagnostics
    missing: list[str] = []
    if oos_sharpe is None:
        missing.append("mean_oos_sharpe")
    if dd_float is None:
        missing.append("max_drawdown_pct")
    if std_oos is None:
        missing.append("std_oos_sharpe")
    if raw_sharpe is None:
        missing.append("sharpe_ratio")

    rob = compute_robustness_score(
        oos_sharpe=oos_sharpe,
        max_drawdown_pct=dd_float,
        std_oos_sharpe=std_oos,
        worst_split_sharpe=worst,
        n_negative_splits=n_neg,
        n_splits=n_splits,
        missing_diagnostics=len(missing),
    )

    # Warnings
    warnings: list[str] = []
    if status == "not_run":
        warnings.append("Experiment has not been run yet.")
    else:
        if raw_sharpe is None:
            warnings.append("sharpe_ratio missing from metrics.json.")
        if oos_sharpe is None:
            warnings.append("OOS Sharpe unavailable (split_metrics not found).")
        if worst is not None and worst < -0.5:
            warnings.append(f"Catastrophic split detected (worst={worst:.2f}).")
        if n_neg is not None and n_splits is not None and n_splits > 0:
            pct = n_neg / n_splits
            if pct >= 0.5:
                warnings.append(
                    f"Majority of splits negative ({n_neg}/{n_splits})."
                )

    return LeaderboardRow(
        rank=0,  # filled in after sorting
        candidate_id=candidate.candidate_id,
        model_id=candidate.model.id,
        model_type=candidate.model.type,
        feature_set_id=candidate.feature_set.id,
        feature_set_name=candidate.feature_set.name,
        cost_bps=candidate.cost_bps,
        base_experiment=candidate.base_experiment,
        oos_sharpe=oos_sharpe,
        annual_return_pct=return_float,
        max_drawdown_pct=dd_float,
        hit_rate_pct=hit_float,
        split_sharpe_mean=oos_sharpe,
        split_sharpe_std=std_oos,
        worst_split_sharpe=worst,
        n_negative_splits=n_neg,
        n_splits=n_splits,
        robustness_score=rob.total if rob else None,
        robustness_components=rob,
        status=status,
        warnings=warnings,
        failure_modes=failure_modes,
        missing_metrics=missing,
        is_baseline=candidate.is_baseline,
    )


# ---------------------------------------------------------------------------
# Leaderboard assembly
# ---------------------------------------------------------------------------


def build_leaderboard(
    results_base: Path | str | None = None,
    candidates: list[SearchCandidate] | None = None,
) -> list[LeaderboardRow]:
    """Build the full leaderboard from disk artefacts.

    Parameters
    ----------
    results_base:
        Root of results/experiments/ tree.  Defaults to ``results/experiments``.
    candidates:
        Override candidate list (default: ``SearchSpace.all_candidates_with_baseline()``).

    Returns a list of LeaderboardRow objects sorted by robustness_score
    (descending).  Not-yet-run candidates are appended after completed ones.
    Baseline is always present.  Failed candidates are never hidden.
    """
    from src.orchestration.utils.filesystem import experiments_root

    rb = Path(results_base) if results_base else experiments_root()
    all_candidates = candidates or SearchSpace.all_candidates_with_baseline()

    rows = [_build_row(c, rb) for c in all_candidates]

    # Sort: completed rows by robustness_score desc; not_run after
    completed = [r for r in rows if r.status == "completed"]
    not_run = [r for r in rows if r.status == "not_run"]

    def _sort_key(r: LeaderboardRow) -> float:
        if r.robustness_score is None:
            return float("-inf")
        return float(r.robustness_score)

    completed.sort(key=_sort_key, reverse=True)

    # Assign ranks
    sorted_rows = completed + not_run
    for i, row in enumerate(sorted_rows, 1):
        row.rank = i

    # Mark flags (only among completed rows with real scores)
    _assign_flags(sorted_rows)

    return sorted_rows


def _assign_flags(rows: list[LeaderboardRow]) -> None:
    """Mutate rows in-place to set best_sharpe and best_robustness flags."""
    completed = [r for r in rows if r.status == "completed"]

    best_sharpe_row = None
    best_rob_row = None

    for r in completed:
        if r.oos_sharpe is not None:
            if best_sharpe_row is None or r.oos_sharpe > best_sharpe_row.oos_sharpe:
                best_sharpe_row = r
        if r.robustness_score is not None:
            if best_rob_row is None or r.robustness_score > best_rob_row.robustness_score:
                best_rob_row = r

    if best_sharpe_row is not None:
        best_sharpe_row.best_sharpe = True
    if best_rob_row is not None:
        best_rob_row.best_robustness = True


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def leaderboard_to_dict(rows: list[LeaderboardRow]) -> dict[str, Any]:
    from .robustness_score import FORMULA_DESCRIPTION, WEIGHTS

    completed = [r for r in rows if r.status == "completed"]
    not_run = [r for r in rows if r.status != "completed"]

    best_sharpe = next((r for r in rows if r.best_sharpe), None)
    best_rob = next((r for r in rows if r.best_robustness), None)

    no_robust_candidate = (
        best_rob is None
        or best_rob.robustness_score is None
        or best_rob.robustness_score < 0
    )

    return {
        "search_phase": "6A",
        "n_total": len(rows),
        "n_completed": len(completed),
        "n_not_run": len(not_run),
        "best_oos_sharpe_candidate": best_sharpe.candidate_id if best_sharpe else None,
        "best_robustness_candidate": best_rob.candidate_id if best_rob else None,
        "no_robust_candidate_warning": no_robust_candidate,
        "robustness_formula": FORMULA_DESCRIPTION,
        "robustness_weights": WEIGHTS,
        "rows": [r.to_dict() for r in rows],
    }


def write_leaderboard_json(
    rows: list[LeaderboardRow],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "leaderboard.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(leaderboard_to_dict(rows), fh, indent=2)
    return path


def write_leaderboard_csv(
    rows: list[LeaderboardRow],
    output_dir: Path,
) -> Path:
    import csv

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "leaderboard.csv"
    if not rows:
        path.write_text("")
        return path
    fieldnames = list(rows[0].to_csv_row().keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())
    return path


def leaderboard_to_markdown_table(rows: list[LeaderboardRow]) -> str:
    """Return a concise markdown table (top rows only for readability)."""
    headers = [
        "Rank", "Candidate", "Model", "Features", "Cost bps",
        "OOS Sharpe", "Max DD%", "Robust Score", "Status", "Flags",
    ]
    sep = [":---:"] + ["---"] * (len(headers) - 1)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in rows:
        flags = []
        if r.is_baseline:
            flags.append("baseline")
        if r.best_robustness:
            flags.append("★ best robustness")
        if r.best_sharpe:
            flags.append("◆ best Sharpe")
        lines.append(
            f"| {r.rank} "
            f"| {r.candidate_id} "
            f"| {r.model_id} "
            f"| {r.feature_set_id} "
            f"| {r.cost_bps} "
            f"| {_fmt(r.oos_sharpe)} "
            f"| {_fmt(r.max_drawdown_pct)} "
            f"| {_fmt(r.robustness_score)} "
            f"| {r.status} "
            f"| {', '.join(flags) if flags else '—'} |"
        )
    return "\n".join(lines)
