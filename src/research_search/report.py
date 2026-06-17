"""Phase 6A: Markdown report and website-ready summary generator.

Produces:
  - Full markdown research report  (search_report.md)
  - Concise website-ready summary  (website_summary.md)

Both outputs are derived purely from leaderboard data and the search manifest.
No LLM, no execution, no external HTTP calls.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from .leaderboard import LeaderboardRow, leaderboard_to_markdown_table
from .robustness_score import FORMULA_DESCRIPTION

# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

_SHARPE_WARNING = (
    "> **Warning:** Do not select a strategy based on OOS Sharpe alone. "
    "A high Sharpe candidate may be unstable across splits, "
    "exhibit high drawdown, or fail under higher transaction costs. "
    "Use the robustness score to identify robust candidates. "
    "A candidate with the highest Sharpe is not necessarily the best choice."
)


def _header(title: str, level: int = 1) -> str:
    return f"{'#' * level} {title}"


def _format_float(v: float | None, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}" if v is not None else "n/a"


def _flags(row: LeaderboardRow) -> str:
    flags = []
    if row.is_baseline:
        flags.append("**[BASELINE]**")
    if row.best_robustness:
        flags.append("**[BEST ROBUSTNESS]**")
    if row.best_sharpe:
        flags.append("**[BEST OOS SHARPE]**")
    return " ".join(flags)


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


def build_full_report(
    rows: list[LeaderboardRow],
    manifest: dict[str, Any] | None = None,
    plots_dir: Path | None = None,
    generated_at: str | None = None,
) -> str:
    ts = generated_at or datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    sections = [
        _section_title(ts),
        _section_sharpe_warning(),
        _section_summary(rows),
        _section_best_candidates(rows),
        _section_no_robust_warning(rows),
        _section_leaderboard_table(rows),
        _section_robustness_formula(),
        _section_search_space(manifest),
        _section_failed_weak_candidates(rows),
        _section_plots(plots_dir),
        _section_governance(),
        _section_footer(ts),
    ]
    return "\n\n".join(s for s in sections if s)


def _section_title(ts: str) -> str:
    return "\n".join([
        _header("Phase 6A: Governed Strategy Search — Robustness Leaderboard"),
        "",
        f"Generated: {ts}  ",
        "Phase: 6A — Controlled sweep over models × feature sets × transaction costs.  ",
        "Universe: canonical_ml_multi_asset (unchanged).  ",
        "Governance: no auto-approval, no auto-execution, all results logged.",
    ])


def _section_sharpe_warning() -> str:
    return "\n".join([
        _header("Important: Do Not Optimise Sharpe Alone", 2),
        "",
        _SHARPE_WARNING,
    ])


def _section_summary(rows: list[LeaderboardRow]) -> str:
    completed = [r for r in rows if r.status == "completed"]
    not_run = [r for r in rows if r.status == "not_run"]
    failed = [r for r in rows if r.status == "failed"]

    best_rob = next((r for r in rows if r.best_robustness), None)
    best_sh = next((r for r in rows if r.best_sharpe), None)

    lines = [
        _header("Summary", 2),
        "",
        f"- Total candidates: **{len(rows)}** (45 generated + 1 baseline)",
        f"- Completed: **{len(completed)}**",
        f"- Not yet run: **{len(not_run)}**",
        f"- Failed: **{len(failed)}**",
    ]
    if best_rob:
        lines.append(
            f"- Best robustness candidate: **{best_rob.candidate_id}** "
            f"(score={_format_float(best_rob.robustness_score, 3)})"
        )
    if best_sh:
        lines.append(
            f"- Best OOS Sharpe candidate: **{best_sh.candidate_id}** "
            f"(OOS Sharpe={_format_float(best_sh.oos_sharpe, 3)})"
        )
    if best_rob and best_sh and best_rob.candidate_id != best_sh.candidate_id:
        lines.append(
            "- ⚠️ Best OOS Sharpe and best robustness are **different candidates**. "
            "See robustness analysis before making a selection."
        )
    return "\n".join(lines)


def _section_best_candidates(rows: list[LeaderboardRow]) -> str:
    best_rob = next((r for r in rows if r.best_robustness), None)
    best_sh = next((r for r in rows if r.best_sharpe), None)
    if not best_rob and not best_sh:
        return ""

    lines = [_header("Top Candidates", 2), ""]

    if best_rob:
        lines += [
            _header("Best Robustness Candidate", 3),
            "",
            f"**{best_rob.candidate_id}**  ",
            f"- Model: {best_rob.model_type}  ",
            f"- Feature set: {best_rob.feature_set_name}  ",
            f"- Transaction cost: {best_rob.cost_bps} bps  ",
            f"- OOS Sharpe: {_format_float(best_rob.oos_sharpe, 3)}  ",
            f"- Max drawdown: {_format_float(best_rob.max_drawdown_pct, 2)}%  ",
            f"- Robustness score: {_format_float(best_rob.robustness_score, 3)}  ",
            f"- Split Sharpe std: {_format_float(best_rob.split_sharpe_std, 3)}  ",
            f"- Negative splits: {best_rob.n_negative_splits}/{best_rob.n_splits}  ",
        ]
        if best_rob.failure_modes:
            lines.append(f"- Failure modes: {', '.join(best_rob.failure_modes)}  ")
        if best_rob.warnings:
            lines.append(f"- Warnings: {'; '.join(best_rob.warnings)}  ")
        lines.append("")

    if best_sh and best_sh is not best_rob:
        lines += [
            _header("Best OOS Sharpe Candidate (for reference only)", 3),
            "",
            _SHARPE_WARNING,
            "",
            f"**{best_sh.candidate_id}**  ",
            f"- Model: {best_sh.model_type}  ",
            f"- Feature set: {best_sh.feature_set_name}  ",
            f"- Transaction cost: {best_sh.cost_bps} bps  ",
            f"- OOS Sharpe: {_format_float(best_sh.oos_sharpe, 3)}  ",
            f"- Max drawdown: {_format_float(best_sh.max_drawdown_pct, 2)}%  ",
            f"- Robustness score: {_format_float(best_sh.robustness_score, 3)}  ",
        ]

    return "\n".join(lines)


def _section_no_robust_warning(rows: list[LeaderboardRow]) -> str:
    best_rob = next((r for r in rows if r.best_robustness), None)
    completed = [r for r in rows if r.status == "completed"]
    if not completed:
        return "\n".join([
            _header("No Experiments Run", 2),
            "",
            "> No candidates have been executed yet. "
            "Run `python scripts/run_strategy_search.py --execute` to run candidates, "
            "then rebuild the leaderboard with `--leaderboard`.",
        ])
    if best_rob is None or best_rob.robustness_score is None or best_rob.robustness_score < 0:
        return "\n".join([
            _header("⚠️ No Robust Candidate Found", 2),
            "",
            "> No candidate achieved a positive robustness score. "
            "Review the failure modes and split stability results before proceeding. "
            "Do not select any candidate for portfolio presentation at this stage.",
        ])
    return ""


def _section_leaderboard_table(rows: list[LeaderboardRow]) -> str:
    if not rows:
        return ""
    lines = [
        _header("Leaderboard (Ranked by Robustness Score)", 2),
        "",
        "_Ranked by Phase 6A robustness score. "
        "◆ = best OOS Sharpe, ★ = best robustness, [BASELINE] = reference experiment._",
        "",
        leaderboard_to_markdown_table(rows),
    ]
    return "\n".join(lines)


def _section_robustness_formula() -> str:
    return "\n".join([
        _header("Robustness Score Formula", 2),
        "",
        "The robustness score is a transparent composite designed to reward "
        "out-of-sample consistency and penalise instability:",
        "",
        f"```\n{FORMULA_DESCRIPTION}\n```",
        "",
        "**Components:**",
        "- `mean_oos_sharpe` — primary reward signal (40% weight)",
        "- `max_drawdown_pct` — penalises large drawdowns (25% weight)",
        "- `std_oos_sharpe` — penalises unstable OOS performance (15% weight)",
        "- `worst_split_sharpe < −0.5` — penalises catastrophic splits (10% weight)",
        "- `n_negative_splits / n_splits` — penalises split instability (10% weight)",
        "- `missing_diagnostics` — penalises incomplete artefacts (2% per field)",
    ])


def _section_search_space(manifest: dict[str, Any] | None) -> str:
    if not manifest:
        return ""
    ss = manifest.get("search_space", {})
    models = ss.get("models", [])
    fs_list = ss.get("feature_sets", [])
    costs = ss.get("cost_bps", [])

    lines = [
        _header("Search Space", 2),
        "",
        f"- **Base experiment:** `{manifest.get('base_experiment', 'canonical_ml_multi_asset')}`",
        f"- **Total candidates:** {manifest.get('n_candidates', 'n/a')} "
        f"(45 generated + 1 baseline)",
        "",
        _header("Models", 3),
    ]
    for m in models:
        lines.append(f"- `{m.get('type')}` (id: `{m.get('id')}`, params: {m.get('params')})")
    lines += ["", _header("Feature Sets", 3)]
    for fs in fs_list:
        lines.append(
            f"- **{fs.get('name')}** (`{fs.get('id')}`) — "
            f"{fs.get('n_features')} features, {fs.get('family_description')}"
        )
    lines += ["", _header("Transaction Costs", 3)]
    lines.append(f"- {', '.join(str(c) + ' bps' for c in costs)}")
    return "\n".join(lines)


def _section_failed_weak_candidates(rows: list[LeaderboardRow]) -> str:
    problem_rows = [
        r for r in rows
        if r.status != "not_run" and (
            r.failure_modes
            or r.warnings
            or (r.robustness_score is not None and r.robustness_score < 0)
        )
    ]
    if not problem_rows:
        return ""

    lines = [
        _header("Failed and Weak Candidates", 2),
        "",
        "The following candidates exhibited failure modes, warnings, "
        "or negative robustness scores. They are included for completeness "
        "and must not be hidden or discarded.",
        "",
    ]
    for r in problem_rows:
        lines.append(f"**{r.candidate_id}** (rank {r.rank}, score={_format_float(r.robustness_score, 3)})")
        if r.failure_modes:
            lines.append(f"  - Failure modes: {', '.join(r.failure_modes)}")
        if r.warnings:
            lines.append(f"  - Warnings: {'; '.join(r.warnings)}")
        lines.append("")

    return "\n".join(lines)


def _section_plots(plots_dir: Path | None) -> str:
    if plots_dir is None or not plots_dir.exists():
        return ""

    plot_files = {
        "leaderboard_bar.png": "Leaderboard Bar Chart — Robustness Score by Candidate",
        "sharpe_vs_drawdown_scatter.png": "Sharpe vs Max Drawdown Scatter",
        "split_stability.png": "Split Stability (Mean ± Std OOS Sharpe)",
        "equity_curve_top3.png": "Equity Curve Comparison — Top 3 Robustness Candidates",
        "drawdown_top3.png": "Drawdown Comparison — Top 3 Robustness Candidates",
    }
    existing = [(stem, label) for stem, label in plot_files.items()
                if (plots_dir / stem).exists()]
    if not existing:
        return ""

    lines = [_header("Plots", 2), ""]
    for stem, label in existing:
        lines.append(f"![{label}]({plots_dir / stem})")
        lines.append("")
    return "\n".join(lines)


def _section_governance() -> str:
    return "\n".join([
        _header("Governance Notes", 2),
        "",
        "- All candidate configs are generated from registry-supported fields only.",
        "- No config was auto-approved or auto-executed.",
        "- Execution requires explicit `--execute` flag and logs each run.",
        "- All artefacts are preserved in `results/experiments/` and "
        "`results/strategy_search/`.",
        "- Leaderboard is reproducible: re-run "
        "`python scripts/run_strategy_search.py --leaderboard --report` "
        "to regenerate from artefacts.",
    ])


def _section_footer(ts: str) -> str:
    return f"---\n_Report generated at {ts}. Phase 6A: Governed Strategy Search._"


# ---------------------------------------------------------------------------
# Website summary
# ---------------------------------------------------------------------------


def build_website_summary(
    rows: list[LeaderboardRow],
    manifest: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> str:
    """Concise website-ready summary (< 400 words)."""
    ts = generated_at or datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    completed = [r for r in rows if r.status == "completed"]
    not_run = [r for r in rows if r.status == "not_run"]
    best_rob = next((r for r in rows if r.best_robustness), None)
    best_sh = next((r for r in rows if r.best_sharpe), None)

    n_total = len(rows)
    n_done = len(completed)
    n_pending = len(not_run)

    lines = [
        _header("Phase 6A Strategy Search — Summary"),
        "",
        _SHARPE_WARNING,
        "",
        "## Results",
        "",
        f"Controlled sweep over {n_total} candidates "
        "(3 models × 5 feature sets × 3 transaction costs + baseline).  ",
        f"Completed: **{n_done}** | Pending: **{n_pending}**  ",
        "",
    ]

    if not completed:
        lines += [
            "No experiments have been executed yet.  ",
            "Run `python scripts/run_strategy_search.py --execute` to proceed.",
        ]
    else:
        if best_rob:
            lines += [
                "### Best Robustness Candidate",
                "",
                f"**{best_rob.candidate_id}**  ",
                f"Model: {best_rob.model_type} | "
                f"Features: {best_rob.feature_set_name} | "
                f"Cost: {best_rob.cost_bps} bps  ",
                f"OOS Sharpe: {_format_float(best_rob.oos_sharpe, 3)} | "
                f"Max Drawdown: {_format_float(best_rob.max_drawdown_pct, 2)}% | "
                f"Robustness Score: {_format_float(best_rob.robustness_score, 3)}  ",
                "",
            ]
        if best_sh and best_sh is not best_rob:
            lines += [
                "### Best OOS Sharpe (reference only — not recommended for selection)",
                "",
                f"**{best_sh.candidate_id}** — "
                f"OOS Sharpe: {_format_float(best_sh.oos_sharpe, 3)} | "
                f"Robustness Score: {_format_float(best_sh.robustness_score, 3)}  ",
                "",
            ]
        elif best_rob and best_sh and best_rob.candidate_id == best_sh.candidate_id:
            lines.append(
                "Best OOS Sharpe and best robustness are the **same candidate** — "
                "this provides stronger evidence for selection.  "
            )
            lines.append("")

    lines += [
        "## Scoring",
        "",
        "Robustness score rewards OOS Sharpe and penalises drawdown, "
        "split instability, catastrophic splits, and missing diagnostics.  ",
        "Best Sharpe is reported separately as a reference point only.",
        "",
        f"_Generated {ts}_",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------


def write_full_report(
    rows: list[LeaderboardRow],
    output_dir: Path,
    manifest: dict[str, Any] | None = None,
    plots_dir: Path | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "search_report.md"
    content = build_full_report(rows, manifest=manifest, plots_dir=plots_dir)
    path.write_text(content, encoding="utf-8")
    return path


def write_website_summary(
    rows: list[LeaderboardRow],
    output_dir: Path,
    manifest: dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "website_summary.md"
    content = build_website_summary(rows, manifest=manifest)
    path.write_text(content, encoding="utf-8")
    return path
