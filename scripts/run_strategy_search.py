#!/usr/bin/env python
"""Phase 6A: Governed Strategy Search — end-to-end runner.

Usage
-----
# Step 1 — Generate candidate configs + manifest (safe: no execution)
python scripts/run_strategy_search.py --generate-only

# Step 2 — Execute all candidates (requires Step 1; runs run_from_config.py for each)
python scripts/run_strategy_search.py --execute

# Step 3 — Build leaderboard from completed results
python scripts/run_strategy_search.py --leaderboard

# Step 4 — Generate report + plots
python scripts/run_strategy_search.py --report

# Full pipeline (generate → execute → leaderboard → report)
python scripts/run_strategy_search.py --all

# Dry-run (print what would happen, no writes, no execution)
python scripts/run_strategy_search.py --all --dry-run

Governance
----------
- Candidates are generated from registry-supported fields only.
- Execution is explicit (--execute flag required).
- Each execution call is logged to results/strategy_search/execution_log.jsonl.
- No candidate is auto-approved or auto-selected.
- All results are preserved (failed experiments are never hidden).
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.research_search.candidate_generator import (
    generate_all_configs,
)
from src.research_search.leaderboard import (
    build_leaderboard,
    leaderboard_to_dict,
    write_leaderboard_csv,
    write_leaderboard_json,
)
from src.research_search.plots import generate_all_plots
from src.research_search.report import write_full_report, write_website_summary
from src.research_search.search_space import SearchSpace

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CONFIGS_DIR = _REPO_ROOT / "configs" / "experiments" / "search"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "results" / "strategy_search"
_DEFAULT_RESULTS_BASE = _REPO_ROOT / "results" / "experiments"
_RUN_SCRIPT = _REPO_ROOT / "scripts" / "run_from_config.py"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 6A: Governed Strategy Search and Robustness Leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--generate-only", action="store_true",
                   help="Write candidate configs + manifest (no execution).")
    p.add_argument("--execute", action="store_true",
                   help="Run each candidate config sequentially via run_from_config.py.")
    p.add_argument("--leaderboard", action="store_true",
                   help="Build leaderboard from completed results.")
    p.add_argument("--report", action="store_true",
                   help="Generate markdown report + website summary + plots.")
    p.add_argument("--all", action="store_true",
                   help="Run all steps: generate → execute → leaderboard → report.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned actions without writing or executing.")
    p.add_argument("--configs-dir", type=Path, default=_DEFAULT_CONFIGS_DIR,
                   help=f"Output dir for generated YAML configs. Default: {_DEFAULT_CONFIGS_DIR}")
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR,
                   help=f"Output dir for leaderboard/report. Default: {_DEFAULT_OUTPUT_DIR}")
    p.add_argument("--results-base", type=Path, default=_DEFAULT_RESULTS_BASE,
                   help=f"Root of results/experiments/. Default: {_DEFAULT_RESULTS_BASE}")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print progress messages.")
    args = p.parse_args(argv)

    # --all expands to all steps
    if args.all:
        args.generate_only = True
        args.execute = True
        args.leaderboard = True
        args.report = True

    # Default to --generate-only if nothing specified
    if not any([args.generate_only, args.execute, args.leaderboard, args.report]):
        args.generate_only = True

    return args


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def step_generate(args: argparse.Namespace) -> dict:
    """Write candidate configs + manifest to disk."""
    _log(args, "=== Step 1: Generating candidate configs ===")
    _log(args, f"  configs_dir : {args.configs_dir}")
    _log(args, f"  output_dir  : {args.output_dir}")

    if args.dry_run:
        candidates = SearchSpace.all_candidates_with_baseline()
        _log(args, f"  [DRY RUN] Would generate {len(candidates)} candidates:")
        for c in candidates[:5]:
            _log(args, f"    {c.candidate_id}")
        if len(candidates) > 5:
            _log(args, f"    ... and {len(candidates) - 5} more")
        return {"dry_run": True, "n_candidates": len(candidates)}

    manifest, config_paths = generate_all_configs(
        configs_dir=args.configs_dir,
        output_dir=args.output_dir,
    )
    n = manifest["n_candidates"]
    _log(args, f"  Generated {n} configs in {args.configs_dir}")
    _log(args, f"  Manifest written to {args.output_dir / 'manifest.json'}")
    return manifest


def step_execute(args: argparse.Namespace, manifest: dict | None = None) -> None:
    """Execute each candidate config via run_from_config.py."""
    _log(args, "=== Step 2: Executing candidates ===")
    _log(args, "  Governance: execution is explicit and logged.")
    _log(args, "  Each run is independent; failures are preserved, not hidden.")

    if not _RUN_SCRIPT.exists():
        _abort(f"run_from_config.py not found at {_RUN_SCRIPT}")

    # Load manifest to get candidate list + config paths
    manifest_path = args.output_dir / "manifest.json"
    if manifest is None:
        if not manifest_path.exists():
            _abort(
                f"Manifest not found at {manifest_path}. "
                "Run --generate-only first."
            )
        with manifest_path.open() as fh:
            manifest = json.load(fh)

    candidates = manifest.get("candidates", [])
    log_path = args.output_dir / "execution_log.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _log(args, f"  Running {len(candidates)} candidates (logging to {log_path})")

    successes, failures = 0, 0
    for entry in candidates:
        cid = entry["candidate_id"]
        cfg_path_str = entry.get("config_path")

        if entry.get("is_baseline"):
            _log(args, f"  [SKIP] {cid} (baseline — use existing results)")
            _write_log(log_path, cid, "skipped", "baseline candidate — not re-executed")
            continue

        if cfg_path_str is None:
            _log(args, f"  [WARN] {cid}: no config_path in manifest, skipping.")
            _write_log(log_path, cid, "skipped", "no config_path in manifest")
            continue

        cfg_path = Path(cfg_path_str)
        if not cfg_path.exists():
            _log(args, f"  [WARN] {cid}: config not found at {cfg_path}, skipping.")
            _write_log(log_path, cid, "skipped", f"config file not found: {cfg_path}")
            continue

        if args.dry_run:
            _log(args, f"  [DRY RUN] Would execute: {cfg_path}")
            continue

        _log(args, f"  Running: {cid}")
        ok, msg = _run_single(cfg_path)
        status = "success" if ok else "failed"
        _write_log(log_path, cid, status, msg)
        if ok:
            successes += 1
            _log(args, f"    ✓ {cid}")
        else:
            failures += 1
            _log(args, f"    ✗ {cid}: {msg[:120]}")

    if not args.dry_run:
        _log(args, f"  Execution complete: {successes} succeeded, {failures} failed.")
        if failures > 0:
            _log(args, "  Failed experiments are preserved in the leaderboard as 'not_run'.")


def _run_single(config_path: Path) -> tuple[bool, str]:
    """Run a single config via run_from_config.py. Returns (ok, message)."""
    try:
        result = subprocess.run(
            [sys.executable, str(_RUN_SCRIPT), str(config_path)],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode == 0:
            return True, "ok"
        err = (result.stderr or result.stdout or "")[:500]
        return False, f"exit {result.returncode}: {err}"
    except subprocess.TimeoutExpired:
        return False, "execution timed out after 3600s"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _write_log(log_path: Path, cid: str, status: str, msg: str) -> None:
    record = {
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "candidate_id": cid,
        "status": status,
        "message": msg,
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def step_leaderboard(args: argparse.Namespace) -> list:
    """Build leaderboard from completed results."""
    _log(args, "=== Step 3: Building leaderboard ===")

    if args.dry_run:
        _log(args, "  [DRY RUN] Would read results from:", args.results_base)
        return []

    rows = build_leaderboard(results_base=args.results_base)
    completed = sum(1 for r in rows if r.status == "completed")
    not_run = sum(1 for r in rows if r.status == "not_run")

    _log(args, f"  {len(rows)} candidates: {completed} completed, {not_run} not yet run.")

    best_rob = next((r for r in rows if r.best_robustness), None)
    best_sh = next((r for r in rows if r.best_sharpe), None)

    if best_rob:
        _log(args,
             f"  Best robustness: {best_rob.candidate_id} "
             f"(score={best_rob.robustness_score:.4f})")
    if best_sh:
        _log(args,
             f"  Best OOS Sharpe: {best_sh.candidate_id} "
             f"(sharpe={best_sh.oos_sharpe:.4f}) [reference only]")

    lb_json = write_leaderboard_json(rows, args.output_dir)
    lb_csv = write_leaderboard_csv(rows, args.output_dir)
    _log(args, f"  Leaderboard JSON → {lb_json}")
    _log(args, f"  Leaderboard CSV  → {lb_csv}")
    return rows


def step_report(args: argparse.Namespace, rows: list | None = None) -> None:
    """Generate report, website summary, and plots."""
    _log(args, "=== Step 4: Generating report + plots ===")

    if args.dry_run:
        _log(args, "  [DRY RUN] Would write report to:", args.output_dir)
        return

    # Load rows from leaderboard.json if not passed in
    if not rows:
        lb_path = args.output_dir / "leaderboard.json"
        if not lb_path.exists():
            _abort(f"Leaderboard not found at {lb_path}. Run --leaderboard first.")
        with lb_path.open() as fh:
            lb_data = json.load(fh)
        # Rebuild lightweight row objects for report rendering
        rows = _rows_from_dict(lb_data)

    # Load manifest
    manifest = None
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as fh:
            manifest = json.load(fh)

    plots_dir = args.output_dir / "plots"

    # Plots
    _log(args, "  Generating plots...")
    plot_results = generate_all_plots(rows, plots_dir, results_base=args.results_base)
    for stem, path in plot_results.items():
        if path:
            _log(args, f"    {stem} → {path}")
        else:
            _log(args, f"    {stem} — skipped (data not available)")

    # Reports
    rpt_path = write_full_report(rows, args.output_dir, manifest=manifest, plots_dir=plots_dir)
    web_path = write_website_summary(rows, args.output_dir, manifest=manifest)
    _log(args, f"  Full report     → {rpt_path}")
    _log(args, f"  Website summary → {web_path}")


def _rows_from_dict(lb_data: dict) -> list:
    """Lightweight reconstruction of leaderboard rows for report rendering."""
    from src.research_search.leaderboard import LeaderboardRow
    from src.research_search.robustness_score import RobustnessComponents

    rows = []
    for d in lb_data.get("rows", []):
        comp_data = d.get("robustness_components")
        rob_comp = None
        if comp_data and isinstance(comp_data, dict):
            try:
                rob_comp = RobustnessComponents(
                    oos_sharpe_contribution=comp_data.get("oos_sharpe_contribution", 0.0),
                    drawdown_contribution=comp_data.get("drawdown_contribution", 0.0),
                    split_instability_penalty=comp_data.get("split_instability_penalty", 0.0),
                    catastrophic_split_penalty=comp_data.get("catastrophic_split_penalty", 0.0),
                    negative_split_penalty=comp_data.get("negative_split_penalty", 0.0),
                    missing_diagnostics_penalty=comp_data.get("missing_diagnostics_penalty", 0.0),
                    total=comp_data.get("total", 0.0),
                    inputs_used=comp_data.get("inputs_used", {}),
                )
            except Exception:  # noqa: BLE001
                pass

        row = LeaderboardRow(
            rank=d.get("rank", 0),
            candidate_id=d.get("candidate_id", ""),
            model_id=d.get("model_id", ""),
            model_type=d.get("model_type", ""),
            feature_set_id=d.get("feature_set_id", ""),
            feature_set_name=d.get("feature_set_name", ""),
            cost_bps=d.get("cost_bps", 0),
            base_experiment=d.get("base_experiment", ""),
            oos_sharpe=d.get("oos_sharpe"),
            annual_return_pct=d.get("annual_return_pct"),
            max_drawdown_pct=d.get("max_drawdown_pct"),
            hit_rate_pct=d.get("hit_rate_pct"),
            split_sharpe_mean=d.get("split_sharpe_mean"),
            split_sharpe_std=d.get("split_sharpe_std"),
            worst_split_sharpe=d.get("worst_split_sharpe"),
            n_negative_splits=d.get("n_negative_splits"),
            n_splits=d.get("n_splits"),
            robustness_score=d.get("robustness_score"),
            robustness_components=rob_comp,
            status=d.get("status", "not_run"),
            warnings=d.get("warnings", []),
            failure_modes=d.get("failure_modes", []),
            missing_metrics=d.get("missing_metrics", []),
            is_baseline=d.get("is_baseline", False),
            best_sharpe=d.get("best_sharpe", False),
            best_robustness=d.get("best_robustness", False),
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log(args: argparse.Namespace, *msgs: str) -> None:
    if args.verbose or True:  # always print progress
        print(*msgs)


def _abort(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    manifest = None
    rows = None

    if args.generate_only or args.all:
        manifest = step_generate(args)

    if args.execute:
        step_execute(args, manifest=manifest)

    if args.leaderboard:
        rows = step_leaderboard(args)

    if args.report:
        step_report(args, rows=rows)

    if not args.dry_run:
        print("\nDone.")
        print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
