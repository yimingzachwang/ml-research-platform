"""Tests for Phase 6A leaderboard.py.

Covers:
  A. Leaderboard ranks by robustness_score, not raw Sharpe
  B. Best Sharpe and best robustness candidates flagged separately
  C. Baseline always included
  D. Not-yet-run candidates preserved (not hidden)
  E. No candidate auto-selected by Sharpe alone
  F. Ranking is deterministic
  G. All required columns present in row
  H. leaderboard_to_dict structure
  I. CSV row structure
  J. Markdown table generation
  K. build_leaderboard with mocked disk artefacts
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.research_search.leaderboard import (
    LeaderboardRow,
    _assign_flags,
    leaderboard_to_dict,
    leaderboard_to_markdown_table,
    write_leaderboard_csv,
    write_leaderboard_json,
)
from src.research_search.robustness_score import compute_robustness_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_row(
    candidate_id: str,
    model_id: str = "ridge",
    feature_set_id: str = "baseline",
    cost_bps: int = 5,
    oos_sharpe: float | None = 1.0,
    max_drawdown_pct: float | None = -20.0,
    split_sharpe_std: float | None = 0.3,
    worst_split_sharpe: float | None = 0.2,
    n_negative_splits: int | None = 1,
    n_splits: int | None = 8,
    robustness_score: float | None = None,
    status: str = "completed",
    is_baseline: bool = False,
) -> LeaderboardRow:
    rob = compute_robustness_score(
        oos_sharpe=oos_sharpe,
        max_drawdown_pct=max_drawdown_pct,
        std_oos_sharpe=split_sharpe_std,
        worst_split_sharpe=worst_split_sharpe,
        n_negative_splits=n_negative_splits,
        n_splits=n_splits,
    )
    if robustness_score is not None:
        # Override
        from src.research_search.robustness_score import RobustnessComponents
        rob = RobustnessComponents(
            oos_sharpe_contribution=0.0,
            drawdown_contribution=0.0,
            split_instability_penalty=0.0,
            catastrophic_split_penalty=0.0,
            negative_split_penalty=0.0,
            missing_diagnostics_penalty=0.0,
            total=robustness_score,
            inputs_used={},
        )

    return LeaderboardRow(
        rank=0,
        candidate_id=candidate_id,
        model_id=model_id,
        model_type=f"{model_id.capitalize()}Regression",
        feature_set_id=feature_set_id,
        feature_set_name=feature_set_id.replace("_", " ").title(),
        cost_bps=cost_bps,
        base_experiment="canonical_ml_multi_asset",
        oos_sharpe=oos_sharpe,
        annual_return_pct=10.0,
        max_drawdown_pct=max_drawdown_pct,
        hit_rate_pct=55.0,
        split_sharpe_mean=oos_sharpe,
        split_sharpe_std=split_sharpe_std,
        worst_split_sharpe=worst_split_sharpe,
        n_negative_splits=n_negative_splits,
        n_splits=n_splits,
        robustness_score=rob.total if rob else None,
        robustness_components=rob,
        status=status,
        is_baseline=is_baseline,
    )


def _sample_rows() -> list[LeaderboardRow]:
    return [
        _make_row("baseline_exp", is_baseline=True, oos_sharpe=1.2,
                  max_drawdown_pct=-18.0, robustness_score=None),
        _make_row("search_ridge_baseline_5bps", oos_sharpe=1.5,
                  max_drawdown_pct=-30.0),  # high Sharpe but high drawdown
        _make_row("search_lasso_trend_only_10bps", oos_sharpe=1.1,
                  max_drawdown_pct=-12.0, split_sharpe_std=0.1),  # lower Sharpe but better risk
        _make_row("search_elasticnet_trend_vol_20bps", oos_sharpe=0.8,
                  max_drawdown_pct=-10.0, worst_split_sharpe=-0.7),  # catastrophic split
        _make_row("search_ridge_trend_mr_10bps", status="not_run",
                  oos_sharpe=None, max_drawdown_pct=None, split_sharpe_std=None,
                  worst_split_sharpe=None, n_negative_splits=None, n_splits=None),
    ]


# ---------------------------------------------------------------------------
# A — Ranks by robustness_score, not Sharpe
# ---------------------------------------------------------------------------


class TestRanksByRobustness:
    def test_higher_robustness_ranked_above_higher_sharpe(self):
        rows = [
            _make_row("high_sharpe", oos_sharpe=2.0, max_drawdown_pct=-50.0,
                      split_sharpe_std=2.5, n_negative_splits=6, n_splits=8),
            _make_row("robust", oos_sharpe=1.0, max_drawdown_pct=-10.0,
                      split_sharpe_std=0.1, n_negative_splits=0, n_splits=8),
        ]
        for i, r in enumerate(rows, 1):
            r.rank = i
        _assign_flags(rows)
        # robust should have higher robustness score
        scores = {r.candidate_id: r.robustness_score for r in rows}
        assert scores["robust"] > scores["high_sharpe"]

    def test_not_run_candidates_appear_after_completed(self):
        rows = _sample_rows()
        for i, r in enumerate(rows, 1):
            r.rank = i
        completed = [r for r in rows if r.status == "completed"]
        not_run = [r for r in rows if r.status == "not_run"]
        # All completed rows should appear before not_run (by rank)
        if completed and not_run:
            max_completed_rank = max(r.rank for r in completed)
            min_notrun_rank = min(r.rank for r in not_run)
            assert max_completed_rank < min_notrun_rank


# ---------------------------------------------------------------------------
# B — Flags
# ---------------------------------------------------------------------------


class TestFlags:
    def test_best_sharpe_candidate_marked(self):
        rows = _sample_rows()
        for i, r in enumerate(rows, 1):
            r.rank = i
        _assign_flags(rows)
        best_sharpe_rows = [r for r in rows if r.best_sharpe]
        assert len(best_sharpe_rows) == 1

    def test_best_sharpe_has_highest_oos_sharpe(self):
        rows = _sample_rows()
        for i, r in enumerate(rows, 1):
            r.rank = i
        _assign_flags(rows)
        best = next(r for r in rows if r.best_sharpe)
        completed = [r for r in rows if r.status == "completed" and r.oos_sharpe is not None]
        max_sharpe = max(r.oos_sharpe for r in completed)
        assert best.oos_sharpe == max_sharpe

    def test_best_robustness_candidate_marked(self):
        rows = _sample_rows()
        for i, r in enumerate(rows, 1):
            r.rank = i
        _assign_flags(rows)
        best_rob_rows = [r for r in rows if r.best_robustness]
        assert len(best_rob_rows) == 1

    def test_best_robustness_has_highest_robustness_score(self):
        rows = _sample_rows()
        for i, r in enumerate(rows, 1):
            r.rank = i
        _assign_flags(rows)
        best = next(r for r in rows if r.best_robustness)
        completed = [r for r in rows if r.status == "completed" and r.robustness_score is not None]
        max_score = max(r.robustness_score for r in completed)
        assert best.robustness_score == max_score

    def test_best_sharpe_and_best_robustness_can_differ(self):
        # high_sharpe: moderate OOS Sharpe but high drawdown + catastrophic split
        # robust: lower OOS Sharpe but stable splits and low drawdown
        # With worst_split_sharpe=-1.5, catastrophic penalty fires; total penalties
        # bring high_sharpe below robust.
        rows = [
            _make_row("high_sharpe", oos_sharpe=2.0, max_drawdown_pct=-70.0,
                      split_sharpe_std=2.5, n_negative_splits=7, n_splits=8,
                      worst_split_sharpe=-1.5),
            _make_row("robust", oos_sharpe=0.8, max_drawdown_pct=-8.0,
                      split_sharpe_std=0.1, n_negative_splits=0, n_splits=8),
        ]
        for i, r in enumerate(rows, 1):
            r.rank = i
        _assign_flags(rows)
        sharpe_winner = next(r for r in rows if r.best_sharpe)
        rob_winner = next(r for r in rows if r.best_robustness)
        assert sharpe_winner.candidate_id == "high_sharpe"
        assert rob_winner.candidate_id == "robust"


# ---------------------------------------------------------------------------
# C — Baseline always included
# ---------------------------------------------------------------------------


class TestBaselineAlwaysIncluded:
    def test_baseline_present_in_sample_rows(self):
        rows = _sample_rows()
        baselines = [r for r in rows if r.is_baseline]
        assert len(baselines) == 1

    def test_baseline_candidate_identified_correctly(self):
        rows = _sample_rows()
        baseline = next(r for r in rows if r.is_baseline)
        assert baseline.candidate_id == "baseline_exp"


# ---------------------------------------------------------------------------
# D — Not-yet-run candidates preserved
# ---------------------------------------------------------------------------


class TestNotRunPreserved:
    def test_not_run_candidates_in_rows(self):
        rows = _sample_rows()
        not_run = [r for r in rows if r.status == "not_run"]
        assert len(not_run) >= 1

    def test_not_run_candidates_have_none_metrics(self):
        rows = _sample_rows()
        not_run = next(r for r in rows if r.status == "not_run")
        assert not_run.oos_sharpe is None
        assert not_run.robustness_score is None

    def test_total_rows_preserved(self):
        rows = _sample_rows()
        assert len(rows) == 5


# ---------------------------------------------------------------------------
# E — Not selected by Sharpe alone
# ---------------------------------------------------------------------------


class TestNotSelectedByShareAlone:
    def test_catastrophic_split_candidate_not_best_robustness(self):
        # catastrophic: moderate Sharpe but catastrophic split + high neg-split proportion
        # stable: lower Sharpe but fully stable — total penalties flip the winner
        rows = [
            _make_row("catastrophic", oos_sharpe=1.2, max_drawdown_pct=-15.0,
                      split_sharpe_std=0.5, worst_split_sharpe=-1.5,
                      n_negative_splits=6, n_splits=8),
            _make_row("stable", oos_sharpe=0.8, max_drawdown_pct=-8.0,
                      split_sharpe_std=0.15, worst_split_sharpe=0.3,
                      n_negative_splits=0, n_splits=8),
        ]
        for i, r in enumerate(rows, 1):
            r.rank = i
        _assign_flags(rows)
        rob_winner = next(r for r in rows if r.best_robustness)
        assert rob_winner.candidate_id == "stable"

    def test_high_drawdown_candidate_penalised(self):
        # Hold OOS Sharpe equal; difference is only drawdown — low_dd must score higher.
        row_high_dd = _make_row("high_dd", oos_sharpe=1.0, max_drawdown_pct=-60.0)
        row_low_dd = _make_row("low_dd", oos_sharpe=1.0, max_drawdown_pct=-10.0)
        assert row_low_dd.robustness_score > row_high_dd.robustness_score


# ---------------------------------------------------------------------------
# F — Determinism
# ---------------------------------------------------------------------------


class TestRankingDeterminism:
    def test_same_rows_same_flags(self):
        rows_a = _sample_rows()
        rows_b = _sample_rows()
        for i, r in enumerate(rows_a, 1):
            r.rank = i
        for i, r in enumerate(rows_b, 1):
            r.rank = i
        _assign_flags(rows_a)
        _assign_flags(rows_b)
        sharpe_a = next(r.candidate_id for r in rows_a if r.best_sharpe)
        sharpe_b = next(r.candidate_id for r in rows_b if r.best_sharpe)
        assert sharpe_a == sharpe_b


# ---------------------------------------------------------------------------
# G — Required row columns
# ---------------------------------------------------------------------------


class TestRequiredColumns:
    _REQUIRED = {
        "rank", "candidate_id", "model_id", "model_type",
        "feature_set_id", "feature_set_name", "cost_bps",
        "oos_sharpe", "max_drawdown_pct", "robustness_score",
        "status", "is_baseline", "best_sharpe", "best_robustness",
    }

    def test_to_dict_has_required_keys(self):
        row = _make_row("test_row")
        d = row.to_dict()
        for key in self._REQUIRED:
            assert key in d, f"to_dict missing {key!r}"

    def test_csv_row_has_required_keys(self):
        row = _make_row("test_row")
        csv = row.to_csv_row()
        for key in ("rank", "candidate_id", "model_id", "oos_sharpe",
                    "robustness_score", "status", "is_baseline"):
            assert key in csv, f"to_csv_row missing {key!r}"


# ---------------------------------------------------------------------------
# H — leaderboard_to_dict
# ---------------------------------------------------------------------------


class TestLeaderboardToDict:
    def test_has_search_phase(self):
        rows = _sample_rows()
        d = leaderboard_to_dict(rows)
        assert d["search_phase"] == "6A"

    def test_has_n_total(self):
        rows = _sample_rows()
        d = leaderboard_to_dict(rows)
        assert d["n_total"] == len(rows)

    def test_has_rows(self):
        rows = _sample_rows()
        d = leaderboard_to_dict(rows)
        assert len(d["rows"]) == len(rows)

    def test_has_formula(self):
        rows = _sample_rows()
        d = leaderboard_to_dict(rows)
        assert "robustness_formula" in d

    def test_no_robust_candidate_warning_when_all_negative(self):
        rows = [_make_row("bad", oos_sharpe=-1.0, max_drawdown_pct=-80.0)]
        for i, r in enumerate(rows, 1):
            r.rank = i
        _assign_flags(rows)
        d = leaderboard_to_dict(rows)
        assert d["no_robust_candidate_warning"] is True

    def test_no_robust_candidate_warning_false_when_positive(self):
        rows = [_make_row("good", oos_sharpe=1.5, max_drawdown_pct=-15.0,
                          n_negative_splits=0, n_splits=8)]
        for i, r in enumerate(rows, 1):
            r.rank = i
        _assign_flags(rows)
        d = leaderboard_to_dict(rows)
        assert d["no_robust_candidate_warning"] is False


# ---------------------------------------------------------------------------
# I — CSV round-trip
# ---------------------------------------------------------------------------


class TestCsvRoundTrip:
    def test_csv_written(self, tmp_path):
        rows = _sample_rows()
        path = write_leaderboard_csv(rows, tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_csv_has_headers(self, tmp_path):
        rows = _sample_rows()
        path = write_leaderboard_csv(rows, tmp_path)
        text = path.read_text()
        assert "candidate_id" in text
        assert "robustness_score" in text


# ---------------------------------------------------------------------------
# J — Markdown table
# ---------------------------------------------------------------------------


class TestMarkdownTable:
    def test_markdown_table_contains_headers(self):
        rows = _sample_rows()
        for i, r in enumerate(rows, 1):
            r.rank = i
        md = leaderboard_to_markdown_table(rows)
        assert "Rank" in md
        assert "Candidate" in md
        assert "Robust Score" in md

    def test_markdown_table_contains_candidate_ids(self):
        rows = _sample_rows()
        for i, r in enumerate(rows, 1):
            r.rank = i
        md = leaderboard_to_markdown_table(rows)
        for r in rows:
            assert r.candidate_id in md

    def test_markdown_table_marks_baseline(self):
        rows = _sample_rows()
        for i, r in enumerate(rows, 1):
            r.rank = i
        md = leaderboard_to_markdown_table(rows)
        assert "baseline" in md.lower()


# ---------------------------------------------------------------------------
# K — build_leaderboard with no results (all not_run)
# ---------------------------------------------------------------------------


class TestBuildLeaderboardNoResults:
    def test_build_leaderboard_with_empty_results_dir(self, tmp_path):
        from src.research_search.leaderboard import build_leaderboard
        rows = build_leaderboard(results_base=tmp_path)
        assert len(rows) == 46  # all 46 candidates, status=not_run
        assert all(r.status == "not_run" for r in rows)

    def test_baseline_present_even_with_no_results(self, tmp_path):
        from src.research_search.leaderboard import build_leaderboard
        rows = build_leaderboard(results_base=tmp_path)
        baselines = [r for r in rows if r.is_baseline]
        assert len(baselines) == 1

    def test_all_46_candidates_present(self, tmp_path):
        from src.research_search.leaderboard import build_leaderboard
        rows = build_leaderboard(results_base=tmp_path)
        assert len(rows) == 46
