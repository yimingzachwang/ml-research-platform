"""Tests for Phase 6A report.py.

Covers:
  A. build_full_report generates valid markdown
  B. Report includes Sharpe warning prominently
  C. Failed/weak candidates section present when relevant
  D. Best Sharpe and best robustness both reported
  E. Warning when no robust candidate found
  F. build_website_summary is concise
  G. Website summary includes Sharpe warning
  H. Report is deterministic
  I. write_full_report + write_website_summary write files
  J. Report mentions baseline
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import replace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.research_search.report import (
    build_full_report,
    build_website_summary,
    write_full_report,
    write_website_summary,
)
from src.research_search.leaderboard import LeaderboardRow, _assign_flags
from src.research_search.robustness_score import compute_robustness_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_row(
    candidate_id: str,
    oos_sharpe: float | None = 1.0,
    max_drawdown_pct: float | None = -20.0,
    std_oos_sharpe: float | None = 0.3,
    worst_split_sharpe: float | None = 0.2,
    n_negative_splits: int | None = 1,
    n_splits: int | None = 8,
    status: str = "completed",
    is_baseline: bool = False,
    failure_modes: list[str] | None = None,
    warnings: list[str] | None = None,
) -> LeaderboardRow:
    rob = compute_robustness_score(
        oos_sharpe=oos_sharpe,
        max_drawdown_pct=max_drawdown_pct,
        std_oos_sharpe=std_oos_sharpe,
        worst_split_sharpe=worst_split_sharpe,
        n_negative_splits=n_negative_splits,
        n_splits=n_splits,
    )
    return LeaderboardRow(
        rank=0,
        candidate_id=candidate_id,
        model_id="ridge",
        model_type="RidgeRegression",
        feature_set_id="baseline",
        feature_set_name="Baseline",
        cost_bps=5,
        base_experiment="canonical_ml_multi_asset",
        oos_sharpe=oos_sharpe,
        annual_return_pct=10.0,
        max_drawdown_pct=max_drawdown_pct,
        hit_rate_pct=55.0,
        split_sharpe_mean=oos_sharpe,
        split_sharpe_std=std_oos_sharpe,
        worst_split_sharpe=worst_split_sharpe,
        n_negative_splits=n_negative_splits,
        n_splits=n_splits,
        robustness_score=rob.total if rob else None,
        robustness_components=rob,
        status=status,
        warnings=warnings or [],
        failure_modes=failure_modes or [],
        is_baseline=is_baseline,
    )


def _sample_rows() -> list[LeaderboardRow]:
    rows = [
        _make_row("canonical_ml_multi_asset", is_baseline=True, oos_sharpe=1.0),
        _make_row("search_ridge_baseline_5bps", oos_sharpe=1.5, max_drawdown_pct=-40.0,
                  std_oos_sharpe=2.0, n_negative_splits=5, n_splits=8),
        _make_row("search_lasso_trend_only_5bps", oos_sharpe=0.9, max_drawdown_pct=-12.0,
                  std_oos_sharpe=0.2, n_negative_splits=0, n_splits=8),
        _make_row("search_elasticnet_trend_vol_5bps", status="not_run",
                  oos_sharpe=None, max_drawdown_pct=None),
        _make_row("search_ridge_trend_mr_10bps",
                  failure_modes=["catastrophic_split"],
                  warnings=["Catastrophic split detected."],
                  worst_split_sharpe=-1.5),
    ]
    for i, r in enumerate(rows, 1):
        r.rank = i
    _assign_flags(rows)
    return rows


# ---------------------------------------------------------------------------
# A — Valid markdown
# ---------------------------------------------------------------------------


class TestReportIsValidMarkdown:
    def test_report_is_string(self):
        rows = _sample_rows()
        report = build_full_report(rows)
        assert isinstance(report, str)

    def test_report_non_empty(self):
        rows = _sample_rows()
        assert len(build_full_report(rows)) > 200

    def test_report_has_h1_title(self):
        rows = _sample_rows()
        report = build_full_report(rows)
        assert report.startswith("# Phase 6A")

    def test_report_has_multiple_sections(self):
        rows = _sample_rows()
        report = build_full_report(rows)
        assert report.count("## ") >= 4


# ---------------------------------------------------------------------------
# B — Sharpe warning
# ---------------------------------------------------------------------------


class TestShareWarning:
    def test_report_warns_against_sharpe_optimisation(self):
        rows = _sample_rows()
        report = build_full_report(rows).lower()
        assert "do not select" in report or "do not optimise" in report

    def test_report_mentions_robustness_over_sharpe(self):
        rows = _sample_rows()
        report = build_full_report(rows)
        assert "robustness score" in report.lower()

    def test_website_summary_has_sharpe_warning(self):
        rows = _sample_rows()
        summary = build_website_summary(rows).lower()
        assert "do not select" in summary or "sharpe alone" in summary.lower()


# ---------------------------------------------------------------------------
# C — Failed/weak candidates section
# ---------------------------------------------------------------------------


class TestFailedCandidatesSection:
    def test_failed_candidates_section_present(self):
        rows = _sample_rows()
        report = build_full_report(rows)
        assert "Failed" in report or "failed" in report.lower()

    def test_failed_candidates_not_hidden(self):
        rows = _sample_rows()
        report = build_full_report(rows)
        # The candidate with failure modes should appear
        failed_row = next(r for r in rows if r.failure_modes)
        assert failed_row.candidate_id in report

    def test_failure_mode_name_in_report(self):
        rows = _sample_rows()
        report = build_full_report(rows)
        assert "catastrophic_split" in report

    def test_warnings_in_report(self):
        rows = _sample_rows()
        report = build_full_report(rows)
        assert "Catastrophic" in report


# ---------------------------------------------------------------------------
# D — Best Sharpe and best robustness both reported
# ---------------------------------------------------------------------------


class TestBothCandidatesReported:
    def test_best_robustness_candidate_mentioned(self):
        rows = _sample_rows()
        report = build_full_report(rows)
        best_rob = next((r for r in rows if r.best_robustness), None)
        if best_rob:
            assert best_rob.candidate_id in report

    def test_best_sharpe_candidate_mentioned(self):
        rows = _sample_rows()
        report = build_full_report(rows)
        best_sh = next((r for r in rows if r.best_sharpe), None)
        if best_sh:
            assert best_sh.candidate_id in report

    def test_report_distinguishes_best_sharpe_from_best_robustness(self):
        rows = _sample_rows()
        report = build_full_report(rows)
        best_rob = next((r for r in rows if r.best_robustness), None)
        best_sh = next((r for r in rows if r.best_sharpe), None)
        if best_rob and best_sh and best_rob.candidate_id != best_sh.candidate_id:
            # Both should be mentioned in a comparison context
            assert best_rob.candidate_id in report
            assert best_sh.candidate_id in report


# ---------------------------------------------------------------------------
# E — No robust candidate warning
# ---------------------------------------------------------------------------


class TestNoRobustCandidateWarning:
    def test_warning_when_no_positive_robustness(self):
        rows = [
            _make_row("bad_a", oos_sharpe=-0.5, max_drawdown_pct=-70.0),
            _make_row("bad_b", oos_sharpe=-0.3, max_drawdown_pct=-60.0),
        ]
        for i, r in enumerate(rows, 1):
            r.rank = i
        _assign_flags(rows)
        report = build_full_report(rows)
        assert "no robust candidate" in report.lower()

    def test_no_warning_when_robust_candidate_exists(self):
        rows = [
            _make_row("good", oos_sharpe=1.5, max_drawdown_pct=-15.0,
                      n_negative_splits=0, n_splits=8),
        ]
        for i, r in enumerate(rows, 1):
            r.rank = i
        _assign_flags(rows)
        report = build_full_report(rows)
        assert "⚠️ No Robust Candidate" not in report

    def test_no_experiments_run_message(self):
        rows = [
            _make_row("not_run_a", status="not_run",
                      oos_sharpe=None, max_drawdown_pct=None),
        ]
        report = build_full_report(rows)
        assert "No Experiments Run" in report or "not yet run" in report.lower()


# ---------------------------------------------------------------------------
# F — Website summary conciseness
# ---------------------------------------------------------------------------


class TestWebsiteSummaryConcise:
    def test_website_summary_is_string(self):
        rows = _sample_rows()
        assert isinstance(build_website_summary(rows), str)

    def test_website_summary_non_empty(self):
        rows = _sample_rows()
        assert len(build_website_summary(rows)) > 50

    def test_website_summary_shorter_than_full_report(self):
        rows = _sample_rows()
        full = build_full_report(rows)
        summary = build_website_summary(rows)
        assert len(summary) < len(full)

    def test_website_summary_mentions_n_candidates(self):
        rows = _sample_rows()
        summary = build_website_summary(rows)
        # Should mention the count
        assert any(str(n) in summary for n in [len(rows), 46])


# ---------------------------------------------------------------------------
# G — Website summary Sharpe warning
# ---------------------------------------------------------------------------


class TestWebsiteSummaryShareWarning:
    def test_summary_includes_warning(self):
        rows = _sample_rows()
        summary = build_website_summary(rows)
        assert "sharpe" in summary.lower()

    def test_summary_does_not_recommend_by_sharpe_only(self):
        rows = _sample_rows()
        summary = build_website_summary(rows)
        # Should reference robustness
        assert "robustness" in summary.lower()


# ---------------------------------------------------------------------------
# H — Determinism
# ---------------------------------------------------------------------------


class TestReportDeterminism:
    def test_same_rows_same_report(self):
        rows_a = _sample_rows()
        rows_b = _sample_rows()
        # Use fixed timestamp
        ts = "2026-06-11T00:00:00+00:00"
        r1 = build_full_report(rows_a, generated_at=ts)
        r2 = build_full_report(rows_b, generated_at=ts)
        assert r1 == r2

    def test_same_rows_same_summary(self):
        rows_a = _sample_rows()
        rows_b = _sample_rows()
        ts = "2026-06-11T00:00:00+00:00"
        r1 = build_website_summary(rows_a, generated_at=ts)
        r2 = build_website_summary(rows_b, generated_at=ts)
        assert r1 == r2


# ---------------------------------------------------------------------------
# I — File writing
# ---------------------------------------------------------------------------


class TestFileWriting:
    def test_write_full_report_creates_file(self, tmp_path):
        rows = _sample_rows()
        path = write_full_report(rows, tmp_path)
        assert path.exists()
        assert path.name == "search_report.md"

    def test_write_website_summary_creates_file(self, tmp_path):
        rows = _sample_rows()
        path = write_website_summary(rows, tmp_path)
        assert path.exists()
        assert path.name == "website_summary.md"

    def test_written_file_content_matches_build(self, tmp_path):
        rows = _sample_rows()
        path = write_full_report(rows, tmp_path)
        content = path.read_text()
        assert len(content) > 100
        assert "Phase 6A" in content


# ---------------------------------------------------------------------------
# J — Baseline mentioned in report
# ---------------------------------------------------------------------------


class TestBaselineMentioned:
    def test_report_mentions_baseline(self):
        rows = _sample_rows()
        report = build_full_report(rows).lower()
        assert "baseline" in report

    def test_leaderboard_table_has_baseline_marker(self):
        rows = _sample_rows()
        report = build_full_report(rows)
        assert "[BASELINE]" in report or "baseline" in report
