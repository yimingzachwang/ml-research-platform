"""Tests for Phase 6A robustness_score.py.

Covers:
  A. Nominal score computation
  B. Each component correctly contributes to total
  C. Drawdown penalty (max_drawdown_pct is negative)
  D. Split instability penalty
  E. Catastrophic split penalty (worst_split_sharpe < -0.5)
  F. Negative split proportion penalty
  G. Missing diagnostics penalty
  H. None inputs handled correctly
  I. Component breakdown sums to total
  J. Score is deterministic
  K. parse_pct_input helper
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.research_search.robustness_score import (
    CATASTROPHIC_THRESHOLD,
    FORMULA_DESCRIPTION,
    W_CATASTRO,
    W_DRAWDOWN,
    W_MISSING,
    W_NEG_SPLITS,
    W_OOS_SHARPE,
    W_SPLIT_STD,
    WEIGHTS,
    RobustnessComponents,
    compute_robustness_score,
    parse_pct_input,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score(**kwargs) -> RobustnessComponents | None:
    return compute_robustness_score(**kwargs)


# ---------------------------------------------------------------------------
# A — Nominal computation
# ---------------------------------------------------------------------------


class TestNominalComputation:
    def test_returns_robustness_components(self):
        r = _score(oos_sharpe=1.0, max_drawdown_pct=-20.0)
        assert isinstance(r, RobustnessComponents)

    def test_total_is_float(self):
        r = _score(oos_sharpe=1.0)
        assert isinstance(r.total, float)

    def test_positive_oos_sharpe_gives_positive_contribution(self):
        r = _score(oos_sharpe=1.0)
        assert r.oos_sharpe_contribution > 0

    def test_zero_oos_sharpe_gives_zero_contribution(self):
        r = _score(oos_sharpe=0.0)
        assert r.oos_sharpe_contribution == 0.0

    def test_formula_description_is_string(self):
        assert isinstance(FORMULA_DESCRIPTION, str)
        assert "mean_oos_sharpe" in FORMULA_DESCRIPTION

    def test_weights_are_positive(self):
        for k, w in WEIGHTS.items():
            assert w > 0, f"Weight {k} must be positive"


# ---------------------------------------------------------------------------
# B — Component contributions
# ---------------------------------------------------------------------------


class TestComponentContributions:
    def test_oos_sharpe_weight(self):
        r = _score(oos_sharpe=1.0)
        assert abs(r.oos_sharpe_contribution - W_OOS_SHARPE * 1.0) < 1e-6

    def test_drawdown_weight(self):
        r = _score(oos_sharpe=None, max_drawdown_pct=-50.0)
        expected = W_DRAWDOWN * (-50.0 / 100.0)
        assert abs(r.drawdown_contribution - expected) < 1e-6

    def test_split_std_weight(self):
        r = _score(oos_sharpe=None, std_oos_sharpe=1.0)
        assert abs(r.split_instability_penalty - W_SPLIT_STD * 1.0) < 1e-6


# ---------------------------------------------------------------------------
# C — Drawdown penalty
# ---------------------------------------------------------------------------


class TestDrawdownPenalty:
    def test_negative_drawdown_reduces_score(self):
        r_no_dd = _score(oos_sharpe=1.0)
        r_with_dd = _score(oos_sharpe=1.0, max_drawdown_pct=-30.0)
        assert r_with_dd.total < r_no_dd.total

    def test_larger_drawdown_reduces_more(self):
        r_small = _score(oos_sharpe=1.0, max_drawdown_pct=-10.0)
        r_large = _score(oos_sharpe=1.0, max_drawdown_pct=-50.0)
        assert r_large.total < r_small.total

    def test_drawdown_contribution_is_non_positive(self):
        r = _score(oos_sharpe=1.0, max_drawdown_pct=-25.0)
        assert r.drawdown_contribution <= 0.0

    def test_drawdown_pct_over_100_clamped(self):
        r1 = _score(oos_sharpe=1.0, max_drawdown_pct=-100.0)
        r2 = _score(oos_sharpe=1.0, max_drawdown_pct=-200.0)
        assert abs(r1.drawdown_contribution - r2.drawdown_contribution) < 1e-6

    def test_drawdown_pct_string_parsed(self):
        r1 = _score(max_drawdown_pct=-25.0)
        pct = parse_pct_input("-25.0%")
        r2 = _score(max_drawdown_pct=pct)
        assert abs(r1.total - r2.total) < 1e-6


# ---------------------------------------------------------------------------
# D — Split instability penalty
# ---------------------------------------------------------------------------


class TestSplitInstabilityPenalty:
    def test_split_std_increases_penalty(self):
        r_stable = _score(oos_sharpe=1.0, std_oos_sharpe=0.2)
        r_unstable = _score(oos_sharpe=1.0, std_oos_sharpe=1.5)
        assert r_unstable.total < r_stable.total

    def test_zero_std_no_penalty(self):
        r = _score(oos_sharpe=1.0, std_oos_sharpe=0.0)
        assert r.split_instability_penalty == 0.0

    def test_high_std_clamped_at_3(self):
        r1 = _score(oos_sharpe=1.0, std_oos_sharpe=3.0)
        r2 = _score(oos_sharpe=1.0, std_oos_sharpe=10.0)
        assert abs(r1.split_instability_penalty - r2.split_instability_penalty) < 1e-6

    def test_split_instability_penalty_non_negative(self):
        r = _score(oos_sharpe=1.0, std_oos_sharpe=1.0)
        assert r.split_instability_penalty >= 0.0


# ---------------------------------------------------------------------------
# E — Catastrophic split penalty
# ---------------------------------------------------------------------------


class TestCatastrophicSplitPenalty:
    def test_catastrophic_split_fires_below_threshold(self):
        r_bad = _score(oos_sharpe=1.0, worst_split_sharpe=CATASTROPHIC_THRESHOLD - 0.1)
        assert r_bad.catastrophic_split_penalty == W_CATASTRO

    def test_no_catastrophic_penalty_at_threshold(self):
        r_ok = _score(oos_sharpe=1.0, worst_split_sharpe=CATASTROPHIC_THRESHOLD)
        assert r_ok.catastrophic_split_penalty == 0.0

    def test_no_catastrophic_penalty_above_threshold(self):
        r_ok = _score(oos_sharpe=1.0, worst_split_sharpe=0.5)
        assert r_ok.catastrophic_split_penalty == 0.0

    def test_catastrophic_reduces_total(self):
        r_ok = _score(oos_sharpe=1.0, worst_split_sharpe=0.5)
        r_bad = _score(oos_sharpe=1.0, worst_split_sharpe=-1.0)
        assert r_bad.total < r_ok.total

    def test_threshold_is_minus_half(self):
        assert CATASTROPHIC_THRESHOLD == -0.5


# ---------------------------------------------------------------------------
# F — Negative split proportion penalty
# ---------------------------------------------------------------------------


class TestNegativeSplitPenalty:
    def test_all_positive_splits_no_penalty(self):
        r = _score(oos_sharpe=1.0, n_negative_splits=0, n_splits=8)
        assert r.negative_split_penalty == 0.0

    def test_all_negative_splits_max_penalty(self):
        r = _score(oos_sharpe=1.0, n_negative_splits=8, n_splits=8)
        assert abs(r.negative_split_penalty - W_NEG_SPLITS) < 1e-6

    def test_half_negative_half_penalty(self):
        r = _score(oos_sharpe=1.0, n_negative_splits=4, n_splits=8)
        assert abs(r.negative_split_penalty - W_NEG_SPLITS * 0.5) < 1e-6

    def test_penalty_is_non_negative(self):
        r = _score(oos_sharpe=1.0, n_negative_splits=3, n_splits=8)
        assert r.negative_split_penalty >= 0.0

    def test_zero_splits_no_penalty(self):
        r = _score(oos_sharpe=1.0, n_negative_splits=0, n_splits=0)
        assert r.negative_split_penalty == 0.0


# ---------------------------------------------------------------------------
# G — Missing diagnostics penalty
# ---------------------------------------------------------------------------


class TestMissingDiagnosticsPenalty:
    def test_zero_missing_no_penalty(self):
        r = _score(oos_sharpe=1.0, missing_diagnostics=0)
        assert r.missing_diagnostics_penalty == 0.0

    def test_one_missing_applies_penalty(self):
        r = _score(oos_sharpe=1.0, missing_diagnostics=1)
        assert abs(r.missing_diagnostics_penalty - W_MISSING) < 1e-6

    def test_more_missing_larger_penalty(self):
        r1 = _score(oos_sharpe=1.0, missing_diagnostics=1)
        r3 = _score(oos_sharpe=1.0, missing_diagnostics=3)
        assert r3.missing_diagnostics_penalty > r1.missing_diagnostics_penalty

    def test_penalty_reduces_total(self):
        r_clean = _score(oos_sharpe=1.0, missing_diagnostics=0)
        r_missing = _score(oos_sharpe=1.0, missing_diagnostics=2)
        assert r_missing.total < r_clean.total


# ---------------------------------------------------------------------------
# H — None inputs
# ---------------------------------------------------------------------------


class TestNoneInputs:
    def test_all_none_returns_none(self):
        r = _score(
            oos_sharpe=None, max_drawdown_pct=None, std_oos_sharpe=None
        )
        assert r is None

    def test_only_oos_sharpe_works(self):
        r = _score(oos_sharpe=1.0)
        assert r is not None
        assert r.total > 0

    def test_only_drawdown_works(self):
        r = _score(max_drawdown_pct=-30.0)
        assert r is not None

    def test_missing_oos_sharpe_zero_contribution(self):
        r = _score(oos_sharpe=None, max_drawdown_pct=-20.0)
        assert r.oos_sharpe_contribution == 0.0

    def test_missing_drawdown_zero_contribution(self):
        r = _score(oos_sharpe=1.0, max_drawdown_pct=None)
        assert r.drawdown_contribution == 0.0

    def test_missing_std_zero_penalty(self):
        r = _score(oos_sharpe=1.0, std_oos_sharpe=None)
        assert r.split_instability_penalty == 0.0

    def test_missing_worst_split_no_catastrophic(self):
        r = _score(oos_sharpe=1.0, worst_split_sharpe=None)
        assert r.catastrophic_split_penalty == 0.0


# ---------------------------------------------------------------------------
# I — Component breakdown integrity
# ---------------------------------------------------------------------------


class TestComponentBreakdown:
    def test_components_sum_to_total(self):
        r = _score(
            oos_sharpe=1.2,
            max_drawdown_pct=-25.0,
            std_oos_sharpe=0.6,
            worst_split_sharpe=-0.8,
            n_negative_splits=3,
            n_splits=8,
            missing_diagnostics=1,
        )
        expected = (
            r.oos_sharpe_contribution
            + r.drawdown_contribution
            - r.split_instability_penalty
            - r.catastrophic_split_penalty
            - r.negative_split_penalty
            - r.missing_diagnostics_penalty
        )
        assert abs(r.total - round(expected, 4)) < 1e-3

    def test_to_dict_has_all_keys(self):
        r = _score(oos_sharpe=1.0)
        d = r.to_dict()
        for key in ("total", "oos_sharpe_contribution", "drawdown_contribution",
                    "split_instability_penalty", "catastrophic_split_penalty",
                    "negative_split_penalty", "missing_diagnostics_penalty",
                    "inputs_used", "weights", "formula"):
            assert key in d, f"to_dict missing {key!r}"

    def test_inputs_used_recorded(self):
        r = _score(oos_sharpe=1.5, max_drawdown_pct=-20.0, n_splits=6)
        assert r.inputs_used["oos_sharpe"] == 1.5
        assert r.inputs_used["n_splits"] == 6


# ---------------------------------------------------------------------------
# J — Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_same_output(self):
        kwargs = dict(
            oos_sharpe=1.2,
            max_drawdown_pct=-25.0,
            std_oos_sharpe=0.5,
            worst_split_sharpe=-0.3,
            n_negative_splits=2,
            n_splits=8,
        )
        r1 = _score(**kwargs)
        r2 = _score(**kwargs)
        assert r1.total == r2.total

    def test_oos_sharpe_clamped_at_3(self):
        r_high = _score(oos_sharpe=10.0)
        r_three = _score(oos_sharpe=3.0)
        assert abs(r_high.oos_sharpe_contribution - r_three.oos_sharpe_contribution) < 1e-6

    def test_oos_sharpe_clamped_at_minus_3(self):
        r_low = _score(oos_sharpe=-10.0)
        r_minus3 = _score(oos_sharpe=-3.0)
        assert abs(r_low.oos_sharpe_contribution - r_minus3.oos_sharpe_contribution) < 1e-6


# ---------------------------------------------------------------------------
# K — parse_pct_input
# ---------------------------------------------------------------------------


class TestParsePctInput:
    def test_float_passthrough(self):
        assert parse_pct_input(-25.0) == -25.0

    def test_int_converted(self):
        assert parse_pct_input(-25) == -25.0

    def test_string_pct(self):
        assert parse_pct_input("-25.0%") == -25.0

    def test_positive_string(self):
        assert parse_pct_input("12.5%") == 12.5

    def test_none_returns_none(self):
        assert parse_pct_input(None) is None

    def test_invalid_string_returns_none(self):
        assert parse_pct_input("not_a_number") is None

    def test_string_without_pct_returns_none(self):
        assert parse_pct_input("25") is None
