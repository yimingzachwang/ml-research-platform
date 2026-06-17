"""Tests for Phase 6A search_space.py.

Covers:
  A. SearchSpace.all_candidates() is deterministic
  B. Only registry-supported model types
  C. Only registry-supported feature types
  D. Feature set IDs and model IDs are unique
  E. Cost bps match spec (5, 10, 20)
  F. Baseline is always included
  G. Candidate count matches spec (3×5×3 = 45 generated, 46 total)
  H. Candidate IDs are stable and slug-formatted
  I. SearchSpace.summary() structure
  J. SearchCandidate.to_dict() contract
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.research_search.search_space import (
    COST_BPS,
    FEATURE_SETS,
    MODELS,
    N_GENERATED,
    N_TOTAL,
    SUPPORTED_FEATURE_TYPES,
    SUPPORTED_MODEL_TYPES,
    SearchSpace,
)


# ---------------------------------------------------------------------------
# A — Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_all_candidates_same_order_called_twice(self):
        ids_a = [c.candidate_id for c in SearchSpace.all_candidates()]
        ids_b = [c.candidate_id for c in SearchSpace.all_candidates()]
        assert ids_a == ids_b

    def test_all_candidates_with_baseline_same_order_twice(self):
        ids_a = [c.candidate_id for c in SearchSpace.all_candidates_with_baseline()]
        ids_b = [c.candidate_id for c in SearchSpace.all_candidates_with_baseline()]
        assert ids_a == ids_b

    def test_candidate_order_is_models_then_features_then_costs(self):
        candidates = SearchSpace.all_candidates()
        # First 15 candidates belong to first model (ridge)
        first_model_candidates = candidates[:15]
        assert all(c.model.id == "ridge" for c in first_model_candidates)


# ---------------------------------------------------------------------------
# B — Registry-supported models only
# ---------------------------------------------------------------------------


class TestRegistrySupportedModels:
    def test_all_model_types_are_supported(self):
        for c in SearchSpace.all_candidates():
            assert c.model.type in SUPPORTED_MODEL_TYPES, (
                f"Unsupported model type {c.model.type!r} in candidate {c.candidate_id}"
            )

    def test_model_params_are_dicts(self):
        for m in MODELS:
            assert isinstance(m.params, dict)

    def test_ridge_has_alpha(self):
        ridge = next(m for m in MODELS if m.id == "ridge")
        assert "alpha" in ridge.params

    def test_lasso_has_alpha(self):
        lasso = next(m for m in MODELS if m.id == "lasso")
        assert "alpha" in lasso.params

    def test_elasticnet_has_alpha_and_l1_ratio(self):
        en = next(m for m in MODELS if m.id == "elasticnet")
        assert "alpha" in en.params
        assert "l1_ratio" in en.params


# ---------------------------------------------------------------------------
# C — Registry-supported feature types only
# ---------------------------------------------------------------------------


class TestRegistrySupportedFeatureTypes:
    def test_all_feature_types_are_supported(self):
        for fs in FEATURE_SETS:
            for entry in fs.entries:
                assert entry.type in SUPPORTED_FEATURE_TYPES, (
                    f"Unsupported feature type {entry.type!r} "
                    f"in feature set {fs.id!r}"
                )

    def test_baseline_feature_set_has_most_features(self):
        baseline = next(fs for fs in FEATURE_SETS if fs.id == "baseline")
        for fs in FEATURE_SETS:
            assert len(fs.entries) <= len(baseline.entries)

    def test_trend_only_contains_only_trend_family(self):
        trend_types = {
            "momentum", "trend_strength", "trend_persistence",
            "breakout_strength", "risk_adjusted_momentum",
        }
        trend_only = next(fs for fs in FEATURE_SETS if fs.id == "trend_only")
        for e in trend_only.entries:
            assert e.type in trend_types, (
                f"trend_only contains non-trend type {e.type!r}"
            )

    def test_trend_vol_contains_volatility_types(self):
        vol_types = {"rolling_volatility", "vol_compression"}
        trend_vol = next(fs for fs in FEATURE_SETS if fs.id == "trend_vol")
        types_present = {e.type for e in trend_vol.entries}
        assert vol_types & types_present, "trend_vol must include volatility features"

    def test_trend_mr_contains_mean_reversion_types(self):
        mr_types = {"rolling_zscore", "drawdown_distance"}
        trend_mr = next(fs for fs in FEATURE_SETS if fs.id == "trend_mr")
        types_present = {e.type for e in trend_mr.entries}
        assert mr_types & types_present, "trend_mr must include mean-reversion features"

    def test_trend_vol_mktstr_contains_market_structure(self):
        ms_types = {"rolling_beta"}
        fset = next(fs for fs in FEATURE_SETS if fs.id == "trend_vol_mktstr")
        types_present = {e.type for e in fset.entries}
        assert ms_types & types_present, "trend_vol_mktstr must include market-structure features"


# ---------------------------------------------------------------------------
# D — Uniqueness
# ---------------------------------------------------------------------------


class TestUniqueness:
    def test_model_ids_unique(self):
        ids = [m.id for m in MODELS]
        assert len(ids) == len(set(ids))

    def test_feature_set_ids_unique(self):
        ids = [fs.id for fs in FEATURE_SETS]
        assert len(ids) == len(set(ids))

    def test_candidate_ids_unique(self):
        ids = [c.candidate_id for c in SearchSpace.all_candidates()]
        assert len(ids) == len(set(ids))

    def test_all_candidates_with_baseline_ids_unique(self):
        ids = [c.candidate_id for c in SearchSpace.all_candidates_with_baseline()]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# E — Cost bps
# ---------------------------------------------------------------------------


class TestCostBps:
    def test_cost_bps_spec(self):
        assert set(COST_BPS) == {5, 10, 20}

    def test_all_candidates_have_valid_costs(self):
        for c in SearchSpace.all_candidates():
            assert c.cost_bps in COST_BPS, (
                f"Candidate {c.candidate_id} has unexpected cost {c.cost_bps}"
            )


# ---------------------------------------------------------------------------
# F — Baseline
# ---------------------------------------------------------------------------


class TestBaseline:
    def test_baseline_is_always_first(self):
        candidates = SearchSpace.all_candidates_with_baseline()
        assert candidates[0].is_baseline is True

    def test_baseline_candidate_id(self):
        b = SearchSpace.baseline_candidate()
        assert b.candidate_id == "canonical_ml_multi_asset"

    def test_baseline_is_not_in_all_candidates(self):
        ids = {c.candidate_id for c in SearchSpace.all_candidates()}
        assert "canonical_ml_multi_asset" not in ids

    def test_only_one_baseline(self):
        candidates = SearchSpace.all_candidates_with_baseline()
        baselines = [c for c in candidates if c.is_baseline]
        assert len(baselines) == 1


# ---------------------------------------------------------------------------
# G — Counts
# ---------------------------------------------------------------------------


class TestCounts:
    def test_n_generated_is_45(self):
        assert N_GENERATED == 45
        assert len(SearchSpace.all_candidates()) == 45

    def test_n_total_is_46(self):
        assert N_TOTAL == 46
        assert len(SearchSpace.all_candidates_with_baseline()) == 46

    def test_3_models(self):
        assert len(MODELS) == 3

    def test_5_feature_sets(self):
        assert len(FEATURE_SETS) == 5

    def test_3_cost_levels(self):
        assert len(COST_BPS) == 3


# ---------------------------------------------------------------------------
# H — Candidate ID format
# ---------------------------------------------------------------------------


class TestCandidateIdFormat:
    def test_generated_ids_start_with_search(self):
        for c in SearchSpace.all_candidates():
            assert c.candidate_id.startswith("search_"), (
                f"{c.candidate_id} does not start with 'search_'"
            )

    def test_ids_contain_model_id(self):
        for c in SearchSpace.all_candidates():
            assert c.model.id in c.candidate_id

    def test_ids_contain_feature_set_id(self):
        for c in SearchSpace.all_candidates():
            assert c.feature_set.id in c.candidate_id

    def test_ids_contain_bps_suffix(self):
        for c in SearchSpace.all_candidates():
            assert str(c.cost_bps) + "bps" in c.candidate_id

    def test_ids_have_no_spaces(self):
        for c in SearchSpace.all_candidates_with_baseline():
            assert " " not in c.candidate_id

    def test_candidate_by_id_round_trip(self):
        for c in SearchSpace.all_candidates():
            found = SearchSpace.candidate_by_id(c.candidate_id)
            assert found is not None
            assert found.candidate_id == c.candidate_id

    def test_candidate_by_id_missing_returns_none(self):
        assert SearchSpace.candidate_by_id("nonexistent_xyz") is None


# ---------------------------------------------------------------------------
# I — Summary structure
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_has_required_keys(self):
        s = SearchSpace.summary()
        for key in ("models", "feature_sets", "cost_bps", "base_experiment",
                    "n_generated", "n_total_with_baseline"):
            assert key in s, f"summary missing key {key!r}"

    def test_summary_n_generated(self):
        assert SearchSpace.summary()["n_generated"] == 45

    def test_summary_n_total(self):
        assert SearchSpace.summary()["n_total_with_baseline"] == 46


# ---------------------------------------------------------------------------
# J — to_dict contract
# ---------------------------------------------------------------------------


class TestCandidateToDict:
    def test_to_dict_has_required_keys(self):
        c = SearchSpace.all_candidates()[0]
        d = c.to_dict()
        for key in ("candidate_id", "model_id", "model_type", "feature_set_id",
                    "feature_set_name", "cost_bps", "base_experiment", "is_baseline"):
            assert key in d, f"to_dict missing key {key!r}"

    def test_to_dict_is_baseline_false_for_generated(self):
        c = SearchSpace.all_candidates()[0]
        assert c.to_dict()["is_baseline"] is False

    def test_baseline_to_dict_is_baseline_true(self):
        b = SearchSpace.baseline_candidate()
        assert b.to_dict()["is_baseline"] is True
