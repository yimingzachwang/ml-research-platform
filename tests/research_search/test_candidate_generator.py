"""Tests for Phase 6A candidate_generator.py.

Covers:
  A. build_candidate_config produces valid structure
  B. Model overrides are applied correctly
  C. Feature overrides match the search space spec
  D. Cost overrides are applied
  E. Baseline candidate preserves base config with metadata only
  F. Metadata fields are stamped correctly
  G. Candidate IDs are stable
  H. generate_all_configs dry_run returns manifest + paths without disk I/O
  I. Only registry-supported model/feature types appear in generated configs
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.research_search.candidate_generator import (
    build_candidate_config,
    build_manifest,
    _candidate_metadata,
    _candidate_tags,
)
from src.research_search.search_space import (
    SUPPORTED_FEATURE_TYPES,
    SUPPORTED_MODEL_TYPES,
    SearchSpace,
)

# ---------------------------------------------------------------------------
# Minimal base config (mirrors canonical_ml_multi_asset structure)
# ---------------------------------------------------------------------------

_BASE_CFG: dict = {
    "version": "2",
    "name": "canonical_ml_multi_asset",
    "description": "test",
    "tags": ["ml", "ridge", "canonical"],
    "universe": {"tickers": ["SPY", "QQQ"]},
    "date_range": {"start": "2013-01-01", "end": "2024-12-31"},
    "features": {
        "ticker": "SPY",
        "entries": [
            {"name": "mom_20", "type": "momentum", "params": {"lookback": 20}},
            {"name": "vol_21", "type": "rolling_volatility", "params": {"window": 21}},
        ],
    },
    "labels": {"type": "ranking_target", "params": {"horizon": 21}},
    "model": {"type": "RidgeRegression", "params": {"alpha": 0.5}},
    "signal": {"type": "top_n", "params": {"n": 3}},
    "validation": {"type": "rolling", "parameters": {"train_months": 48, "test_months": 12}},
    "execution": {"transaction_cost_bps": 5.0},
    "output": {"base_dir": "results/experiments"},
}


# ---------------------------------------------------------------------------
# A — Valid structure
# ---------------------------------------------------------------------------


class TestCandidateConfigStructure:
    def _first_candidate(self):
        c = SearchSpace.all_candidates()[0]
        return c, build_candidate_config(c, _BASE_CFG)

    def test_config_has_name(self):
        c, cfg = self._first_candidate()
        assert cfg["name"] == c.candidate_id

    def test_config_has_model(self):
        _, cfg = self._first_candidate()
        assert "model" in cfg
        assert "type" in cfg["model"]
        assert "params" in cfg["model"]

    def test_config_has_features(self):
        _, cfg = self._first_candidate()
        assert "features" in cfg
        assert "entries" in cfg["features"]
        assert len(cfg["features"]["entries"]) > 0

    def test_config_has_execution(self):
        _, cfg = self._first_candidate()
        assert "execution" in cfg
        assert "transaction_cost_bps" in cfg["execution"]

    def test_config_has_metadata(self):
        _, cfg = self._first_candidate()
        assert "metadata" in cfg

    def test_config_has_tags(self):
        _, cfg = self._first_candidate()
        assert "tags" in cfg
        assert isinstance(cfg["tags"], list)

    def test_config_does_not_mutate_base(self):
        c = SearchSpace.all_candidates()[0]
        base_copy = copy.deepcopy(_BASE_CFG)
        build_candidate_config(c, _BASE_CFG)
        assert _BASE_CFG == base_copy


# ---------------------------------------------------------------------------
# B — Model overrides
# ---------------------------------------------------------------------------


class TestModelOverride:
    @pytest.mark.parametrize("model_id", ["ridge", "lasso", "elasticnet"])
    def test_model_type_matches_search_space(self, model_id):
        model = next(m for m in SearchSpace.models if m.id == model_id)
        fs = SearchSpace.feature_sets[0]
        from src.research_search.search_space import SearchCandidate
        c = SearchCandidate(
            candidate_id=f"search_{model_id}_baseline_5bps",
            model=model, feature_set=fs, cost_bps=5,
            base_experiment="canonical_ml_multi_asset",
        )
        cfg = build_candidate_config(c, _BASE_CFG)
        assert cfg["model"]["type"] == model.type

    def test_ridge_alpha_applied(self):
        c = next(c for c in SearchSpace.all_candidates() if c.model.id == "ridge")
        cfg = build_candidate_config(c, _BASE_CFG)
        assert cfg["model"]["params"]["alpha"] == c.model.params["alpha"]

    def test_lasso_alpha_applied(self):
        c = next(c for c in SearchSpace.all_candidates() if c.model.id == "lasso")
        cfg = build_candidate_config(c, _BASE_CFG)
        assert cfg["model"]["params"]["alpha"] == c.model.params["alpha"]

    def test_elasticnet_has_l1_ratio(self):
        c = next(c for c in SearchSpace.all_candidates() if c.model.id == "elasticnet")
        cfg = build_candidate_config(c, _BASE_CFG)
        assert "l1_ratio" in cfg["model"]["params"]


# ---------------------------------------------------------------------------
# C — Feature overrides
# ---------------------------------------------------------------------------


class TestFeatureOverride:
    def test_feature_entries_replaced(self):
        c = next(c for c in SearchSpace.all_candidates() if c.feature_set.id == "trend_only")
        cfg = build_candidate_config(c, _BASE_CFG)
        expected_names = {e.name for e in c.feature_set.entries}
        actual_names = {e["name"] for e in cfg["features"]["entries"]}
        assert expected_names == actual_names

    def test_feature_types_match_search_space(self):
        for c in SearchSpace.all_candidates():
            cfg = build_candidate_config(c, _BASE_CFG)
            for entry in cfg["features"]["entries"]:
                assert entry["type"] in SUPPORTED_FEATURE_TYPES

    def test_feature_entry_has_required_keys(self):
        c = SearchSpace.all_candidates()[0]
        cfg = build_candidate_config(c, _BASE_CFG)
        for entry in cfg["features"]["entries"]:
            assert "name" in entry
            assert "type" in entry
            assert "params" in entry

    def test_baseline_feature_set_has_most_entries(self):
        candidates = SearchSpace.all_candidates()
        baseline_c = next(c for c in candidates if c.feature_set.id == "baseline")
        baseline_cfg = build_candidate_config(baseline_c, _BASE_CFG)
        baseline_count = len(baseline_cfg["features"]["entries"])
        for c in candidates:
            cfg = build_candidate_config(c, _BASE_CFG)
            assert len(cfg["features"]["entries"]) <= baseline_count

    def test_ticker_preserved_from_base(self):
        c = SearchSpace.all_candidates()[0]
        cfg = build_candidate_config(c, _BASE_CFG)
        assert cfg["features"].get("ticker") == _BASE_CFG["features"]["ticker"]


# ---------------------------------------------------------------------------
# D — Cost overrides
# ---------------------------------------------------------------------------


class TestCostOverride:
    @pytest.mark.parametrize("cost", [5, 10, 20])
    def test_cost_applied_as_float(self, cost):
        candidates = SearchSpace.all_candidates()
        c = next(c for c in candidates if c.cost_bps == cost)
        cfg = build_candidate_config(c, _BASE_CFG)
        assert cfg["execution"]["transaction_cost_bps"] == float(cost)


# ---------------------------------------------------------------------------
# E — Baseline candidate
# ---------------------------------------------------------------------------


class TestBaselineCandidate:
    def test_baseline_config_name_updated(self):
        b = SearchSpace.baseline_candidate()
        cfg = build_candidate_config(b, _BASE_CFG)
        assert cfg["name"] == b.candidate_id

    def test_baseline_model_not_overridden(self):
        b = SearchSpace.baseline_candidate()
        cfg = build_candidate_config(b, _BASE_CFG)
        # Baseline: model should not be changed
        assert cfg["model"]["type"] == _BASE_CFG["model"]["type"]

    def test_baseline_features_not_overridden(self):
        b = SearchSpace.baseline_candidate()
        cfg = build_candidate_config(b, _BASE_CFG)
        assert cfg["features"]["entries"] == _BASE_CFG["features"]["entries"]

    def test_baseline_metadata_stamped(self):
        b = SearchSpace.baseline_candidate()
        cfg = build_candidate_config(b, _BASE_CFG)
        assert cfg["metadata"]["is_baseline"] is True
        assert cfg["metadata"]["search_phase"] == "6A"

    def test_baseline_tags_include_baseline_tag(self):
        b = SearchSpace.baseline_candidate()
        cfg = build_candidate_config(b, _BASE_CFG)
        assert "baseline" in cfg["tags"]
        assert "search_6a" in cfg["tags"]


# ---------------------------------------------------------------------------
# F — Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_metadata_candidate_id(self):
        c = SearchSpace.all_candidates()[0]
        meta = _candidate_metadata(c)
        assert meta["candidate_id"] == c.candidate_id

    def test_metadata_search_phase(self):
        c = SearchSpace.all_candidates()[0]
        meta = _candidate_metadata(c)
        assert meta["search_phase"] == "6A"

    def test_metadata_model_id(self):
        c = SearchSpace.all_candidates()[0]
        meta = _candidate_metadata(c)
        assert meta["model_id"] == c.model.id

    def test_metadata_feature_set_id(self):
        c = SearchSpace.all_candidates()[0]
        meta = _candidate_metadata(c)
        assert meta["feature_set_id"] == c.feature_set.id

    def test_metadata_cost_bps(self):
        c = SearchSpace.all_candidates()[0]
        meta = _candidate_metadata(c)
        assert meta["cost_bps"] == c.cost_bps

    def test_metadata_is_baseline_false(self):
        c = SearchSpace.all_candidates()[0]
        meta = _candidate_metadata(c)
        assert meta["is_baseline"] is False


# ---------------------------------------------------------------------------
# G — ID stability
# ---------------------------------------------------------------------------


class TestIdStability:
    def test_candidate_id_format_stable(self):
        c = SearchSpace.all_candidates()[0]
        expected = f"search_{c.model.id}_{c.feature_set.id}_{c.cost_bps}bps"
        assert c.candidate_id == expected

    def test_all_ids_deterministic_across_calls(self):
        ids_a = [c.candidate_id for c in SearchSpace.all_candidates()]
        ids_b = [c.candidate_id for c in SearchSpace.all_candidates()]
        assert ids_a == ids_b


# ---------------------------------------------------------------------------
# H — generate_all_configs dry_run
# ---------------------------------------------------------------------------


class TestGenerateAllConfigsDryRun:
    def test_dry_run_returns_manifest(self, tmp_path):
        manifest, paths = _generate_dry(tmp_path)
        assert isinstance(manifest, dict)
        assert "candidates" in manifest

    def test_dry_run_n_candidates_46(self, tmp_path):
        manifest, _ = _generate_dry(tmp_path)
        assert manifest["n_candidates"] == 46

    def test_dry_run_n_generated_45(self, tmp_path):
        manifest, _ = _generate_dry(tmp_path)
        assert manifest["n_generated"] == 45

    def test_dry_run_no_files_written(self, tmp_path):
        configs_dir = tmp_path / "configs" / "experiments" / "search"
        output_dir = tmp_path / "results" / "strategy_search"
        _generate_dry(tmp_path, configs_dir=configs_dir, output_dir=output_dir)
        # No YAML files should be written in dry_run mode
        assert not any(configs_dir.glob("*.yaml")) if configs_dir.exists() else True

    def test_dry_run_paths_have_correct_structure(self, tmp_path):
        _, paths = _generate_dry(tmp_path)
        assert len(paths) == 46  # 45 generated + 1 baseline

    def test_manifest_has_search_phase(self, tmp_path):
        manifest, _ = _generate_dry(tmp_path)
        assert manifest["search_phase"] == "6A"

    def test_manifest_has_base_experiment(self, tmp_path):
        manifest, _ = _generate_dry(tmp_path)
        assert manifest["base_experiment"] == "canonical_ml_multi_asset"

    def test_manifest_candidates_have_required_keys(self, tmp_path):
        manifest, _ = _generate_dry(tmp_path)
        for entry in manifest["candidates"]:
            for key in ("candidate_id", "model_id", "feature_set_id", "cost_bps", "is_baseline"):
                assert key in entry, f"manifest entry missing {key!r}"


def _generate_dry(tmp_path, configs_dir=None, output_dir=None):
    """Call generate_all_configs in dry_run mode with a mock base config."""
    from src.research_search.candidate_generator import generate_all_configs

    _cd = configs_dir or tmp_path / "configs" / "experiments" / "search"
    _od = output_dir or tmp_path / "results" / "strategy_search"

    # Patch load_base_config to avoid needing the real YAML on disk
    with patch("src.research_search.candidate_generator.load_base_config",
               return_value=copy.deepcopy(_BASE_CFG)):
        return generate_all_configs(
            configs_dir=_cd,
            output_dir=_od,
            dry_run=True,
        )


# ---------------------------------------------------------------------------
# I — Registry compliance in all generated configs
# ---------------------------------------------------------------------------


class TestRegistryCompliance:
    def test_no_unsupported_model_types(self):
        for c in SearchSpace.all_candidates():
            cfg = build_candidate_config(c, _BASE_CFG)
            assert cfg["model"]["type"] in SUPPORTED_MODEL_TYPES

    def test_no_unsupported_feature_types(self):
        for c in SearchSpace.all_candidates():
            cfg = build_candidate_config(c, _BASE_CFG)
            for entry in cfg["features"]["entries"]:
                assert entry["type"] in SUPPORTED_FEATURE_TYPES
