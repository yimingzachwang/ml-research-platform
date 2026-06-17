"""Registry consistency tests.

These tests are the primary guard against registry drift. They assert that
the registry YAML files stay in sync with the engine's authoritative
frozensets in ml_config.py and draft_validator.py.

A failure here means either:
 (a) a YAML file was edited without updating the engine, or
 (b) the engine was updated without updating the YAML file.

Both are bugs. Fix the YAML to match the engine — never the reverse
(the engine frozensets are the source of truth in Phase 1).
"""

from __future__ import annotations

import pytest

from src.agent.registry.registry_loader import load_registry
from src.experiments.ml_config import (
    get_feature_required_params,
    get_valid_feature_types,
    get_valid_label_types,
    get_valid_model_types,
    get_valid_prediction_normalizations,
    get_valid_signal_types,
    get_valid_weighting_schemes,
)
from src.orchestration.config_generation.draft_validator import allowed_change_paths

# These 8 types appear in src/features/families.FEATURE_FAMILIES but are NOT
# in _VALID_FEATURE_TYPES and cannot be executed by the engine.
# Phase 1 must not include them in the registry.
_UNIMPLEMENTED_FAMILY_TYPES: frozenset[str] = frozenset({
    "cross_sectional_rank",
    "ewm_volatility",
    "relative_momentum",
    "rolling_kurtosis",
    "rolling_minmax",
    "rolling_rank",
    "sma_crossover",
    "vol_expansion",
})


@pytest.fixture(scope="module")
def registry():
    return load_registry()


# ---------------------------------------------------------------------------
# Engine-registry parity assertions
# ---------------------------------------------------------------------------

class TestModelRegistryParity:
    def test_model_names_equal_engine_frozenset(self, registry):
        assert registry.model_names() == get_valid_model_types()

    def test_all_engine_models_present(self, registry):
        for name in get_valid_model_types():
            assert name in registry.model_names(), f"Engine model '{name}' missing from registry"

    def test_no_extra_models_beyond_engine(self, registry):
        extra = registry.model_names() - get_valid_model_types()
        assert not extra, f"Registry contains models not in engine: {extra}"


class TestFeatureRegistryParity:
    def test_feature_names_equal_engine_frozenset(self, registry):
        assert registry.feature_names() == get_valid_feature_types()

    def test_all_engine_features_present(self, registry):
        for name in get_valid_feature_types():
            assert name in registry.feature_names(), f"Engine feature '{name}' missing from registry"

    def test_no_extra_features_beyond_engine(self, registry):
        extra = registry.feature_names() - get_valid_feature_types()
        assert not extra, f"Registry contains features not in engine: {extra}"

    def test_feature_required_params_match_engine(self, registry):
        engine_params = get_feature_required_params()
        for fname in registry.feature_names():
            expected = set(engine_params.get(fname, frozenset()))
            actual = set(registry.feature(fname).required_params)
            assert actual == expected, (
                f"required_params mismatch for feature '{fname}': "
                f"registry={actual}, engine={expected}"
            )

    def test_unimplemented_family_types_absent(self, registry):
        """Types in FEATURE_FAMILIES but not in _VALID_FEATURE_TYPES must not appear."""
        present = registry.feature_names() & _UNIMPLEMENTED_FAMILY_TYPES
        assert not present, (
            f"Registry contains unimplemented feature family types: {present}. "
            f"These are in families.py but not in _VALID_FEATURE_TYPES."
        )

    def test_compute_returns_present(self, registry):
        """compute_returns is in _VALID_FEATURE_TYPES but NOT in FEATURE_FAMILIES."""
        assert "compute_returns" in registry.feature_names()

    def test_compute_returns_has_no_required_params(self, registry):
        assert len(registry.feature("compute_returns").required_params) == 0

    def test_vol_compression_has_two_required_params(self, registry):
        rp = registry.feature("vol_compression").required_params
        assert rp == frozenset({"long_window", "short_window"})


class TestSignalRegistryParity:
    def test_signal_names_equal_engine_frozenset(self, registry):
        assert registry.signal_names() == get_valid_signal_types()

    def test_panel_signals_flagged_requires_multi_asset(self, registry):
        for name in {"top_n", "long_short", "normalize"}:
            assert registry.signal(name).requires_multi_asset is True, (
                f"Panel signal '{name}' must have requires_multi_asset=True"
            )

    def test_single_asset_signals_not_flagged(self, registry):
        for name in {"sign", "threshold"}:
            assert registry.signal(name).requires_multi_asset is False, (
                f"Single-asset signal '{name}' must have requires_multi_asset=False"
            )

    def test_top_n_requires_n_param(self, registry):
        assert "n" in registry.signal("top_n").required_params

    def test_long_short_requires_n_long_and_n_short(self, registry):
        rp = registry.signal("long_short").required_params
        assert rp == frozenset({"n_long", "n_short"})


class TestLabelRegistryParity:
    def test_label_names_equal_engine_frozenset(self, registry):
        assert registry.label_names() == get_valid_label_types()

    def test_ranking_target_requires_multi_asset(self, registry):
        assert registry.label("ranking_target").requires_multi_asset is True

    def test_non_panel_labels_not_flagged(self, registry):
        for name in {"forward_returns", "binary_direction", "volatility_target"}:
            assert registry.label(name).requires_multi_asset is False, (
                f"Label '{name}' should not be flagged requires_multi_asset"
            )

    def test_all_labels_have_horizon_required_param(self, registry):
        for name in registry.label_names():
            assert "horizon" in registry.label(name).required_params, (
                f"Label '{name}' should require 'horizon' param"
            )


class TestWeightingSchemasParity:
    def test_weighting_scheme_names_equal_engine_frozenset(self, registry):
        assert registry.weighting_scheme_names() == get_valid_weighting_schemes()

    def test_prediction_normalization_names_equal_engine_frozenset(self, registry):
        assert registry.prediction_normalization_names() == get_valid_prediction_normalizations()


class TestChangePathsParity:
    def test_changeable_paths_equal_allowed_change_paths(self, registry):
        engine_paths = frozenset(allowed_change_paths())
        assert registry.changeable_paths() == engine_paths

    def test_no_extra_paths_beyond_engine(self, registry):
        extra = registry.changeable_paths() - frozenset(allowed_change_paths())
        assert not extra, f"Registry contains change paths not in engine whitelist: {extra}"

    def test_no_missing_paths_from_engine(self, registry):
        missing = frozenset(allowed_change_paths()) - registry.changeable_paths()
        assert not missing, f"Engine change paths absent from registry: {missing}"

    def test_change_path_count(self, registry):
        assert len(registry.changeable_paths()) == len(allowed_change_paths())


# ---------------------------------------------------------------------------
# Structural / format assertions
# ---------------------------------------------------------------------------

class TestRegistryStructure:
    def test_all_model_entries_have_display_name(self, registry):
        for name in registry.model_names():
            assert registry.model(name).display_name, f"Model '{name}' missing display_name"

    def test_all_feature_entries_have_display_name(self, registry):
        for name in registry.feature_names():
            assert registry.feature(name).display_name, f"Feature '{name}' missing display_name"

    def test_all_signal_entries_have_display_name(self, registry):
        for name in registry.signal_names():
            assert registry.signal(name).display_name, f"Signal '{name}' missing display_name"

    def test_all_label_entries_have_display_name(self, registry):
        for name in registry.label_names():
            assert registry.label(name).display_name, f"Label '{name}' missing display_name"

    def test_model_count_matches_engine(self, registry):
        assert len(registry.model_names()) == len(get_valid_model_types())

    def test_feature_count_matches_engine(self, registry):
        assert len(registry.feature_names()) == len(get_valid_feature_types())

    def test_signal_count_matches_engine(self, registry):
        assert len(registry.signal_names()) == len(get_valid_signal_types())

    def test_label_count_matches_engine(self, registry):
        assert len(registry.label_names()) == len(get_valid_label_types())

    def test_rolling_beta_documents_market_ticker(self, registry):
        entry = registry.feature("rolling_beta")
        assert "market_ticker" in entry.optional_params, (
            "rolling_beta must document market_ticker as an optional_param "
            "(defaults to SPY in ml_factory.py:178)"
        )

    def test_validation_names_include_all_engine_types(self, registry):
        from src.experiments.ml_config import _VALID_VALIDATION_TYPES
        assert registry.validation_names() == _VALID_VALIDATION_TYPES
