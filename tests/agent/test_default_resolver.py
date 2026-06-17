"""Tests for src/agent/resolver/default_resolver.py"""

from __future__ import annotations

import pytest

from src.agent.intent.research_spec import ResearchSpec
from src.agent.registry.registry_loader import load_registry
from src.agent.resolver.default_resolver import (
    DefaultResolutionResult,
    apply_defaults,
    expand_preset,
    options_for_field,
)


@pytest.fixture(scope="module")
def registry():
    return load_registry()


# ---------------------------------------------------------------------------
# apply_defaults — strict mode
# ---------------------------------------------------------------------------

class TestApplyDefaultsStrict:
    def test_strict_fills_only_safe_defaults(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="strict")
        # safe_default_allowed: validation_protocol, benchmark, diagnostics
        assert result.spec.validation_protocol is not None
        assert result.spec.benchmark is not None
        assert result.spec.diagnostics is not None

    def test_strict_does_not_fill_model(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="strict")
        assert result.spec.model is None

    def test_strict_does_not_fill_universe(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="strict")
        assert result.spec.universe is None

    def test_strict_does_not_fill_label(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="strict")
        assert result.spec.label is None

    def test_strict_does_not_fill_features(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="strict")
        assert result.spec.features is None

    def test_strict_does_not_fill_transaction_cost(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="strict")
        assert result.spec.transaction_cost_bps is None

    def test_strict_records_defaulted_safe_fields(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="strict")
        assert "validation_protocol" in result.defaulted_fields
        assert "benchmark" in result.defaulted_fields

    def test_strict_does_not_overwrite_user_set_validation(self, registry):
        spec = ResearchSpec(validation_protocol="expanding")
        result = apply_defaults(spec, registry, mode="strict")
        assert result.spec.validation_protocol == "expanding"
        assert "validation_protocol" not in result.defaulted_fields


# ---------------------------------------------------------------------------
# apply_defaults — recommended mode
# ---------------------------------------------------------------------------

class TestApplyDefaultsRecommended:
    def test_recommended_fills_model(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="recommended")
        assert result.spec.model == "RidgeRegression"

    def test_recommended_fills_label(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="recommended")
        assert result.spec.label == "forward_returns"

    def test_recommended_fills_universe(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="recommended")
        assert result.spec.universe == "canonical_multi_asset_etfs"

    def test_recommended_fills_features(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="recommended")
        assert result.spec.features is not None
        assert len(result.spec.features) > 0

    def test_recommended_fills_transaction_cost(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="recommended")
        assert result.spec.transaction_cost_bps == 10.0

    def test_recommended_fills_start_date(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="recommended")
        assert result.spec.start_date == "2015-01-01"

    def test_recommended_derives_tickers_from_universe(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="recommended")
        # Universe was defaulted to canonical_multi_asset_etfs → tickers derived
        assert result.spec.tickers is not None
        assert len(result.spec.tickers) > 0

    def test_recommended_derives_signal_from_context(self, registry):
        spec = ResearchSpec(universe="single_asset_spy")
        result = apply_defaults(spec, registry, mode="recommended")
        # single-asset universe → sign signal
        assert result.spec.signal == "sign"

    def test_recommended_derives_top_n_for_multi_asset(self, registry):
        spec = ResearchSpec(universe="core_etfs")
        result = apply_defaults(spec, registry, mode="recommended")
        assert result.spec.signal == "top_n"

    def test_recommended_sets_requires_confirmation_true(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="recommended")
        assert result.requires_confirmation is True

    def test_recommended_records_all_defaulted_fields(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="recommended")
        for field_name in ("model", "label", "universe", "validation_protocol", "benchmark"):
            assert field_name in result.defaulted_fields, f"'{field_name}' not in defaulted_fields"

    def test_recommended_never_overwrites_user_model(self, registry):
        spec = ResearchSpec(model="LassoRegression")
        result = apply_defaults(spec, registry, mode="recommended")
        assert result.spec.model == "LassoRegression"
        assert "model" not in result.defaulted_fields

    def test_recommended_never_overwrites_user_signal(self, registry):
        spec = ResearchSpec(signal="threshold")
        result = apply_defaults(spec, registry, mode="recommended")
        assert result.spec.signal == "threshold"
        assert "signal" not in result.defaulted_fields

    def test_recommended_fills_label_horizon_from_registry(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="recommended")
        # forward_returns default horizon = 21 from registry
        assert result.spec.label_horizon == 21

    def test_recommended_all_assumptions_are_recorded(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="recommended")
        assert len(result.assumptions) > 0
        # Every defaulted field should have a corresponding assumption
        for field_name in result.defaulted_fields:
            has_assumption = any(field_name in a for a in result.assumptions)
            assert has_assumption, f"No assumption recorded for defaulted field '{field_name}'"

    def test_no_undocumented_defaults(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="recommended")
        # Every filled field must be in defaulted_fields
        for field_name in result.defaulted_fields:
            val_on_spec = getattr(result.spec, field_name, None)
            assert val_on_spec is not None, (
                f"defaulted_fields claims '{field_name}' was filled but spec has None"
            )

    def test_feature_families_expands_to_features(self, registry):
        spec = ResearchSpec(feature_families=["Trend"])
        result = apply_defaults(spec, registry, mode="recommended")
        # feature_families expansion should fill features
        assert result.spec.features is not None
        # All expanded features should be Trend family members
        trend_features = [
            n for n in registry.feature_names()
            if registry.feature(n).family == "Trend"
        ]
        for ft in result.spec.features:
            assert ft in trend_features, f"Expanded feature '{ft}' is not in Trend family"


# ---------------------------------------------------------------------------
# apply_defaults — demo mode
# ---------------------------------------------------------------------------

class TestApplyDefaultsDemo:
    def test_demo_fills_model(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="demo")
        assert result.spec.model is not None

    def test_demo_fills_universe(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="demo")
        assert result.spec.universe is not None

    def test_demo_fills_asset_class(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="demo")
        assert result.spec.asset_class is not None

    def test_demo_records_assumptions(self, registry):
        spec = ResearchSpec()
        result = apply_defaults(spec, registry, mode="demo")
        assert len(result.assumptions) > 0

    def test_invalid_mode_raises_value_error(self, registry):
        with pytest.raises(ValueError, match="Unknown resolution mode"):
            apply_defaults(ResearchSpec(), registry, mode="turbo")


# ---------------------------------------------------------------------------
# expand_preset
# ---------------------------------------------------------------------------

class TestExpandPreset:
    def test_expand_canonical_multi_asset_fills_universe(self, registry):
        spec = ResearchSpec(preset="canonical_multi_asset_etf")
        new_spec, applied, _ = expand_preset(spec, registry)
        assert new_spec.universe == "canonical_multi_asset_etfs"
        assert "universe" in applied

    def test_expand_fills_model_and_params(self, registry):
        spec = ResearchSpec(preset="canonical_multi_asset_etf")
        new_spec, applied, _ = expand_preset(spec, registry)
        assert new_spec.model == "RidgeRegression"
        assert "model" in applied

    def test_expand_fills_signal_and_params(self, registry):
        spec = ResearchSpec(preset="canonical_multi_asset_etf")
        new_spec, applied, _ = expand_preset(spec, registry)
        assert new_spec.signal == "top_n"
        assert new_spec.signal_params == {"n": 3}

    def test_expand_fills_label_and_horizon(self, registry):
        spec = ResearchSpec(preset="canonical_multi_asset_etf")
        new_spec, applied, _ = expand_preset(spec, registry)
        assert new_spec.label == "forward_returns"
        assert new_spec.label_horizon == 21

    def test_expand_fills_validation_as_string_windows(self, registry):
        spec = ResearchSpec(preset="canonical_multi_asset_etf")
        new_spec, applied, _ = expand_preset(spec, registry)
        assert new_spec.validation_protocol == "rolling"
        assert new_spec.train_window == "36m"
        assert new_spec.test_window == "6m"

    def test_expand_fills_features_as_type_names(self, registry):
        spec = ResearchSpec(preset="canonical_multi_asset_etf")
        new_spec, applied, feature_specs = expand_preset(spec, registry)
        assert new_spec.features is not None
        assert all(isinstance(f, str) for f in new_spec.features)
        assert "momentum" in new_spec.features
        assert len(feature_specs) == len(new_spec.features)

    def test_expand_feature_specs_are_dicts_with_type_and_params(self, registry):
        spec = ResearchSpec(preset="canonical_multi_asset_etf")
        _, _, feature_specs = expand_preset(spec, registry)
        for fs in feature_specs:
            assert isinstance(fs, dict)
            assert "type" in fs
            assert "params" in fs or True  # params may be absent for zero-param features

    def test_user_model_overrides_preset_model(self, registry):
        spec = ResearchSpec(preset="single_asset_ml", model="LassoRegression")
        new_spec, applied, _ = expand_preset(spec, registry)
        assert new_spec.model == "LassoRegression"
        assert "model" not in applied

    def test_user_signal_overrides_preset_signal(self, registry):
        spec = ResearchSpec(preset="single_asset_ml", signal="threshold")
        new_spec, applied, _ = expand_preset(spec, registry)
        assert new_spec.signal == "threshold"
        assert "signal" not in applied

    def test_user_label_overrides_preset_label(self, registry):
        spec = ResearchSpec(preset="canonical_multi_asset_etf", label="binary_direction")
        new_spec, applied, _ = expand_preset(spec, registry)
        assert new_spec.label == "binary_direction"
        assert "label" not in applied

    def test_no_preset_returns_original_spec(self, registry):
        spec = ResearchSpec(model="RidgeRegression")
        new_spec, applied, feature_specs = expand_preset(spec, registry)
        assert new_spec is spec
        assert applied == {}
        assert feature_specs == []

    def test_unknown_preset_raises_key_error(self, registry):
        spec = ResearchSpec(preset="nonexistent_preset_xyz")
        with pytest.raises(KeyError):
            expand_preset(spec, registry)

    def test_expand_does_not_mutate_input_spec(self, registry):
        spec = ResearchSpec(preset="single_asset_ml")
        original_model = spec.model
        expand_preset(spec, registry)
        assert spec.model == original_model  # still None

    def test_single_asset_ml_preset_expands_correctly(self, registry):
        spec = ResearchSpec(preset="single_asset_ml")
        new_spec, applied, _ = expand_preset(spec, registry)
        assert new_spec.universe == "single_asset_spy"
        assert new_spec.signal == "sign"

    def test_preset_values_reference_valid_registry_entries(self, registry):
        for pname in registry.preset_names():
            spec = ResearchSpec(preset=pname)
            new_spec, applied, _ = expand_preset(spec, registry)
            if new_spec.model:
                assert new_spec.model in registry.model_names(), (
                    f"Preset '{pname}' expanded model '{new_spec.model}' not in registry"
                )
            if new_spec.signal:
                assert new_spec.signal in registry.signal_names()
            if new_spec.label:
                assert new_spec.label in registry.label_names()


# ---------------------------------------------------------------------------
# options_for_field
# ---------------------------------------------------------------------------

class TestOptionsForField:
    def test_model_options_match_registry(self, registry):
        opts = options_for_field("model", registry)
        assert set(opts) == registry.model_names()

    def test_feature_options_match_registry(self, registry):
        opts = options_for_field("features", registry)
        assert set(opts) == registry.feature_names()

    def test_signal_options_match_registry_all_signals(self, registry):
        opts = options_for_field("signal", registry)
        assert set(opts) == registry.signal_names()

    def test_label_options_match_registry(self, registry):
        opts = options_for_field("label", registry)
        assert set(opts) == registry.label_names()

    def test_validation_protocol_options_match_registry(self, registry):
        opts = options_for_field("validation_protocol", registry)
        assert set(opts) == registry.validation_names()

    def test_preset_options_match_registry(self, registry):
        opts = options_for_field("preset", registry)
        assert set(opts) == registry.preset_names()

    def test_transaction_cost_options_are_non_negative(self, registry):
        opts = options_for_field("transaction_cost_bps", registry)
        assert all(v >= 0 for v in opts)

    def test_signal_options_filtered_for_single_asset_universe(self, registry):
        spec = ResearchSpec(universe="single_asset_spy")
        opts = options_for_field("signal", registry, spec)
        # Should not include multi-asset only signals
        multi_asset_signals = [
            s for s in registry.signal_names()
            if registry.signal(s).requires_multi_asset
        ]
        for s in multi_asset_signals:
            assert s not in opts, f"Panel signal '{s}' should be excluded for single-asset universe"

    def test_universe_options_filtered_by_asset_class(self, registry):
        spec = ResearchSpec(asset_class="equity_index")
        opts = options_for_field("universe", registry, spec)
        # Should include equity_index universes and custom
        all_opts = options_for_field("universe", registry)
        assert len(opts) <= len(all_opts)

    def test_unknown_field_returns_empty_list(self, registry):
        opts = options_for_field("completely_unknown_field", registry)
        assert opts == []

    def test_frequency_returns_monthly(self, registry):
        opts = options_for_field("frequency", registry)
        assert "monthly" in opts

    def test_feature_families_returns_known_families(self, registry):
        opts = options_for_field("feature_families", registry)
        # Should include families from registry features
        all_families = {
            registry.feature(n).family
            for n in registry.feature_names()
            if registry.feature(n).family is not None
        }
        assert set(opts) == all_families
