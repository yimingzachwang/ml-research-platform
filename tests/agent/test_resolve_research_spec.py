"""End-to-end tests for resolve_research_spec().

Covers all 9 specified test scenarios plus structural and provenance tests.
"""

from __future__ import annotations

import pytest

from src.agent.intent.research_spec import ResearchSpec
from src.agent.registry.registry_loader import load_registry
from src.agent.resolver.resolution_result import ResearchSpecResolutionResult
from src.agent.resolver.resolve_research_spec import resolve_research_spec
from src.agent.resolver.resolved_spec import ResolvedResearchSpec


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def _resolve(spec: ResearchSpec, mode: str = "recommended", reg=None):
    return resolve_research_spec(spec, registry=reg, mode=mode)


class TestResultStructure:
    def test_returns_resolution_result(self, registry):
        result = resolve_research_spec(ResearchSpec(), registry=registry)
        assert isinstance(result, ResearchSpecResolutionResult)

    def test_input_spec_preserved(self, registry):
        spec = ResearchSpec(model="RidgeRegression")
        result = resolve_research_spec(spec, registry=registry)
        assert result.input_spec is spec

    def test_resolved_spec_is_resolved_research_spec(self, registry):
        result = resolve_research_spec(ResearchSpec(), registry=registry)
        assert result.resolved_spec is None or isinstance(result.resolved_spec, ResolvedResearchSpec)

    def test_status_is_one_of_three_values(self, registry):
        result = resolve_research_spec(ResearchSpec(), registry=registry)
        assert result.status in ("resolved", "needs_user_input", "invalid")

    def test_missing_fields_is_list(self, registry):
        result = resolve_research_spec(ResearchSpec(), registry=registry)
        assert isinstance(result.missing_fields, list)

    def test_field_options_is_dict(self, registry):
        result = resolve_research_spec(ResearchSpec(), registry=registry)
        assert isinstance(result.field_options, dict)

    def test_registry_loaded_automatically_when_none(self):
        result = resolve_research_spec(ResearchSpec(end_date="2024-12-31"), registry=None, mode="recommended")
        assert result is not None

    def test_summary_returns_string(self, registry):
        result = resolve_research_spec(ResearchSpec(), registry=registry)
        assert isinstance(result.summary(), str)


# ---------------------------------------------------------------------------
# Scenario 1: empty spec in strict mode → needs_user_input
# ---------------------------------------------------------------------------

class TestScenario1EmptySpecStrictMode:
    def test_status_is_needs_user_input(self, registry):
        result = resolve_research_spec(ResearchSpec(), registry=registry, mode="strict")
        assert result.status == "needs_user_input"

    def test_missing_fields_are_not_empty(self, registry):
        result = resolve_research_spec(ResearchSpec(), registry=registry, mode="strict")
        assert len(result.missing_fields) > 0

    def test_field_options_provided_for_each_missing_field(self, registry):
        result = resolve_research_spec(ResearchSpec(), registry=registry, mode="strict")
        for mf in result.missing_fields:
            # field_options may be empty for some fields, but key should exist
            assert mf in result.field_options

    def test_resolved_spec_present_even_in_needs_user_input(self, registry):
        result = resolve_research_spec(ResearchSpec(), registry=registry, mode="strict")
        assert result.resolved_spec is not None


# ---------------------------------------------------------------------------
# Scenario 2: empty spec in recommended mode
# ---------------------------------------------------------------------------

class TestScenario2EmptySpecRecommendedMode:
    def test_status_is_needs_user_input_because_end_date_missing(self, registry):
        result = resolve_research_spec(ResearchSpec(), registry=registry, mode="recommended")
        assert result.status == "needs_user_input"
        assert "end_date" in result.missing_fields

    def test_most_fields_are_defaulted(self, registry):
        result = resolve_research_spec(ResearchSpec(), registry=registry, mode="recommended")
        assert result.resolved_spec is not None
        rs = result.resolved_spec
        # model, label, universe, features should all be filled
        assert rs.model is not None
        assert rs.label is not None
        assert rs.universe is not None
        assert rs.features

    def test_all_defaulted_fields_are_explicit(self, registry):
        result = resolve_research_spec(ResearchSpec(), registry=registry, mode="recommended")
        assert len(result.defaulted_fields) > 0
        assert len(result.assumptions) > 0

    def test_requires_confirmation_true(self, registry):
        result = resolve_research_spec(ResearchSpec(), registry=registry, mode="recommended")
        assert result.requires_confirmation is True


# ---------------------------------------------------------------------------
# Scenario 3: preset-only spec
# ---------------------------------------------------------------------------

class TestScenario3PresetOnly:
    def test_canonical_preset_with_end_date_resolves(self, registry):
        spec = ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31")
        result = resolve_research_spec(spec, registry=registry)
        assert result.status == "resolved"

    def test_single_asset_preset_with_end_date_resolves(self, registry):
        spec = ResearchSpec(preset="single_asset_ml", end_date="2024-12-31")
        result = resolve_research_spec(spec, registry=registry)
        assert result.status == "resolved"

    def test_preset_without_end_date_needs_user_input(self, registry):
        spec = ResearchSpec(preset="canonical_multi_asset_etf")
        result = resolve_research_spec(spec, registry=registry)
        assert result.status == "needs_user_input"
        assert "end_date" in result.missing_fields


# ---------------------------------------------------------------------------
# Scenario 4: preset with override
# ---------------------------------------------------------------------------

class TestScenario4PresetWithOverride:
    def test_user_model_overrides_preset(self, registry):
        spec = ResearchSpec(
            preset="single_asset_ml",
            model="LassoRegression",
            model_params={"alpha": 0.5},
            end_date="2024-12-31",
        )
        result = resolve_research_spec(spec, registry=registry)
        assert result.status == "resolved"
        assert result.resolved_spec.model == "LassoRegression"

    def test_preset_signal_preserved_when_not_overridden(self, registry):
        spec = ResearchSpec(
            preset="single_asset_ml",
            end_date="2024-12-31",
        )
        result = resolve_research_spec(spec, registry=registry)
        assert result.resolved_spec.signal == "sign"  # from preset

    def test_user_signal_overrides_preset(self, registry):
        spec = ResearchSpec(
            preset="single_asset_ml",
            signal="threshold",
            end_date="2024-12-31",
        )
        result = resolve_research_spec(spec, registry=registry)
        assert result.resolved_spec.signal == "threshold"

    def test_user_label_override_checked_for_compatibility(self, registry):
        # Override single-asset preset label with ranking_target → should be invalid
        spec = ResearchSpec(
            preset="single_asset_ml",
            label="ranking_target",
            label_horizon=21,
            end_date="2024-12-31",
        )
        result = resolve_research_spec(spec, registry=registry)
        assert result.status == "invalid"

    def test_provenance_shows_user_win_over_preset(self, registry):
        spec = ResearchSpec(
            preset="single_asset_ml",
            model="LassoRegression",
            end_date="2024-12-31",
        )
        result = resolve_research_spec(spec, registry=registry)
        rs = result.resolved_spec
        assert rs.source_of("model") == "user"
        assert rs.source_of("signal") == "preset"


# ---------------------------------------------------------------------------
# Scenario 5: invalid design — single-asset + panel label
# ---------------------------------------------------------------------------

class TestScenario5InvalidDesign:
    def test_single_asset_ranking_target_is_invalid(self, registry):
        spec = ResearchSpec(
            universe="single_asset_spy",
            label="ranking_target",
            label_horizon=21,
            model="RidgeRegression",
            signal="sign",
            features=["momentum"],
            asset_class="equity_index",
            frequency="monthly",
            start_date="2015-01-01",
            end_date="2024-12-31",
            validation_protocol="rolling",
            transaction_cost_bps=10.0,
        )
        result = resolve_research_spec(spec, registry=registry)
        assert result.status == "invalid"
        assert len(result.errors) > 0

    def test_single_asset_panel_signal_is_invalid(self, registry):
        spec = ResearchSpec(
            universe="single_asset_spy",
            label="forward_returns",
            label_horizon=21,
            model="RidgeRegression",
            signal="top_n",
            signal_params={"n": 3},
            features=["momentum"],
            asset_class="equity_index",
            frequency="monthly",
            start_date="2015-01-01",
            end_date="2024-12-31",
            validation_protocol="rolling",
            transaction_cost_bps=10.0,
        )
        result = resolve_research_spec(spec, registry=registry)
        assert result.status == "invalid"

    def test_error_message_is_informative(self, registry):
        spec = ResearchSpec(
            preset="single_asset_ml",
            label="ranking_target",
            label_horizon=21,
            end_date="2024-12-31",
        )
        result = resolve_research_spec(spec, registry=registry)
        assert result.status == "invalid"
        assert any("ranking_target" in e for e in result.errors)

    def test_resolved_spec_present_on_invalid(self, registry):
        spec = ResearchSpec(
            preset="single_asset_ml",
            label="ranking_target",
            label_horizon=21,
            end_date="2024-12-31",
        )
        result = resolve_research_spec(spec, registry=registry)
        # resolved_spec is still returned so the caller can inspect what was built
        assert result.resolved_spec is not None
        assert result.resolved_spec.compatibility_errors


# ---------------------------------------------------------------------------
# Scenario 6: multi-asset canonical preset resolves
# ---------------------------------------------------------------------------

class TestScenario6CanonicalMultiAssetPreset:
    def test_canonical_preset_resolves_successfully(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            end_date="2024-12-31",
        )
        result = resolve_research_spec(spec, registry=registry)
        assert result.status == "resolved"

    def test_resolved_has_13_tickers(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            end_date="2024-12-31",
        )
        result = resolve_research_spec(spec, registry=registry)
        assert len(result.resolved_spec.tickers) == 13

    def test_resolved_model_is_ridge(self, registry):
        spec = ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31")
        result = resolve_research_spec(spec, registry=registry)
        assert result.resolved_spec.model == "RidgeRegression"

    def test_resolved_signal_is_top_n(self, registry):
        spec = ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31")
        result = resolve_research_spec(spec, registry=registry)
        assert result.resolved_spec.signal == "top_n"

    def test_resolved_no_compat_errors(self, registry):
        spec = ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31")
        result = resolve_research_spec(spec, registry=registry)
        assert not result.errors


# ---------------------------------------------------------------------------
# Scenario 7: custom ticker list without tickers → needs_user_input
# ---------------------------------------------------------------------------

class TestScenario7CustomTickerListWithoutTickers:
    def test_custom_ticker_list_no_tickers_needs_input(self, registry):
        spec = ResearchSpec(
            universe="custom_ticker_list",
            model="RidgeRegression",
            label="forward_returns",
            label_horizon=21,
            signal="sign",
            features=["momentum"],
            start_date="2015-01-01",
            end_date="2024-12-31",
            frequency="monthly",
            validation_protocol="rolling",
            transaction_cost_bps=10.0,
        )
        result = resolve_research_spec(spec, registry=registry)
        # asset_class is still missing for custom universe
        assert result.status == "needs_user_input"

    def test_custom_ticker_list_without_tickers_no_compat_error_on_panel_signal(self, registry):
        # The panel signal compatibility check uses tickers list — if empty it is skipped
        spec = ResearchSpec(
            universe="custom_ticker_list",
            signal="top_n",
        )
        result = resolve_research_spec(spec, registry=registry)
        # Should be needs_user_input (missing fields) not invalid (no tickers to count)
        compat_errors = [e for e in result.errors if "COMPAT_001" in e]
        assert not compat_errors


# ---------------------------------------------------------------------------
# Scenario 8: custom ticker list with tickers → resolves if all fields present
# ---------------------------------------------------------------------------

class TestScenario8CustomTickerListWithTickers:
    def test_custom_ticker_list_with_tickers_resolves(self, registry):
        spec = ResearchSpec(
            asset_class="equity_index",
            universe="custom_ticker_list",
            tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            model="RidgeRegression",
            model_params={"alpha": 1.0},
            label="forward_returns",
            label_horizon=21,
            signal="top_n",
            signal_params={"n": 2},
            features=["momentum", "rolling_volatility"],
            start_date="2015-01-01",
            end_date="2024-12-31",
            frequency="monthly",
            validation_protocol="rolling",
            train_window="36m",
            test_window="6m",
            transaction_cost_bps=10.0,
        )
        result = resolve_research_spec(spec, registry=registry)
        assert result.status == "resolved"

    def test_custom_single_ticker_with_panel_signal_is_invalid(self, registry):
        spec = ResearchSpec(
            asset_class="equity_index",
            universe="custom_ticker_list",
            tickers=["AAPL"],
            model="RidgeRegression",
            model_params={"alpha": 1.0},
            label="forward_returns",
            label_horizon=21,
            signal="top_n",
            signal_params={"n": 1},
            features=["momentum"],
            start_date="2015-01-01",
            end_date="2024-12-31",
            frequency="monthly",
            validation_protocol="rolling",
            transaction_cost_bps=10.0,
        )
        result = resolve_research_spec(spec, registry=registry)
        assert result.status == "invalid"


# ---------------------------------------------------------------------------
# Provenance tests
# ---------------------------------------------------------------------------

class TestProvenanceTracking:
    def test_user_fields_recorded_in_user_specified(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            end_date="2024-12-31",
        )
        result = resolve_research_spec(spec, registry=registry)
        rs = result.resolved_spec
        assert "end_date" in rs.user_specified_fields
        assert "preset" in rs.user_specified_fields

    def test_preset_fields_recorded_in_preset_fields(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            end_date="2024-12-31",
        )
        result = resolve_research_spec(spec, registry=registry)
        rs = result.resolved_spec
        assert "universe" in rs.preset_fields
        assert "model" in rs.preset_fields

    def test_defaulted_fields_recorded_in_defaulted_fields(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            end_date="2024-12-31",
        )
        result = resolve_research_spec(spec, registry=registry)
        rs = result.resolved_spec
        # asset_class and start_date should come from defaults (not preset)
        assert "start_date" in rs.defaulted_fields

    def test_source_of_returns_correct_provenance(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            end_date="2024-12-31",
        )
        result = resolve_research_spec(spec, registry=registry)
        rs = result.resolved_spec
        assert rs.source_of("end_date") == "user"
        assert rs.source_of("universe") == "preset"
        assert rs.source_of("start_date") == "default"

    def test_assumptions_list_not_empty_after_recommended_resolution(self, registry):
        spec = ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31")
        result = resolve_research_spec(spec, registry=registry)
        assert len(result.assumptions) > 0

    def test_requires_confirmation_when_defaults_applied(self, registry):
        spec = ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31")
        result = resolve_research_spec(spec, registry=registry)
        # start_date was defaulted (recommended default) → requires confirmation
        assert result.requires_confirmation is True


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_unknown_preset_returns_invalid_not_exception(self, registry):
        spec = ResearchSpec(preset="totally_unknown_preset_xyz")
        result = resolve_research_spec(spec, registry=registry)
        assert result.status == "invalid"
        assert len(result.errors) > 0

    def test_negative_transaction_cost_is_invalid(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            end_date="2024-12-31",
            transaction_cost_bps=-5.0,
        )
        result = resolve_research_spec(spec, registry=registry)
        assert result.status == "invalid"

    def test_bad_date_order_is_invalid(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            start_date="2024-01-01",
            end_date="2020-01-01",
        )
        result = resolve_research_spec(spec, registry=registry)
        assert result.status == "invalid"

    def test_unknown_feature_is_invalid(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            end_date="2024-12-31",
            features=["momentum", "cross_sectional_rank"],  # not in registry
        )
        result = resolve_research_spec(spec, registry=registry)
        assert result.status == "invalid"


# ---------------------------------------------------------------------------
# Warnings coexist with resolution
# ---------------------------------------------------------------------------

class TestWarningsWithResolution:
    def test_rolling_beta_warning_does_not_prevent_resolution(self, registry):
        spec = ResearchSpec(
            preset="single_asset_ml",
            end_date="2024-12-31",
            features=["momentum", "rolling_beta"],
        )
        result = resolve_research_spec(spec, registry=registry)
        assert result.status == "resolved"
        assert len(result.warnings) > 0

    def test_lasso_info_does_not_prevent_resolution(self, registry):
        spec = ResearchSpec(
            preset="single_asset_ml",
            model="LassoRegression",
            model_params={"alpha": 1.0},
            end_date="2024-12-31",
        )
        result = resolve_research_spec(spec, registry=registry)
        assert result.status == "resolved"
        assert len(result.warnings) > 0  # info messages surfaced as warnings

    def test_zero_cost_panel_signal_warning_but_resolved(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            end_date="2024-12-31",
            transaction_cost_bps=0.0,
        )
        result = resolve_research_spec(spec, registry=registry)
        assert result.status == "resolved"
        assert any("transaction_cost_bps=0" in w or "COMPAT_006" in w or "0 bps" in w
                   for w in result.warnings)
