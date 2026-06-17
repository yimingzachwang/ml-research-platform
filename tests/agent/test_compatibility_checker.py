"""Tests for src/agent/critic/compatibility_checker.py"""

from __future__ import annotations

import pytest

from src.agent.critic.compatibility_checker import check_compatibility
from src.agent.critic.compatibility_result import CompatibilityIssue, CompatibilityResult
from src.agent.intent.research_spec import ResearchSpec
from src.agent.registry.registry_loader import load_registry
from src.agent.resolver.resolved_spec import ResolvedResearchSpec


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def _spec(**kwargs) -> ResearchSpec:
    """Shorthand for building a ResearchSpec with explicit fields."""
    return ResearchSpec(**kwargs)


class TestRegistryMembershipChecks:
    def test_unknown_model_is_error(self, registry):
        spec = _spec(model="RandomForestRegressor")
        result = check_compatibility(spec, registry)
        assert not result.is_valid
        assert any(i.rule_id == "CHECK_A" for i in result.errors)

    def test_known_model_passes(self, registry):
        spec = _spec(model="RidgeRegression")
        result = check_compatibility(spec, registry)
        errors = [i for i in result.errors if i.rule_id == "CHECK_A"]
        assert not errors

    def test_unknown_feature_type_is_error(self, registry):
        spec = _spec(features=["momentum", "ewm_volatility"])  # ewm_volatility not in registry
        result = check_compatibility(spec, registry)
        assert not result.is_valid
        assert any(i.rule_id == "CHECK_B" for i in result.errors)

    def test_all_known_features_pass(self, registry):
        spec = _spec(features=["momentum", "rolling_volatility", "trend_strength"])
        result = check_compatibility(spec, registry)
        errors = [i for i in result.errors if i.rule_id == "CHECK_B"]
        assert not errors

    def test_unknown_signal_is_error(self, registry):
        spec = _spec(signal="kelly_criterion")
        result = check_compatibility(spec, registry)
        assert not result.is_valid
        assert any(i.rule_id == "CHECK_C" for i in result.errors)

    def test_known_signal_passes(self, registry):
        spec = _spec(signal="sign")
        result = check_compatibility(spec, registry)
        errors = [i for i in result.errors if i.rule_id == "CHECK_C"]
        assert not errors

    def test_unknown_label_is_error(self, registry):
        spec = _spec(label="sharpe_target")
        result = check_compatibility(spec, registry)
        assert not result.is_valid
        assert any(i.rule_id == "CHECK_D" for i in result.errors)

    def test_known_label_passes(self, registry):
        spec = _spec(label="forward_returns")
        result = check_compatibility(spec, registry)
        errors = [i for i in result.errors if i.rule_id == "CHECK_D"]
        assert not errors

    def test_unknown_validation_protocol_is_error(self, registry):
        spec = _spec(validation_protocol="blocked_time_series")
        result = check_compatibility(spec, registry)
        assert not result.is_valid
        assert any(i.rule_id == "CHECK_E" for i in result.errors)

    def test_known_validation_protocol_passes(self, registry):
        spec = _spec(validation_protocol="rolling")
        result = check_compatibility(spec, registry)
        errors = [i for i in result.errors if i.rule_id == "CHECK_E"]
        assert not errors


class TestPanelSignalUniverseCompatibility:
    def test_panel_signal_on_single_asset_is_error(self, registry):
        spec = _spec(signal="top_n", universe="single_asset_spy")
        result = check_compatibility(spec, registry)
        assert not result.is_valid
        assert any(i.rule_id == "COMPAT_001" for i in result.errors)

    def test_panel_signal_long_short_on_single_asset_is_error(self, registry):
        spec = _spec(signal="long_short", universe="single_asset_spy")
        result = check_compatibility(spec, registry)
        assert not result.is_valid

    def test_panel_signal_normalize_on_single_asset_is_error(self, registry):
        spec = _spec(signal="normalize", universe="single_asset_spy")
        result = check_compatibility(spec, registry)
        assert not result.is_valid

    def test_panel_signal_on_multi_asset_is_valid(self, registry):
        spec = _spec(signal="top_n", universe="core_etfs")
        result = check_compatibility(spec, registry)
        assert result.is_valid

    def test_single_asset_signal_on_single_asset_is_valid(self, registry):
        spec = _spec(signal="sign", universe="single_asset_spy")
        result = check_compatibility(spec, registry)
        assert result.is_valid

    def test_signal_check_skipped_when_universe_unknown(self, registry):
        # universe not set → n_tickers is None → check skipped
        spec = _spec(signal="top_n")
        result = check_compatibility(spec, registry)
        compat_001_errors = [i for i in result.errors if i.rule_id == "COMPAT_001"]
        assert not compat_001_errors

    def test_signal_check_uses_tickers_when_set(self, registry):
        # tickers provided directly (1 ticker)
        spec = _spec(signal="top_n", tickers=["SPY"])
        result = check_compatibility(spec, registry)
        assert not result.is_valid


class TestPanelLabelUniverseCompatibility:
    def test_ranking_target_on_single_asset_is_error(self, registry):
        spec = _spec(label="ranking_target", universe="single_asset_spy")
        result = check_compatibility(spec, registry)
        assert not result.is_valid
        assert any(i.rule_id == "COMPAT_002" for i in result.errors)

    def test_ranking_target_on_multi_asset_is_valid(self, registry):
        spec = _spec(label="ranking_target", universe="core_etfs")
        result = check_compatibility(spec, registry)
        assert result.is_valid

    def test_forward_returns_on_single_asset_is_valid(self, registry):
        spec = _spec(label="forward_returns", universe="single_asset_spy")
        result = check_compatibility(spec, registry)
        assert result.is_valid

    def test_binary_direction_on_single_asset_is_valid(self, registry):
        spec = _spec(label="binary_direction", universe="single_asset_spy")
        result = check_compatibility(spec, registry)
        assert result.is_valid


class TestDateOrderCheck:
    def test_start_after_end_is_error(self, registry):
        spec = _spec(start_date="2024-01-01", end_date="2020-01-01")
        result = check_compatibility(spec, registry)
        assert not result.is_valid
        assert any(i.rule_id == "CHECK_I" for i in result.errors)

    def test_start_equal_end_is_error(self, registry):
        spec = _spec(start_date="2020-01-01", end_date="2020-01-01")
        result = check_compatibility(spec, registry)
        assert not result.is_valid

    def test_start_before_end_is_valid(self, registry):
        spec = _spec(start_date="2015-01-01", end_date="2024-12-31")
        result = check_compatibility(spec, registry)
        date_errors = [i for i in result.errors if i.rule_id == "CHECK_I"]
        assert not date_errors

    def test_check_skipped_when_dates_absent(self, registry):
        spec = _spec()
        result = check_compatibility(spec, registry)
        date_errors = [i for i in result.errors if i.rule_id == "CHECK_I"]
        assert not date_errors


class TestTransactionCostCheck:
    def test_negative_transaction_cost_is_error(self, registry):
        spec = _spec(transaction_cost_bps=-1.0)
        result = check_compatibility(spec, registry)
        assert not result.is_valid
        assert any(i.rule_id == "CHECK_J" for i in result.errors)

    def test_zero_transaction_cost_is_valid(self, registry):
        spec = _spec(transaction_cost_bps=0.0)
        result = check_compatibility(spec, registry)
        tc_errors = [i for i in result.errors if i.rule_id == "CHECK_J"]
        assert not tc_errors

    def test_positive_transaction_cost_is_valid(self, registry):
        spec = _spec(transaction_cost_bps=10.0)
        result = check_compatibility(spec, registry)
        tc_errors = [i for i in result.errors if i.rule_id == "CHECK_J"]
        assert not tc_errors


class TestRollingBetaMarketTickerWarning:
    def test_rolling_beta_without_market_ticker_is_warning(self, registry):
        spec = _spec(features=["momentum", "rolling_beta"])
        result = check_compatibility(spec, registry)
        assert result.is_valid  # warnings do not block validity
        assert any(i.rule_id == "COMPAT_004" for i in result.warnings)

    def test_rolling_beta_with_explicit_market_ticker_no_warning(self, registry):
        # Use ResolvedResearchSpec to carry feature_specs (ResearchSpec does not have this field)
        spec = ResolvedResearchSpec(
            features=["rolling_beta"],
            feature_specs=[{"type": "rolling_beta", "params": {"window": 63, "market_ticker": "SPY"}}],
        )
        result = check_compatibility(spec, registry)
        compat_004 = [i for i in result.warnings if i.rule_id == "COMPAT_004"]
        assert not compat_004

    def test_rolling_beta_warning_does_not_block_validity(self, registry):
        spec = _spec(features=["rolling_beta"])
        result = check_compatibility(spec, registry)
        assert result.is_valid  # is_valid only considers errors


class TestVolCompressionWindowWarning:
    def test_short_window_gt_long_window_is_warning(self, registry):
        # Use ResolvedResearchSpec to carry feature_specs
        spec = ResolvedResearchSpec(
            features=["vol_compression"],
            feature_specs=[{
                "type": "vol_compression",
                "params": {"short_window": 63, "long_window": 21}  # inverted
            }],
        )
        result = check_compatibility(spec, registry)
        assert any(i.rule_id == "COMPAT_009" for i in result.warnings)
        assert result.is_valid  # warning only

    def test_correct_vol_compression_windows_no_warning(self, registry):
        spec = ResolvedResearchSpec(
            features=["vol_compression"],
            feature_specs=[{
                "type": "vol_compression",
                "params": {"short_window": 21, "long_window": 63}
            }],
        )
        result = check_compatibility(spec, registry)
        compat_009 = [i for i in result.warnings if i.rule_id == "COMPAT_009"]
        assert not compat_009


class TestLogisticRegressionInfo:
    def test_logistic_with_non_binary_label_produces_info(self, registry):
        spec = _spec(model="LogisticRegression", label="forward_returns")
        result = check_compatibility(spec, registry)
        assert any(i.rule_id == "COMPAT_007" for i in result.info)
        assert result.is_valid

    def test_logistic_with_binary_direction_no_info(self, registry):
        spec = _spec(model="LogisticRegression", label="binary_direction")
        result = check_compatibility(spec, registry)
        compat_007 = [i for i in result.info if i.rule_id == "COMPAT_007"]
        assert not compat_007

    def test_ridge_with_any_label_no_logistic_info(self, registry):
        spec = _spec(model="RidgeRegression", label="forward_returns")
        result = check_compatibility(spec, registry)
        compat_007 = [i for i in result.info if i.rule_id == "COMPAT_007"]
        assert not compat_007


class TestLassoConvergenceInfo:
    def test_lasso_default_max_iter_produces_info(self, registry):
        spec = _spec(model="LassoRegression", model_params={"alpha": 1.0})
        result = check_compatibility(spec, registry)
        assert any(i.rule_id == "COMPAT_010" for i in result.info)
        assert result.is_valid

    def test_elastic_net_low_max_iter_produces_info(self, registry):
        spec = _spec(model="ElasticNetRegression", model_params={"alpha": 1.0, "max_iter": 500})
        result = check_compatibility(spec, registry)
        assert any(i.rule_id == "COMPAT_010" for i in result.info)

    def test_lasso_high_max_iter_no_info(self, registry):
        spec = _spec(model="LassoRegression", model_params={"alpha": 1.0, "max_iter": 5000})
        result = check_compatibility(spec, registry)
        compat_010 = [i for i in result.info if i.rule_id == "COMPAT_010"]
        assert not compat_010

    def test_ridge_no_convergence_info(self, registry):
        spec = _spec(model="RidgeRegression", model_params={"alpha": 1.0})
        result = check_compatibility(spec, registry)
        compat_010 = [i for i in result.info if i.rule_id == "COMPAT_010"]
        assert not compat_010


class TestHighTurnoverTransactionCostWarning:
    def test_panel_signal_with_zero_cost_is_warning(self, registry):
        spec = _spec(signal="top_n", universe="core_etfs", transaction_cost_bps=0.0)
        result = check_compatibility(spec, registry)
        assert any(i.rule_id == "COMPAT_006" for i in result.warnings)

    def test_panel_signal_with_nonzero_cost_no_warning(self, registry):
        spec = _spec(signal="top_n", universe="core_etfs", transaction_cost_bps=10.0)
        result = check_compatibility(spec, registry)
        compat_006 = [i for i in result.warnings if i.rule_id == "COMPAT_006"]
        assert not compat_006

    def test_single_asset_signal_zero_cost_no_warning(self, registry):
        spec = _spec(signal="sign", universe="single_asset_spy", transaction_cost_bps=0.0)
        result = check_compatibility(spec, registry)
        compat_006 = [i for i in result.warnings if i.rule_id == "COMPAT_006"]
        assert not compat_006


class TestCompatibilityResultStructure:
    def test_empty_spec_returns_valid_result(self, registry):
        spec = _spec()
        result = check_compatibility(spec, registry)
        assert isinstance(result, CompatibilityResult)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.info, list)

    def test_is_valid_true_when_no_errors(self, registry):
        spec = _spec(signal="sign", universe="single_asset_spy")
        result = check_compatibility(spec, registry)
        assert result.is_valid

    def test_warnings_do_not_affect_is_valid(self, registry):
        spec = _spec(features=["rolling_beta"])
        result = check_compatibility(spec, registry)
        assert result.is_valid  # warnings only

    def test_all_issues_aggregation(self, registry):
        spec = _spec(
            features=["rolling_beta"],                    # warning
            model="LassoRegression",                     # info
            model_params={"alpha": 1.0},
        )
        result = check_compatibility(spec, registry)
        all_issues = result.all_issues
        assert len(all_issues) == len(result.errors) + len(result.warnings) + len(result.info)

    def test_issue_has_required_fields(self, registry):
        spec = _spec(model="UnknownModel")
        result = check_compatibility(spec, registry)
        for issue in result.errors:
            assert issue.rule_id
            assert issue.severity in ("error", "warning", "info")
            assert issue.message
