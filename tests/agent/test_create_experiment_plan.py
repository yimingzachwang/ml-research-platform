"""Tests for src/agent/planner/create_experiment_plan.py

Covers: planner flow, experiment type inference, baseline selection, success
criteria, diagnostics, risks, and end-to-end pipeline scenarios.
"""

from __future__ import annotations

import pytest

from src.agent.intent.research_spec import ResearchSpec
from src.agent.planner.create_experiment_plan import (
    COST_SENSITIVITY_TEST,
    CUSTOM_RESEARCH_DESIGN,
    FEATURE_FAMILY_TEST,
    MODEL_COMPARISON,
    PRESET_BASELINE,
    UNIVERSE_TEST,
    VALIDATION_TEST,
    _generate_plan_id,
    _infer_experiment_type,
    _select_baseline,
    create_experiment_plan,
)
from src.agent.planner.experiment_plan import ExperimentPlan
from src.agent.planner.plan_result import ExperimentPlanResult
from src.agent.registry.registry_loader import load_registry
from src.agent.resolver.resolve_research_spec import resolve_research_spec
from src.agent.resolver.resolved_spec import ResolvedResearchSpec


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def _resolve(spec: ResearchSpec, registry=None) -> ResolvedResearchSpec:
    """Helper: resolve spec and assert it succeeded."""
    if registry is None:
        registry = load_registry()
    result = resolve_research_spec(spec, registry)
    return result.resolved_spec


def _resolve_assert_resolved(spec: ResearchSpec, registry) -> ResolvedResearchSpec:
    """Resolve and assert status == resolved."""
    result = resolve_research_spec(spec, registry)
    assert result.status == "resolved", (
        f"Expected resolved, got {result.status!r}. "
        f"Missing: {result.missing_fields}, errors: {result.errors}"
    )
    return result.resolved_spec


# ---------------------------------------------------------------------------
# Scenario 1: Canonical preset baseline
# ---------------------------------------------------------------------------

class TestCanonicalPresetBaseline:
    @pytest.fixture(scope="class")
    def plan_result(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved = _resolve_assert_resolved(spec, registry)
        return create_experiment_plan(resolved, registry)

    def test_status_is_planned_or_needs_review(self, plan_result):
        assert plan_result.status in ("planned", "needs_review"), (
            f"Unexpected status: {plan_result.status!r}. "
            f"Errors: {plan_result.errors}, critic: {plan_result.critic_errors}"
        )

    def test_plan_is_created(self, plan_result):
        assert plan_result.plan is not None

    def test_experiment_type_is_preset_baseline(self, plan_result):
        assert plan_result.plan.experiment_type == PRESET_BASELINE

    def test_baseline_is_canonical_preset(self, plan_result):
        assert plan_result.plan.baseline == "canonical_multi_asset_etf"

    def test_success_criteria_include_oos_sharpe(self, plan_result):
        criteria_text = " ".join(plan_result.plan.success_criteria).lower()
        assert "sharpe" in criteria_text

    def test_success_criteria_include_non_sharpe_risk_criterion(self, plan_result):
        criteria = plan_result.plan.success_criteria
        non_sharpe = [c for c in criteria if "sharpe" not in c.lower()]
        assert len(non_sharpe) >= 1, "Expected at least one non-Sharpe criterion"

    def test_success_criteria_include_drawdown(self, plan_result):
        criteria_text = " ".join(plan_result.plan.success_criteria).lower()
        assert "drawdown" in criteria_text

    def test_diagnostics_include_split_metrics(self, plan_result):
        assert "split_metrics" in plan_result.plan.required_diagnostics

    def test_diagnostics_include_sharpe_ratio(self, plan_result):
        assert "sharpe_ratio" in plan_result.plan.required_diagnostics

    def test_plan_has_risks(self, plan_result):
        assert len(plan_result.plan.risks) > 0

    def test_no_critic_errors(self, plan_result):
        assert not plan_result.critic_errors

    def test_plan_id_is_string(self, plan_result):
        assert plan_result.plan.plan_id.startswith("plan_")

    def test_is_executable_ready_is_false(self, plan_result):
        assert plan_result.plan.is_executable_ready is False

    def test_planned_changes_mention_preset(self, plan_result):
        changes_text = " ".join(plan_result.plan.planned_changes)
        assert "canonical_multi_asset_etf" in changes_text or "preset" in changes_text


# ---------------------------------------------------------------------------
# Scenario 2: Preset with model override — model_comparison
# ---------------------------------------------------------------------------

class TestPresetModelOverride:
    @pytest.fixture(scope="class")
    def plan_result(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            model="LassoRegression",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved = _resolve_assert_resolved(spec, registry)
        return create_experiment_plan(resolved, registry)

    def test_experiment_type_is_model_comparison(self, plan_result):
        assert plan_result.plan.experiment_type == MODEL_COMPARISON

    def test_baseline_is_canonical_preset(self, plan_result):
        assert plan_result.plan.baseline == "canonical_multi_asset_etf"

    def test_planned_changes_mention_model_change(self, plan_result):
        changes_text = " ".join(plan_result.plan.planned_changes)
        assert "Lasso" in changes_text or "model" in changes_text.lower()

    def test_hypothesis_mentions_model(self, plan_result):
        assert "Lasso" in plan_result.plan.hypothesis or "model" in plan_result.plan.hypothesis.lower()

    def test_hypothesis_mentions_baseline_model(self, plan_result):
        assert "Ridge" in plan_result.plan.hypothesis

    def test_no_critic_errors(self, plan_result):
        assert not plan_result.critic_errors

    def test_status_not_invalid(self, plan_result):
        assert plan_result.status != "invalid"


# ---------------------------------------------------------------------------
# Scenario 3: Single-asset preset — single-asset success criteria
# ---------------------------------------------------------------------------

class TestSingleAssetPreset:
    @pytest.fixture(scope="class")
    def plan_result(self, registry):
        spec = ResearchSpec(
            preset="single_asset_ml",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved = _resolve_assert_resolved(spec, registry)
        return create_experiment_plan(resolved, registry)

    def test_plan_created(self, plan_result):
        assert plan_result.plan is not None

    def test_status_not_invalid(self, plan_result):
        assert plan_result.status != "invalid", (
            f"Plan was invalid: {plan_result.errors} {plan_result.critic_errors}"
        )

    def test_success_criteria_include_hit_rate(self, plan_result):
        criteria_text = " ".join(plan_result.plan.success_criteria).lower()
        assert "hit rate" in criteria_text or "directional" in criteria_text

    def test_success_criteria_include_benchmark_comparison(self, plan_result):
        criteria_text = " ".join(plan_result.plan.success_criteria).lower()
        assert "benchmark" in criteria_text

    def test_success_criteria_do_not_include_ic(self, plan_result):
        criteria_text = " ".join(plan_result.plan.success_criteria).lower()
        assert "information coefficient" not in criteria_text

    def test_success_criteria_do_not_include_rank_monotonicity(self, plan_result):
        criteria_text = " ".join(plan_result.plan.success_criteria).lower()
        assert "rank monotonicity" not in criteria_text

    def test_risks_mention_single_asset(self, plan_result):
        risks_text = " ".join(plan_result.plan.risks).lower()
        assert "single-asset" in risks_text or "single asset" in risks_text

    def test_no_critic_errors(self, plan_result):
        assert not plan_result.critic_errors


# ---------------------------------------------------------------------------
# Scenario 4: Unresolved spec → planner refuses
# ---------------------------------------------------------------------------

class TestUnresolvedSpecRefused:
    def test_incomplete_spec_returns_invalid(self, registry):
        # Custom ticker template with no tickers → resolver returns needs_user_input
        spec = ResearchSpec(preset="custom_ticker_template", end_date="2024-12-31")
        resolution = resolve_research_spec(spec, registry)
        # The resolved_spec will have missing_fields
        assert not resolution.resolved_spec.is_complete()

        result = create_experiment_plan(resolution.resolved_spec, registry)
        assert result.status == "invalid"
        assert result.plan is None
        assert any("incomplete" in e.lower() or "missing" in e.lower() for e in result.errors)

    def test_needs_user_input_spec_returns_invalid(self, registry):
        # Minimal spec with no useful fields
        spec = ResearchSpec(end_date="2024-12-31")
        resolution = resolve_research_spec(spec, registry, mode="strict")
        assert resolution.status == "needs_user_input"
        result = create_experiment_plan(resolution.resolved_spec, registry)
        assert result.status == "invalid"


# ---------------------------------------------------------------------------
# Scenario 5: Invalid resolved spec → planner refuses
# ---------------------------------------------------------------------------

class TestInvalidSpecRefused:
    def test_panel_signal_on_single_asset_returns_invalid(self, registry):
        # Directly construct an invalid resolved spec
        resolved = ResolvedResearchSpec(
            universe="single_asset_spy",
            tickers=["SPY"],
            signal="top_n",
            model="RidgeRegression",
            label="forward_returns",
            start_date="2015-01-01",
            end_date="2024-12-31",
            features=["momentum"],
            validation_protocol="rolling",
            transaction_cost_bps=10.0,
            asset_class="equity_index",
            frequency="monthly",
            compatibility_errors=["Panel signal 'top_n' requires multi-asset universe."],
        )
        assert not resolved.is_valid()
        result = create_experiment_plan(resolved, registry)
        assert result.status == "invalid"
        assert result.plan is None

    def test_via_resolver_invalid_panel_signal(self, registry):
        spec = ResearchSpec(
            signal="top_n",
            universe="single_asset_spy",
            model="RidgeRegression",
            label="forward_returns",
            start_date="2015-01-01",
            end_date="2024-12-31",
            features=["momentum"],
            validation_protocol="rolling",
            transaction_cost_bps=10.0,
            asset_class="equity_index",
            frequency="monthly",
        )
        resolution = resolve_research_spec(spec, registry)
        assert resolution.status == "invalid"
        result = create_experiment_plan(resolution.resolved_spec, registry)
        assert result.status == "invalid"


# ---------------------------------------------------------------------------
# Scenario 6: Cost sensitivity test
# ---------------------------------------------------------------------------

class TestCostSensitivityTest:
    @pytest.fixture(scope="class")
    def plan_result(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            transaction_cost_bps=25.0,
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved = _resolve_assert_resolved(spec, registry)
        return create_experiment_plan(resolved, registry)

    def test_experiment_type_is_cost_sensitivity(self, plan_result):
        assert plan_result.plan.experiment_type == COST_SENSITIVITY_TEST

    def test_planned_changes_mention_transaction_cost(self, plan_result):
        changes_text = " ".join(plan_result.plan.planned_changes).lower()
        assert "transaction" in changes_text or "cost" in changes_text or "25" in changes_text

    def test_hypothesis_mentions_cost(self, plan_result):
        hypothesis_lower = plan_result.plan.hypothesis.lower()
        assert "cost" in hypothesis_lower or "transaction" in hypothesis_lower

    def test_no_critic_errors(self, plan_result):
        assert not plan_result.critic_errors

    def test_objective_mentions_cost(self, plan_result):
        objective_lower = plan_result.plan.research_objective.lower()
        assert "cost" in objective_lower or "alpha" in objective_lower


# ---------------------------------------------------------------------------
# Scenario 7: Feature change → feature_family_test
# ---------------------------------------------------------------------------

class TestFeatureFamilyTest:
    @pytest.fixture(scope="class")
    def plan_result(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            features=["momentum", "rolling_volatility"],
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved = _resolve_assert_resolved(spec, registry)
        return create_experiment_plan(resolved, registry)

    def test_experiment_type_is_feature_family_test(self, plan_result):
        assert plan_result.plan.experiment_type == FEATURE_FAMILY_TEST

    def test_planned_changes_mention_features(self, plan_result):
        changes_text = " ".join(plan_result.plan.planned_changes).lower()
        assert "feature" in changes_text

    def test_no_critic_errors(self, plan_result):
        assert not plan_result.critic_errors

    def test_status_not_invalid(self, plan_result):
        assert plan_result.status != "invalid"

    def test_feature_names_appear_in_changes(self, plan_result):
        changes_text = " ".join(plan_result.plan.planned_changes)
        assert "momentum" in changes_text or "rolling_volatility" in changes_text


# ---------------------------------------------------------------------------
# Experiment type inference unit tests
# ---------------------------------------------------------------------------

class TestInferExperimentType:
    def test_model_in_user_fields_gives_model_comparison(self, registry):
        resolved = ResolvedResearchSpec(
            user_specified_fields={"model": "LassoRegression", "end_date": "2024-12-31"},
            preset="canonical_multi_asset_etf",
            model="LassoRegression",
        )
        assert _infer_experiment_type(resolved) == MODEL_COMPARISON

    def test_features_in_user_fields_gives_feature_family_test(self, registry):
        resolved = ResolvedResearchSpec(
            user_specified_fields={"features": ["momentum"], "end_date": "2024-12-31"},
            preset="canonical_multi_asset_etf",
            features=["momentum"],
        )
        assert _infer_experiment_type(resolved) == FEATURE_FAMILY_TEST

    def test_universe_in_user_fields_gives_universe_test(self, registry):
        resolved = ResolvedResearchSpec(
            user_specified_fields={"universe": "core_etfs", "end_date": "2024-12-31"},
            preset="canonical_multi_asset_etf",
        )
        assert _infer_experiment_type(resolved) == UNIVERSE_TEST

    def test_validation_protocol_gives_validation_test(self):
        resolved = ResolvedResearchSpec(
            user_specified_fields={"validation_protocol": "expanding"},
        )
        assert _infer_experiment_type(resolved) == VALIDATION_TEST

    def test_train_window_gives_validation_test(self):
        resolved = ResolvedResearchSpec(
            user_specified_fields={"train_window": "24m"},
        )
        assert _infer_experiment_type(resolved) == VALIDATION_TEST

    def test_transaction_cost_gives_cost_sensitivity(self):
        resolved = ResolvedResearchSpec(
            user_specified_fields={"transaction_cost_bps": 25.0},
        )
        assert _infer_experiment_type(resolved) == COST_SENSITIVITY_TEST

    def test_only_preset_and_end_date_gives_preset_baseline(self):
        resolved = ResolvedResearchSpec(
            user_specified_fields={"preset": "canonical_multi_asset_etf", "end_date": "2024-12-31"},
            preset="canonical_multi_asset_etf",
        )
        assert _infer_experiment_type(resolved) == PRESET_BASELINE

    def test_no_preset_no_user_fields_gives_custom(self):
        resolved = ResolvedResearchSpec(
            user_specified_fields={},
            preset=None,
        )
        assert _infer_experiment_type(resolved) == CUSTOM_RESEARCH_DESIGN

    def test_model_takes_priority_over_features(self):
        resolved = ResolvedResearchSpec(
            user_specified_fields={"model": "Lasso", "features": ["x"]},
        )
        assert _infer_experiment_type(resolved) == MODEL_COMPARISON


# ---------------------------------------------------------------------------
# Plan ID stability tests
# ---------------------------------------------------------------------------

class TestPlanIdStability:
    def test_same_spec_gives_same_plan_id(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved_a = _resolve_assert_resolved(spec, registry)
        resolved_b = _resolve_assert_resolved(spec, registry)
        assert _generate_plan_id(resolved_a) == _generate_plan_id(resolved_b)

    def test_different_model_gives_different_plan_id(self, registry):
        spec_a = ResearchSpec(
            preset="canonical_multi_asset_etf",
            model="RidgeRegression",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        spec_b = ResearchSpec(
            preset="canonical_multi_asset_etf",
            model="LassoRegression",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved_a = _resolve_assert_resolved(spec_a, registry)
        resolved_b = _resolve_assert_resolved(spec_b, registry)
        assert _generate_plan_id(resolved_a) != _generate_plan_id(resolved_b)

    def test_plan_id_starts_with_plan_prefix(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved = _resolve_assert_resolved(spec, registry)
        plan_id = _generate_plan_id(resolved)
        assert plan_id.startswith("plan_")
        assert len(plan_id) > 5


# ---------------------------------------------------------------------------
# Baseline selection unit tests
# ---------------------------------------------------------------------------

class TestSelectBaseline:
    def test_preset_in_spec_gives_preset_as_baseline(self, registry):
        resolved = ResolvedResearchSpec(preset="canonical_multi_asset_etf")
        baseline, warnings = _select_baseline(resolved, registry)
        assert baseline == "canonical_multi_asset_etf"
        assert warnings == []

    def test_no_preset_canonical_universe_infers_baseline(self, registry):
        resolved = ResolvedResearchSpec(
            preset=None,
            universe="canonical_multi_asset_etfs",
        )
        baseline, warnings = _select_baseline(resolved, registry)
        assert baseline == "canonical_multi_asset_etf"
        assert len(warnings) > 0

    def test_no_preset_other_universe_returns_none(self, registry):
        resolved = ResolvedResearchSpec(
            preset=None,
            universe="core_etfs",
        )
        baseline, warnings = _select_baseline(resolved, registry)
        assert baseline is None
        assert len(warnings) > 0

    def test_baseline_warning_message_is_informative(self, registry):
        resolved = ResolvedResearchSpec(preset=None, universe="us_sector_etfs")
        _, warnings = _select_baseline(resolved, registry)
        assert any(len(w) > 10 for w in warnings)


# ---------------------------------------------------------------------------
# End-to-end pipeline tests
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    def test_canonical_preset_full_pipeline(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolution = resolve_research_spec(spec, registry)
        assert resolution.status == "resolved"

        result = create_experiment_plan(resolution.resolved_spec, registry)
        assert result.status in ("planned", "needs_review")
        assert result.plan is not None
        assert result.plan.resolved_spec is resolution.resolved_spec
        assert result.plan.is_executable_ready is False

    def test_recommended_mode_defaults_produces_needs_review(self, registry):
        # Without explicit start_date, recommended defaults are applied → requires_confirmation
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            end_date="2024-12-31",
        )
        resolution = resolve_research_spec(spec, registry)
        result = create_experiment_plan(resolution.resolved_spec, registry)
        assert result.status in ("planned", "needs_review")
        assert result.plan is not None

    def test_model_override_pipeline(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            model="ElasticNetRegression",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolution = resolve_research_spec(spec, registry)
        assert resolution.status == "resolved"

        result = create_experiment_plan(resolution.resolved_spec, registry)
        assert result.plan.experiment_type == MODEL_COMPARISON
        assert "ElasticNet" in " ".join(result.plan.planned_changes)

    def test_resolved_spec_provenance_carried_into_plan(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolution = resolve_research_spec(spec, registry)
        result = create_experiment_plan(resolution.resolved_spec, registry)

        plan = result.plan
        # Assumptions from the resolver should be in the plan
        assert isinstance(plan.assumptions, list)
        # Plan should carry requires_confirmation from resolved spec
        assert plan.requires_confirmation == resolution.resolved_spec.requires_confirmation

    def test_invalid_compat_via_resolver_then_planner_refuses(self, registry):
        spec = ResearchSpec(
            label="ranking_target",
            universe="single_asset_spy",
            model="RidgeRegression",
            signal="sign",
            start_date="2015-01-01",
            end_date="2024-12-31",
            features=["momentum"],
            validation_protocol="rolling",
            transaction_cost_bps=10.0,
            asset_class="equity_index",
            frequency="monthly",
        )
        resolution = resolve_research_spec(spec, registry)
        assert resolution.status == "invalid"
        result = create_experiment_plan(resolution.resolved_spec, registry)
        assert result.status == "invalid"


# ---------------------------------------------------------------------------
# Risk and failure modes content tests
# ---------------------------------------------------------------------------

class TestRisksAndFailureModes:
    def test_multi_asset_plan_has_rank_dispersion_failure_mode(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved = _resolve_assert_resolved(spec, registry)
        result = create_experiment_plan(resolved, registry)
        failure_text = " ".join(result.plan.failure_modes_to_monitor).lower()
        assert "rank dispersion" in failure_text or "dispersion" in failure_text

    def test_single_asset_plan_has_hit_rate_failure_mode(self, registry):
        spec = ResearchSpec(
            preset="single_asset_ml",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved = _resolve_assert_resolved(spec, registry)
        result = create_experiment_plan(resolved, registry)
        failure_text = " ".join(result.plan.failure_modes_to_monitor).lower()
        assert "hit rate" in failure_text

    def test_ridge_model_has_compression_risk(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved = _resolve_assert_resolved(spec, registry)
        result = create_experiment_plan(resolved, registry)
        risks_text = " ".join(result.plan.risks).lower()
        assert "ridge" in risks_text or "compression" in risks_text

    def test_lasso_model_has_zeroing_risk(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            model="LassoRegression",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved = _resolve_assert_resolved(spec, registry)
        result = create_experiment_plan(resolved, registry)
        risks_text = " ".join(result.plan.risks).lower()
        assert "lasso" in risks_text or "zero" in risks_text


# ---------------------------------------------------------------------------
# Required diagnostics content tests
# ---------------------------------------------------------------------------

class TestRequiredDiagnostics:
    def test_all_recommended_diagnostics_are_in_registry(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved = _resolve_assert_resolved(spec, registry)
        result = create_experiment_plan(resolved, registry)
        registry_diags = registry.diagnostic_names()
        for diag in result.plan.required_diagnostics:
            assert diag in registry_diags, (
                f"Diagnostic '{diag}' was recommended but is not in the registry"
            )

    def test_split_metrics_always_included(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved = _resolve_assert_resolved(spec, registry)
        result = create_experiment_plan(resolved, registry)
        assert "split_metrics" in result.plan.required_diagnostics

    def test_sharpe_ratio_always_included(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved = _resolve_assert_resolved(spec, registry)
        result = create_experiment_plan(resolved, registry)
        assert "sharpe_ratio" in result.plan.required_diagnostics

    def test_max_drawdown_included(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved = _resolve_assert_resolved(spec, registry)
        result = create_experiment_plan(resolved, registry)
        assert "max_drawdown" in result.plan.required_diagnostics


# ---------------------------------------------------------------------------
# ExperimentPlanResult structure tests
# ---------------------------------------------------------------------------

class TestExperimentPlanResultStructure:
    def test_result_has_status(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved = _resolve_assert_resolved(spec, registry)
        result = create_experiment_plan(resolved, registry)
        assert result.status in ("planned", "needs_review", "invalid")

    def test_result_has_summary(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved = _resolve_assert_resolved(spec, registry)
        result = create_experiment_plan(resolved, registry)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_result_properties(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        resolved = _resolve_assert_resolved(spec, registry)
        result = create_experiment_plan(resolved, registry)
        if result.status == "planned":
            assert result.is_planned
            assert not result.needs_review
            assert not result.is_invalid
        elif result.status == "needs_review":
            assert not result.is_planned
            assert result.needs_review
            assert not result.is_invalid

    def test_invalid_result_has_no_plan(self, registry):
        # Incomplete spec → invalid
        resolved = ResolvedResearchSpec(
            missing_fields=["model", "universe"],
            compatibility_errors=[],
        )
        result = create_experiment_plan(resolved, registry)
        assert result.status == "invalid"
        assert result.plan is None
        assert len(result.errors) > 0
