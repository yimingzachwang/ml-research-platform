"""Tests for src/agent/planner/plan_critic.py

Covers all 11 critic checks (PLAN_A through PLAN_K) including:
- Plan must have objective (PLAN_A)
- Plan must have hypothesis (PLAN_B)
- Plan must have success criteria (PLAN_C)
- No Sharpe-only evaluation (PLAN_D)
- Plan must have risks (PLAN_E)
- Plan must have diagnostics (PLAN_F)
- Comparison experiments need baseline (PLAN_G)
- Model comparison: selected ≠ baseline model (PLAN_H)
- Feature test changes mention features (PLAN_I)
- Compat warnings forwarded (PLAN_J)
- Confirmation consistency (PLAN_K)
"""

from __future__ import annotations

import pytest

from src.agent.intent.research_spec import ResearchSpec
from src.agent.planner.create_experiment_plan import (
    MODEL_COMPARISON,
    PRESET_BASELINE,
    FEATURE_FAMILY_TEST,
    create_experiment_plan,
)
from src.agent.planner.experiment_plan import ExperimentPlan
from src.agent.planner.plan_critic import (
    PlanCriticIssue,
    PlanCriticResult,
    critique_experiment_plan,
)
from src.agent.registry.registry_loader import load_registry
from src.agent.resolver.resolve_research_spec import resolve_research_spec
from src.agent.resolver.resolved_spec import ResolvedResearchSpec


@pytest.fixture(scope="module")
def registry():
    return load_registry()


@pytest.fixture(scope="module")
def canonical_resolved(registry):
    spec = ResearchSpec(
        preset="canonical_multi_asset_etf",
        start_date="2015-01-01",
        end_date="2024-12-31",
    )
    result = resolve_research_spec(spec, registry)
    assert result.status == "resolved"
    return result.resolved_spec


def _make_plan(
    resolved_spec: ResolvedResearchSpec,
    **overrides,
) -> ExperimentPlan:
    """Build a valid minimal ExperimentPlan with overrides applied."""
    defaults = dict(
        plan_id="plan_test",
        name="test plan",
        description="test",
        resolved_spec=resolved_spec,
        research_objective="Test objective.",
        hypothesis="Test hypothesis.",
        experiment_type=PRESET_BASELINE,
        baseline="canonical_multi_asset_etf",
        comparison_policy="No comparison.",
        planned_changes=["Running preset as-is."],
        controlled_variables=[],
        success_criteria=["OOS Sharpe is positive", "Drawdown is acceptable"],
        primary_metric="OOS Sharpe ratio",
        secondary_metrics=["Maximum drawdown"],
        required_diagnostics=["split_metrics", "sharpe_ratio"],
        expected_evidence=["Split metrics per fold"],
        risks=["Feature effects may be regime-dependent."],
        failure_modes_to_monitor=["OOS Sharpe near zero."],
        assumptions=[],
        warnings=[],
        requires_confirmation=False,
        is_executable_ready=False,
    )
    defaults.update(overrides)
    return ExperimentPlan(**defaults)


# ---------------------------------------------------------------------------
# Result structure tests
# ---------------------------------------------------------------------------

class TestPlanCriticResultStructure:
    def test_returns_plan_critic_result(self, canonical_resolved, registry):
        plan = _make_plan(canonical_resolved)
        result = critique_experiment_plan(plan, registry)
        assert isinstance(result, PlanCriticResult)

    def test_valid_plan_returns_no_errors(self, canonical_resolved, registry):
        plan = _make_plan(canonical_resolved)
        result = critique_experiment_plan(plan, registry)
        assert result.is_valid, f"Unexpected errors: {result.errors}"

    def test_result_has_errors_warnings_info_lists(self, canonical_resolved, registry):
        plan = _make_plan(canonical_resolved)
        result = critique_experiment_plan(plan, registry)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.info, list)

    def test_all_issues_aggregates_all_severities(self, canonical_resolved, registry):
        plan = _make_plan(canonical_resolved)
        result = critique_experiment_plan(plan, registry)
        all_count = len(result.errors) + len(result.warnings) + len(result.info)
        assert len(result.all_issues) == all_count

    def test_issue_has_required_fields(self, canonical_resolved, registry):
        plan = _make_plan(canonical_resolved, research_objective="")
        result = critique_experiment_plan(plan, registry)
        for issue in result.errors:
            assert isinstance(issue, PlanCriticIssue)
            assert issue.check_id
            assert issue.severity in ("error", "warning", "info")
            assert issue.message


# ---------------------------------------------------------------------------
# PLAN_A: Research objective
# ---------------------------------------------------------------------------

class TestCheckA:
    def test_missing_objective_is_error(self, canonical_resolved, registry):
        plan = _make_plan(canonical_resolved, research_objective="")
        result = critique_experiment_plan(plan, registry)
        assert any(i.check_id == "PLAN_A" for i in result.errors)
        assert not result.is_valid

    def test_whitespace_only_objective_is_error(self, canonical_resolved, registry):
        plan = _make_plan(canonical_resolved, research_objective="   ")
        result = critique_experiment_plan(plan, registry)
        assert any(i.check_id == "PLAN_A" for i in result.errors)

    def test_non_empty_objective_passes(self, canonical_resolved, registry):
        plan = _make_plan(canonical_resolved, research_objective="Evaluate model X.")
        result = critique_experiment_plan(plan, registry)
        plan_a_errors = [i for i in result.errors if i.check_id == "PLAN_A"]
        assert not plan_a_errors


# ---------------------------------------------------------------------------
# PLAN_B: Hypothesis
# ---------------------------------------------------------------------------

class TestCheckB:
    def test_missing_hypothesis_is_error(self, canonical_resolved, registry):
        plan = _make_plan(canonical_resolved, hypothesis="")
        result = critique_experiment_plan(plan, registry)
        assert any(i.check_id == "PLAN_B" for i in result.errors)
        assert not result.is_valid

    def test_whitespace_only_hypothesis_is_error(self, canonical_resolved, registry):
        plan = _make_plan(canonical_resolved, hypothesis="  ")
        result = critique_experiment_plan(plan, registry)
        assert any(i.check_id == "PLAN_B" for i in result.errors)

    def test_valid_hypothesis_passes(self, canonical_resolved, registry):
        plan = _make_plan(canonical_resolved, hypothesis="Model X improves ranking.")
        result = critique_experiment_plan(plan, registry)
        plan_b_errors = [i for i in result.errors if i.check_id == "PLAN_B"]
        assert not plan_b_errors


# ---------------------------------------------------------------------------
# PLAN_C: Success criteria
# ---------------------------------------------------------------------------

class TestCheckC:
    def test_empty_success_criteria_is_error(self, canonical_resolved, registry):
        plan = _make_plan(canonical_resolved, success_criteria=[])
        result = critique_experiment_plan(plan, registry)
        assert any(i.check_id == "PLAN_C" for i in result.errors)

    def test_one_criterion_passes(self, canonical_resolved, registry):
        plan = _make_plan(
            canonical_resolved,
            success_criteria=["OOS Sharpe is positive after transaction costs"],
        )
        result = critique_experiment_plan(plan, registry)
        plan_c_errors = [i for i in result.errors if i.check_id == "PLAN_C"]
        assert not plan_c_errors


# ---------------------------------------------------------------------------
# PLAN_D: No Sharpe-only evaluation
# ---------------------------------------------------------------------------

class TestCheckD:
    def test_sharpe_only_criteria_is_error(self, canonical_resolved, registry):
        plan = _make_plan(
            canonical_resolved,
            success_criteria=["OOS Sharpe is positive", "Mean fold Sharpe is stable"],
        )
        result = critique_experiment_plan(plan, registry)
        assert any(i.check_id == "PLAN_D" for i in result.errors)
        assert not result.is_valid

    def test_mixed_criteria_passes(self, canonical_resolved, registry):
        plan = _make_plan(
            canonical_resolved,
            success_criteria=[
                "OOS Sharpe is positive",
                "Maximum drawdown is acceptable",
            ],
        )
        result = critique_experiment_plan(plan, registry)
        plan_d_errors = [i for i in result.errors if i.check_id == "PLAN_D"]
        assert not plan_d_errors

    def test_empty_criteria_does_not_trigger_sharpe_check(self, canonical_resolved, registry):
        plan = _make_plan(canonical_resolved, success_criteria=[])
        result = critique_experiment_plan(plan, registry)
        # PLAN_C fires, but PLAN_D should not (no criteria to evaluate)
        plan_d_errors = [i for i in result.errors if i.check_id == "PLAN_D"]
        assert not plan_d_errors

    def test_sharpe_in_all_criteria_is_error(self, canonical_resolved, registry):
        plan = _make_plan(
            canonical_resolved,
            success_criteria=["Annualised Sharpe exceeds 1.0"],
        )
        result = critique_experiment_plan(plan, registry)
        assert any(i.check_id == "PLAN_D" for i in result.errors)


# ---------------------------------------------------------------------------
# PLAN_E: Risks required
# ---------------------------------------------------------------------------

class TestCheckE:
    def test_no_risks_and_no_failure_modes_is_error(self, canonical_resolved, registry):
        plan = _make_plan(
            canonical_resolved,
            risks=[],
            failure_modes_to_monitor=[],
        )
        result = critique_experiment_plan(plan, registry)
        assert any(i.check_id == "PLAN_E" for i in result.errors)

    def test_risks_only_passes(self, canonical_resolved, registry):
        plan = _make_plan(canonical_resolved, risks=["Feature regime risk."], failure_modes_to_monitor=[])
        result = critique_experiment_plan(plan, registry)
        plan_e_errors = [i for i in result.errors if i.check_id == "PLAN_E"]
        assert not plan_e_errors

    def test_failure_modes_only_passes(self, canonical_resolved, registry):
        plan = _make_plan(canonical_resolved, risks=[], failure_modes_to_monitor=["IC goes negative."])
        result = critique_experiment_plan(plan, registry)
        plan_e_errors = [i for i in result.errors if i.check_id == "PLAN_E"]
        assert not plan_e_errors


# ---------------------------------------------------------------------------
# PLAN_F: Diagnostics required
# ---------------------------------------------------------------------------

class TestCheckF:
    def test_empty_diagnostics_is_error(self, canonical_resolved, registry):
        plan = _make_plan(canonical_resolved, required_diagnostics=[])
        result = critique_experiment_plan(plan, registry)
        assert any(i.check_id == "PLAN_F" for i in result.errors)

    def test_one_diagnostic_passes(self, canonical_resolved, registry):
        plan = _make_plan(canonical_resolved, required_diagnostics=["split_metrics"])
        result = critique_experiment_plan(plan, registry)
        plan_f_errors = [i for i in result.errors if i.check_id == "PLAN_F"]
        assert not plan_f_errors


# ---------------------------------------------------------------------------
# PLAN_G: Baseline required for comparison experiments
# ---------------------------------------------------------------------------

class TestCheckG:
    def test_model_comparison_without_baseline_is_warning(self, canonical_resolved, registry):
        plan = _make_plan(
            canonical_resolved,
            experiment_type=MODEL_COMPARISON,
            baseline=None,
        )
        result = critique_experiment_plan(plan, registry)
        assert any(i.check_id == "PLAN_G" for i in result.warnings)

    def test_model_comparison_with_baseline_no_warning(self, canonical_resolved, registry):
        plan = _make_plan(
            canonical_resolved,
            experiment_type=MODEL_COMPARISON,
            baseline="canonical_multi_asset_etf",
        )
        result = critique_experiment_plan(plan, registry)
        plan_g_warnings = [i for i in result.warnings if i.check_id == "PLAN_G"]
        assert not plan_g_warnings

    def test_preset_baseline_without_baseline_no_warning(self, canonical_resolved, registry):
        plan = _make_plan(
            canonical_resolved,
            experiment_type=PRESET_BASELINE,
            baseline=None,  # No comparison needed for preset_baseline
        )
        result = critique_experiment_plan(plan, registry)
        plan_g_warnings = [i for i in result.warnings if i.check_id == "PLAN_G"]
        assert not plan_g_warnings

    def test_feature_family_test_without_baseline_is_warning(self, canonical_resolved, registry):
        plan = _make_plan(
            canonical_resolved,
            experiment_type=FEATURE_FAMILY_TEST,
            baseline=None,
        )
        result = critique_experiment_plan(plan, registry)
        assert any(i.check_id == "PLAN_G" for i in result.warnings)


# ---------------------------------------------------------------------------
# PLAN_H: Model comparison — selected model must differ from baseline model
# ---------------------------------------------------------------------------

class TestCheckH:
    def test_same_model_as_baseline_is_warning(self, registry):
        # canonical_multi_asset_etf preset uses RidgeRegression
        resolved = ResolvedResearchSpec(
            preset="canonical_multi_asset_etf",
            model="RidgeRegression",
            user_specified_fields={"model": "RidgeRegression"},
        )
        plan = _make_plan(
            resolved,
            experiment_type=MODEL_COMPARISON,
            baseline="canonical_multi_asset_etf",
        )
        result = critique_experiment_plan(plan, registry)
        assert any(i.check_id == "PLAN_H" for i in result.warnings)

    def test_different_model_from_baseline_no_warning(self, canonical_resolved, registry):
        # canonical preset uses RidgeRegression; we set LassoRegression
        resolved = ResolvedResearchSpec(
            preset="canonical_multi_asset_etf",
            model="LassoRegression",
            user_specified_fields={"model": "LassoRegression"},
        )
        plan = _make_plan(
            resolved,
            experiment_type=MODEL_COMPARISON,
            baseline="canonical_multi_asset_etf",
        )
        result = critique_experiment_plan(plan, registry)
        plan_h_warnings = [i for i in result.warnings if i.check_id == "PLAN_H"]
        assert not plan_h_warnings

    def test_check_h_skipped_when_not_model_comparison(self, canonical_resolved, registry):
        # Even if same model, PLAN_H shouldn't fire for non-model-comparison types
        plan = _make_plan(
            canonical_resolved,
            experiment_type=PRESET_BASELINE,
            baseline="canonical_multi_asset_etf",
        )
        result = critique_experiment_plan(plan, registry)
        plan_h_warnings = [i for i in result.warnings if i.check_id == "PLAN_H"]
        assert not plan_h_warnings

    def test_check_h_skipped_when_baseline_is_none(self, registry):
        resolved = ResolvedResearchSpec(
            model="RidgeRegression",
            user_specified_fields={"model": "RidgeRegression"},
        )
        plan = _make_plan(
            resolved,
            experiment_type=MODEL_COMPARISON,
            baseline=None,
        )
        result = critique_experiment_plan(plan, registry)
        plan_h_warnings = [i for i in result.warnings if i.check_id == "PLAN_H"]
        assert not plan_h_warnings  # PLAN_G fires instead


# ---------------------------------------------------------------------------
# PLAN_I: Feature test changes must mention features
# ---------------------------------------------------------------------------

class TestCheckI:
    def test_feature_test_without_feature_mention_in_changes_is_warning(
        self, canonical_resolved, registry
    ):
        plan = _make_plan(
            canonical_resolved,
            experiment_type=FEATURE_FAMILY_TEST,
            planned_changes=["Running experiment with baseline settings."],
        )
        result = critique_experiment_plan(plan, registry)
        assert any(i.check_id == "PLAN_I" for i in result.warnings)

    def test_feature_test_with_feature_mention_no_warning(
        self, canonical_resolved, registry
    ):
        plan = _make_plan(
            canonical_resolved,
            experiment_type=FEATURE_FAMILY_TEST,
            planned_changes=["Feature set changed to ['momentum', 'trend_strength']."],
        )
        result = critique_experiment_plan(plan, registry)
        plan_i_warnings = [i for i in result.warnings if i.check_id == "PLAN_I"]
        assert not plan_i_warnings

    def test_check_i_skipped_for_non_feature_test(self, canonical_resolved, registry):
        plan = _make_plan(
            canonical_resolved,
            experiment_type=PRESET_BASELINE,
            planned_changes=["Running preset as-is."],
        )
        result = critique_experiment_plan(plan, registry)
        plan_i_warnings = [i for i in result.warnings if i.check_id == "PLAN_I"]
        assert not plan_i_warnings


# ---------------------------------------------------------------------------
# PLAN_J: Compatibility warnings forwarded as info
# ---------------------------------------------------------------------------

class TestCheckJ:
    def test_resolver_warnings_forwarded_as_info(self, registry):
        # Build a resolved spec with warnings (rolling_beta without market_ticker)
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            features=["momentum", "rolling_beta"],  # rolling_beta triggers warning
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        result_resolution = resolve_research_spec(spec, registry)
        resolved = result_resolution.resolved_spec

        plan = _make_plan(resolved)
        result = critique_experiment_plan(plan, registry)

        # If resolver produced warnings, they should appear in info
        if resolved.warnings:
            plan_j_info = [i for i in result.info if i.check_id == "PLAN_J"]
            assert len(plan_j_info) == len(resolved.warnings)

    def test_no_resolver_warnings_no_plan_j_issues(self, canonical_resolved, registry):
        # canonical_resolved from simple preset should have no resolver warnings
        # (unless COMPAT rules fire — but canonical_multi_asset_etf is clean)
        if not canonical_resolved.warnings:
            plan = _make_plan(canonical_resolved)
            result = critique_experiment_plan(plan, registry)
            plan_j_info = [i for i in result.info if i.check_id == "PLAN_J"]
            assert not plan_j_info

    def test_plan_j_produces_info_not_error(self, registry):
        resolved = ResolvedResearchSpec(
            warnings=["Test resolver warning."],
        )
        plan = _make_plan(resolved)
        result = critique_experiment_plan(plan, registry)
        plan_j = [i for i in result.info if i.check_id == "PLAN_J"]
        assert all(i.severity == "info" for i in plan_j)


# ---------------------------------------------------------------------------
# PLAN_K: Confirmation consistency
# ---------------------------------------------------------------------------

class TestCheckK:
    def test_spec_requires_confirmation_but_plan_does_not_is_warning(self, registry):
        resolved = ResolvedResearchSpec(requires_confirmation=True)
        plan = _make_plan(resolved, requires_confirmation=False)
        result = critique_experiment_plan(plan, registry)
        assert any(i.check_id == "PLAN_K" for i in result.warnings)

    def test_both_require_confirmation_no_warning(self, registry):
        resolved = ResolvedResearchSpec(requires_confirmation=True)
        plan = _make_plan(resolved, requires_confirmation=True)
        result = critique_experiment_plan(plan, registry)
        plan_k_warnings = [i for i in result.warnings if i.check_id == "PLAN_K"]
        assert not plan_k_warnings

    def test_neither_requires_confirmation_no_warning(self, registry):
        resolved = ResolvedResearchSpec(requires_confirmation=False)
        plan = _make_plan(resolved, requires_confirmation=False)
        result = critique_experiment_plan(plan, registry)
        plan_k_warnings = [i for i in result.warnings if i.check_id == "PLAN_K"]
        assert not plan_k_warnings

    def test_plan_requires_confirmation_but_spec_does_not_no_warning(self, registry):
        resolved = ResolvedResearchSpec(requires_confirmation=False)
        plan = _make_plan(resolved, requires_confirmation=True)
        result = critique_experiment_plan(plan, registry)
        plan_k_warnings = [i for i in result.warnings if i.check_id == "PLAN_K"]
        assert not plan_k_warnings


# ---------------------------------------------------------------------------
# Integration: critique via create_experiment_plan
# ---------------------------------------------------------------------------

class TestCriticViaPlanner:
    def test_canonical_preset_plan_passes_critic(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        result_resolution = resolve_research_spec(spec, registry)
        assert result_resolution.status == "resolved"

        plan_result = create_experiment_plan(result_resolution.resolved_spec, registry)
        assert not plan_result.critic_errors, (
            f"Critic errors: {plan_result.critic_errors}"
        )

    def test_single_asset_plan_passes_critic(self, registry):
        spec = ResearchSpec(
            preset="single_asset_ml",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        result_resolution = resolve_research_spec(spec, registry)
        assert result_resolution.status == "resolved"

        plan_result = create_experiment_plan(result_resolution.resolved_spec, registry)
        assert not plan_result.critic_errors, (
            f"Critic errors: {plan_result.critic_errors}"
        )

    def test_model_override_plan_passes_critic(self, registry):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            model="LassoRegression",
            start_date="2015-01-01",
            end_date="2024-12-31",
        )
        result_resolution = resolve_research_spec(spec, registry)
        plan_result = create_experiment_plan(result_resolution.resolved_spec, registry)
        assert not plan_result.critic_errors, (
            f"Critic errors: {plan_result.critic_errors}"
        )

    def test_critic_issues_are_plan_critic_issues(self, registry):
        resolved = ResolvedResearchSpec(
            requires_confirmation=True,
            warnings=["Some resolver warning."],
        )
        plan = _make_plan(resolved, requires_confirmation=False)
        result = critique_experiment_plan(plan, registry)
        for issue in result.all_issues:
            assert isinstance(issue, PlanCriticIssue)
