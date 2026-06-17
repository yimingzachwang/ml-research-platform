"""Tests for src/agent/planner/experiment_plan.py

Covers: ExperimentPlan dataclass construction, to_dict, and
        summarise_experiment_plan helper.
"""

from __future__ import annotations

import pytest

from src.agent.intent.research_spec import ResearchSpec
from src.agent.planner.experiment_plan import ExperimentPlan, summarise_experiment_plan
from src.agent.registry.registry_loader import load_registry
from src.agent.resolver.resolve_research_spec import resolve_research_spec
from src.agent.resolver.resolved_spec import ResolvedResearchSpec


@pytest.fixture(scope="module")
def registry():
    return load_registry()


@pytest.fixture(scope="module")
def canonical_resolved(registry):
    """Fully resolved spec from canonical_multi_asset_etf preset."""
    spec = ResearchSpec(
        preset="canonical_multi_asset_etf",
        start_date="2015-01-01",
        end_date="2024-12-31",
    )
    result = resolve_research_spec(spec, registry)
    assert result.status == "resolved", f"Unexpected status: {result.status}"
    return result.resolved_spec


@pytest.fixture(scope="module")
def minimal_plan(canonical_resolved):
    """A minimal ExperimentPlan for schema tests."""
    return ExperimentPlan(
        plan_id="plan_test001",
        name="test plan",
        description="minimal plan for schema tests",
        resolved_spec=canonical_resolved,
        research_objective="Establish a baseline.",
        hypothesis="This is the baseline.",
        experiment_type="preset_baseline",
        baseline="canonical_multi_asset_etf",
        comparison_policy="No comparison required.",
        planned_changes=["Running preset as-is."],
        controlled_variables=["model: 'RidgeRegression' (from preset)"],
        success_criteria=["OOS Sharpe is positive", "Drawdown is acceptable"],
        primary_metric="OOS Sharpe ratio",
        secondary_metrics=["Maximum drawdown"],
        required_diagnostics=["split_metrics", "sharpe_ratio"],
        expected_evidence=["Split metrics per fold"],
        risks=["Ridge compression may limit discrimination."],
        failure_modes_to_monitor=["OOS Sharpe near zero."],
        assumptions=["start_date defaulted to 2015-01-01"],
        warnings=[],
        requires_confirmation=False,
        is_executable_ready=False,
    )


# ---------------------------------------------------------------------------
# Schema construction tests
# ---------------------------------------------------------------------------

class TestExperimentPlanConstruction:
    def test_plan_has_plan_id(self, minimal_plan):
        assert minimal_plan.plan_id == "plan_test001"

    def test_plan_has_name(self, minimal_plan):
        assert minimal_plan.name == "test plan"

    def test_plan_has_description(self, minimal_plan):
        assert isinstance(minimal_plan.description, str)

    def test_plan_preserves_resolved_spec(self, minimal_plan, canonical_resolved):
        assert minimal_plan.resolved_spec is canonical_resolved

    def test_plan_has_research_objective(self, minimal_plan):
        assert minimal_plan.research_objective
        assert isinstance(minimal_plan.research_objective, str)

    def test_plan_has_hypothesis(self, minimal_plan):
        assert minimal_plan.hypothesis
        assert isinstance(minimal_plan.hypothesis, str)

    def test_plan_has_experiment_type(self, minimal_plan):
        assert minimal_plan.experiment_type == "preset_baseline"

    def test_plan_has_baseline(self, minimal_plan):
        assert minimal_plan.baseline == "canonical_multi_asset_etf"

    def test_plan_has_success_criteria(self, minimal_plan):
        assert len(minimal_plan.success_criteria) >= 2

    def test_plan_has_diagnostics(self, minimal_plan):
        assert "split_metrics" in minimal_plan.required_diagnostics

    def test_plan_has_risks(self, minimal_plan):
        assert len(minimal_plan.risks) > 0

    def test_is_executable_ready_defaults_false(self, minimal_plan):
        assert minimal_plan.is_executable_ready is False

    def test_requires_confirmation_set_correctly(self, minimal_plan):
        assert minimal_plan.requires_confirmation is False

    def test_created_at_defaults_none(self, minimal_plan):
        assert minimal_plan.created_at is None


class TestExperimentPlanDefaults:
    def test_empty_lists_are_default_factory(self, canonical_resolved):
        plan = ExperimentPlan(
            plan_id="x",
            name="x",
            description="x",
            resolved_spec=canonical_resolved,
            research_objective="x",
            hypothesis="x",
            experiment_type="preset_baseline",
            baseline=None,
            comparison_policy="x",
        )
        assert plan.planned_changes == []
        assert plan.controlled_variables == []
        assert plan.success_criteria == []
        assert plan.secondary_metrics == []
        assert plan.required_diagnostics == []
        assert plan.expected_evidence == []
        assert plan.risks == []
        assert plan.failure_modes_to_monitor == []
        assert plan.assumptions == []
        assert plan.warnings == []

    def test_list_fields_are_independent_instances(self, canonical_resolved):
        plan_a = ExperimentPlan(
            plan_id="a", name="a", description="a",
            resolved_spec=canonical_resolved,
            research_objective="a", hypothesis="a",
            experiment_type="preset_baseline", baseline=None,
            comparison_policy="a",
        )
        plan_b = ExperimentPlan(
            plan_id="b", name="b", description="b",
            resolved_spec=canonical_resolved,
            research_objective="b", hypothesis="b",
            experiment_type="preset_baseline", baseline=None,
            comparison_policy="b",
        )
        plan_a.risks.append("risk A")
        assert plan_b.risks == []


# ---------------------------------------------------------------------------
# Provenance carried forward
# ---------------------------------------------------------------------------

class TestProvenanceCarriedForward:
    def test_assumptions_from_resolved_spec_are_accessible(self, canonical_resolved):
        plan = ExperimentPlan(
            plan_id="x", name="x", description="x",
            resolved_spec=canonical_resolved,
            research_objective="x", hypothesis="x",
            experiment_type="preset_baseline", baseline=None,
            comparison_policy="x",
            assumptions=list(canonical_resolved.assumptions),
        )
        # If the resolver applied defaults, assumptions will be non-empty
        # (start_date defaulted for canonical preset without explicit start_date)
        assert isinstance(plan.assumptions, list)

    def test_requires_confirmation_preserved_from_resolved_spec(
        self, canonical_resolved
    ):
        plan = ExperimentPlan(
            plan_id="x", name="x", description="x",
            resolved_spec=canonical_resolved,
            research_objective="x", hypothesis="x",
            experiment_type="preset_baseline", baseline=None,
            comparison_policy="x",
            requires_confirmation=canonical_resolved.requires_confirmation,
        )
        assert plan.requires_confirmation == canonical_resolved.requires_confirmation

    def test_resolved_spec_user_specified_fields_preserved(self, canonical_resolved):
        assert isinstance(canonical_resolved.user_specified_fields, dict)
        # The plan's resolved_spec still has full provenance
        plan = ExperimentPlan(
            plan_id="x", name="x", description="x",
            resolved_spec=canonical_resolved,
            research_objective="x", hypothesis="x",
            experiment_type="preset_baseline", baseline=None,
            comparison_policy="x",
        )
        assert plan.resolved_spec.user_specified_fields is canonical_resolved.user_specified_fields


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestExperimentPlanToDict:
    def test_to_dict_returns_dict(self, minimal_plan):
        d = minimal_plan.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_required_keys(self, minimal_plan):
        d = minimal_plan.to_dict()
        for key in (
            "plan_id", "name", "description", "resolved_spec",
            "research_objective", "hypothesis", "experiment_type",
            "baseline", "comparison_policy",
            "planned_changes", "controlled_variables",
            "success_criteria", "primary_metric", "secondary_metrics",
            "required_diagnostics", "expected_evidence",
            "risks", "failure_modes_to_monitor",
            "assumptions", "warnings",
            "requires_confirmation", "is_executable_ready", "created_at",
        ):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_resolved_spec_is_dict(self, minimal_plan):
        d = minimal_plan.to_dict()
        assert isinstance(d["resolved_spec"], dict)

    def test_to_dict_list_fields_are_lists(self, minimal_plan):
        d = minimal_plan.to_dict()
        for key in (
            "planned_changes", "controlled_variables", "success_criteria",
            "secondary_metrics", "required_diagnostics", "expected_evidence",
            "risks", "failure_modes_to_monitor", "assumptions", "warnings",
        ):
            assert isinstance(d[key], list), f"{key} should be a list"

    def test_to_dict_plan_id_preserved(self, minimal_plan):
        d = minimal_plan.to_dict()
        assert d["plan_id"] == "plan_test001"

    def test_to_dict_list_mutation_does_not_affect_plan(self, minimal_plan):
        d = minimal_plan.to_dict()
        original_len = len(minimal_plan.risks)
        d["risks"].append("injected risk")
        assert len(minimal_plan.risks) == original_len


# ---------------------------------------------------------------------------
# summarise_experiment_plan
# ---------------------------------------------------------------------------

class TestSummariseExperimentPlan:
    def test_summary_is_string(self, minimal_plan):
        summary = summarise_experiment_plan(minimal_plan)
        assert isinstance(summary, str)
        assert len(summary) > 50

    def test_summary_contains_objective(self, minimal_plan):
        summary = summarise_experiment_plan(minimal_plan)
        assert "Establish a baseline" in summary

    def test_summary_contains_hypothesis(self, minimal_plan):
        summary = summarise_experiment_plan(minimal_plan)
        assert "baseline" in summary.lower()

    def test_summary_contains_experiment_type(self, minimal_plan):
        summary = summarise_experiment_plan(minimal_plan)
        assert "preset_baseline" in summary

    def test_summary_contains_baseline(self, minimal_plan):
        summary = summarise_experiment_plan(minimal_plan)
        assert "canonical_multi_asset_etf" in summary

    def test_summary_contains_primary_metric(self, minimal_plan):
        summary = summarise_experiment_plan(minimal_plan)
        assert "OOS Sharpe ratio" in summary

    def test_summary_contains_diagnostics(self, minimal_plan):
        summary = summarise_experiment_plan(minimal_plan)
        assert "split_metrics" in summary

    def test_summary_contains_risks(self, minimal_plan):
        summary = summarise_experiment_plan(minimal_plan)
        assert "Ridge" in summary

    def test_summary_contains_confirmation_flag(self, minimal_plan):
        summary = summarise_experiment_plan(minimal_plan)
        assert "Requires confirmation" in summary

    def test_summary_contains_executable_ready_flag(self, minimal_plan):
        summary = summarise_experiment_plan(minimal_plan)
        assert "Executable ready" in summary

    def test_summary_shows_warnings_when_present(self, canonical_resolved):
        plan = ExperimentPlan(
            plan_id="x", name="x", description="x",
            resolved_spec=canonical_resolved,
            research_objective="x", hypothesis="x",
            experiment_type="preset_baseline", baseline=None,
            comparison_policy="x",
            warnings=["Test warning"],
        )
        summary = summarise_experiment_plan(plan)
        assert "Test warning" in summary

    def test_summary_no_warnings_section_when_empty(self, minimal_plan):
        summary = summarise_experiment_plan(minimal_plan)
        assert "Warnings:" not in summary

    def test_summary_plan_id_present(self, minimal_plan):
        summary = summarise_experiment_plan(minimal_plan)
        assert "plan_test001" in summary
