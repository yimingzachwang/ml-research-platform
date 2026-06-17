"""Plan critic — structural validation of ExperimentPlan objects.

All checks are deterministic. No LLM calls.

Checks:
  PLAN_A  Plan must have a research objective.
  PLAN_B  Plan must have a hypothesis.
  PLAN_C  Plan must have at least one success criterion.
  PLAN_D  Plan must not evaluate only Sharpe ratio.
  PLAN_E  Plan must include at least one risk or failure mode.
  PLAN_F  Plan must include at least one required diagnostic.
  PLAN_G  Comparison experiments must have a defined baseline.
  PLAN_H  Model comparison: selected model must differ from baseline model.
  PLAN_I  Feature-family test: planned_changes must mention features.
  PLAN_J  Compatibility warnings from resolved spec are carried forward as info.
  PLAN_K  Plan confirmation requirement must match the resolved spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.agent.planner.experiment_plan import ExperimentPlan
    from src.agent.registry.registry_schema import ResearchConfigRegistry

Severity = Literal["error", "warning", "info"]

# Experiment types that require a defined baseline for fair comparison
_COMPARISON_EXPERIMENT_TYPES = frozenset({
    "model_comparison",
    "feature_family_test",
    "universe_test",
    "validation_test",
    "cost_sensitivity_test",
    "portfolio_construction_test",
})


@dataclass(frozen=True)
class PlanCriticIssue:
    """A single plan critic finding."""
    check_id: str
    severity: Severity
    message: str
    field: str | None = None


@dataclass
class PlanCriticResult:
    """Aggregated result of critique_experiment_plan()."""

    errors: list[PlanCriticIssue] = field(default_factory=list)
    warnings: list[PlanCriticIssue] = field(default_factory=list)
    info: list[PlanCriticIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when there are no error-severity issues."""
        return not self.errors

    @property
    def all_issues(self) -> list[PlanCriticIssue]:
        return self.errors + self.warnings + self.info

    def error_messages(self) -> list[str]:
        return [i.message for i in self.errors]

    def warning_messages(self) -> list[str]:
        return [i.message for i in self.warnings]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def critique_experiment_plan(
    plan: ExperimentPlan,
    registry: ResearchConfigRegistry | None = None,
) -> PlanCriticResult:
    """Run all critic checks against an ExperimentPlan.

    Accepts any ExperimentPlan regardless of how it was constructed.
    Returns a PlanCriticResult with errors, warnings, and info issues.

    Args:
        plan:     The experiment plan to validate.
        registry: Pre-loaded registry. Loaded from disk if None.
    """
    if registry is None:
        from src.agent.registry.registry_loader import load_registry
        registry = load_registry()

    result = PlanCriticResult()

    _check_a_research_objective(plan, result)
    _check_b_hypothesis(plan, result)
    _check_c_success_criteria(plan, result)
    _check_d_sharpe_only(plan, result)
    _check_e_risks(plan, result)
    _check_f_diagnostics(plan, result)
    _check_g_baseline_missing(plan, result)
    _check_h_model_comparison_same_model(plan, result, registry)
    _check_i_feature_test_changes(plan, result)
    _check_j_compat_warnings_forwarded(plan, result)
    _check_k_confirmation_consistency(plan, result)

    return result


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_a_research_objective(plan: ExperimentPlan, result: PlanCriticResult) -> None:
    """A: Plan must have a non-empty research objective."""
    if not plan.research_objective or not plan.research_objective.strip():
        result.errors.append(PlanCriticIssue(
            check_id="PLAN_A",
            severity="error",
            message="Plan is missing a research objective.",
            field="research_objective",
        ))


def _check_b_hypothesis(plan: ExperimentPlan, result: PlanCriticResult) -> None:
    """B: Plan must have a non-empty hypothesis."""
    if not plan.hypothesis or not plan.hypothesis.strip():
        result.errors.append(PlanCriticIssue(
            check_id="PLAN_B",
            severity="error",
            message="Plan is missing a hypothesis.",
            field="hypothesis",
        ))


def _check_c_success_criteria(plan: ExperimentPlan, result: PlanCriticResult) -> None:
    """C: Plan must have at least one success criterion."""
    if not plan.success_criteria:
        result.errors.append(PlanCriticIssue(
            check_id="PLAN_C",
            severity="error",
            message="Plan has no success criteria. Add measurable evaluation criteria.",
            field="success_criteria",
        ))


def _check_d_sharpe_only(plan: ExperimentPlan, result: PlanCriticResult) -> None:
    """D: Plan must not evaluate only Sharpe ratio.

    Fires when all success criteria mention only "sharpe" (case-insensitive).
    Does not fire when criteria is empty (covered by check C).
    """
    criteria = plan.success_criteria
    if criteria and all("sharpe" in c.lower() for c in criteria):
        result.errors.append(PlanCriticIssue(
            check_id="PLAN_D",
            severity="error",
            message=(
                "Success criteria only reference Sharpe ratio. "
                "Add risk and capacity metrics (drawdown, turnover, hit rate)."
            ),
            field="success_criteria",
        ))


def _check_e_risks(plan: ExperimentPlan, result: PlanCriticResult) -> None:
    """E: Plan must include at least one risk or failure mode."""
    if not plan.risks and not plan.failure_modes_to_monitor:
        result.errors.append(PlanCriticIssue(
            check_id="PLAN_E",
            severity="error",
            message=(
                "Plan has no risks or failure modes identified. "
                "Every experiment should document what could go wrong."
            ),
            field="risks",
        ))


def _check_f_diagnostics(plan: ExperimentPlan, result: PlanCriticResult) -> None:
    """F: Plan must include at least one required diagnostic."""
    if not plan.required_diagnostics:
        result.errors.append(PlanCriticIssue(
            check_id="PLAN_F",
            severity="error",
            message=(
                "Plan has no required diagnostics. "
                "At minimum, split_metrics or sharpe_ratio is expected."
            ),
            field="required_diagnostics",
        ))


def _check_g_baseline_missing(plan: ExperimentPlan, result: PlanCriticResult) -> None:
    """G: Comparison experiments must have a defined baseline."""
    if plan.experiment_type in _COMPARISON_EXPERIMENT_TYPES and plan.baseline is None:
        result.warnings.append(PlanCriticIssue(
            check_id="PLAN_G",
            severity="warning",
            message=(
                f"Experiment type '{plan.experiment_type}' is a comparison experiment "
                "but no baseline is defined. Results cannot be interpreted without a "
                "reference design."
            ),
            field="baseline",
        ))


def _check_h_model_comparison_same_model(
    plan: ExperimentPlan,
    result: PlanCriticResult,
    registry: ResearchConfigRegistry,
) -> None:
    """H: In a model comparison, selected model must differ from the baseline model."""
    if plan.experiment_type != "model_comparison":
        return
    if plan.baseline is None:
        return  # Covered by PLAN_G

    baseline_model: str | None = None
    try:
        preset = registry.preset(plan.baseline)
        if isinstance(preset.model, dict):
            baseline_model = preset.model.get("type")
    except KeyError:
        return  # Baseline preset not in registry — skip check

    selected_model = plan.resolved_spec.model
    if baseline_model and selected_model and baseline_model == selected_model:
        result.warnings.append(PlanCriticIssue(
            check_id="PLAN_H",
            severity="warning",
            message=(
                f"Selected model '{selected_model}' is the same as the baseline "
                f"preset model '{baseline_model}'. No model comparison is possible."
            ),
            field="model",
        ))


def _check_i_feature_test_changes(plan: ExperimentPlan, result: PlanCriticResult) -> None:
    """I: Feature-family test: planned_changes must mention features."""
    if plan.experiment_type != "feature_family_test":
        return
    has_feature_mention = any("feature" in ch.lower() for ch in plan.planned_changes)
    if not has_feature_mention:
        result.warnings.append(PlanCriticIssue(
            check_id="PLAN_I",
            severity="warning",
            message=(
                "Experiment type is 'feature_family_test' but planned_changes do not "
                "mention features. Clarify what changed in the feature configuration."
            ),
            field="planned_changes",
        ))


def _check_j_compat_warnings_forwarded(
    plan: ExperimentPlan,
    result: PlanCriticResult,
) -> None:
    """J: Compatibility warnings from the resolved spec are surfaced as info."""
    for w in plan.resolved_spec.warnings:
        result.info.append(PlanCriticIssue(
            check_id="PLAN_J",
            severity="info",
            message=f"Resolver warning: {w}",
        ))


def _check_k_confirmation_consistency(
    plan: ExperimentPlan,
    result: PlanCriticResult,
) -> None:
    """K: Plan confirmation requirement must match the resolved spec."""
    if plan.resolved_spec.requires_confirmation and not plan.requires_confirmation:
        result.warnings.append(PlanCriticIssue(
            check_id="PLAN_K",
            severity="warning",
            message=(
                "The resolved spec has requires_confirmation=True (recommended defaults "
                "were applied), but the plan has requires_confirmation=False. "
                "Set plan.requires_confirmation=True before execution."
            ),
            field="requires_confirmation",
        ))
