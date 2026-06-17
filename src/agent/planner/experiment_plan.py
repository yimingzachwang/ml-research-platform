"""ExperimentPlan — human-readable research design artifact.

Not executable. Represents the research reasoning layer between
ResolvedResearchSpec and future config draft generation (Phase 3B).

The plan explains:
  - what research question is being tested
  - what hypothesis is being evaluated
  - what baseline is used for comparison
  - what changes are planned and what is held constant
  - what success criteria apply
  - what evidence to inspect
  - what risks and failure modes to monitor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.resolver.resolved_spec import ResolvedResearchSpec


@dataclass
class ExperimentPlan:
    """A non-executable research design document.

    Derived deterministically from a ResolvedResearchSpec. Preserves full
    provenance. Must be reviewed and approved before Phase 3B config generation.
    """

    # -----------------------------------------------------------------
    # Identity
    # -----------------------------------------------------------------
    plan_id: str
    name: str
    description: str

    # -----------------------------------------------------------------
    # Source — the fully-resolved spec this plan was derived from
    # -----------------------------------------------------------------
    resolved_spec: ResolvedResearchSpec

    # -----------------------------------------------------------------
    # Research framing
    # -----------------------------------------------------------------
    research_objective: str
    hypothesis: str
    # One of: preset_baseline | model_comparison | feature_family_test |
    #         universe_test | validation_test | cost_sensitivity_test |
    #         portfolio_construction_test | custom_research_design
    experiment_type: str

    # -----------------------------------------------------------------
    # Comparison framing
    # -----------------------------------------------------------------
    # Name of the preset or design used as the comparison reference.
    # None when no valid baseline can be inferred.
    baseline: str | None
    comparison_policy: str

    # -----------------------------------------------------------------
    # Changes and controls
    # -----------------------------------------------------------------
    planned_changes: list[str] = field(default_factory=list)
    controlled_variables: list[str] = field(default_factory=list)

    # -----------------------------------------------------------------
    # Evaluation design
    # -----------------------------------------------------------------
    success_criteria: list[str] = field(default_factory=list)
    primary_metric: str | None = None
    secondary_metrics: list[str] = field(default_factory=list)

    # -----------------------------------------------------------------
    # Evidence collection
    # -----------------------------------------------------------------
    required_diagnostics: list[str] = field(default_factory=list)
    expected_evidence: list[str] = field(default_factory=list)

    # -----------------------------------------------------------------
    # Risk and failure monitoring
    # -----------------------------------------------------------------
    risks: list[str] = field(default_factory=list)
    failure_modes_to_monitor: list[str] = field(default_factory=list)

    # -----------------------------------------------------------------
    # Provenance carried forward from resolved spec
    # -----------------------------------------------------------------
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # -----------------------------------------------------------------
    # Review gates
    # -----------------------------------------------------------------
    requires_confirmation: bool = False
    # Always False in Phase 3A; Phase 3B config draft generation sets this.
    is_executable_ready: bool = False

    created_at: str | None = None

    # -----------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the plan to a plain dict (suitable for JSON/YAML)."""
        return {
            "plan_id": self.plan_id,
            "name": self.name,
            "description": self.description,
            "resolved_spec": self.resolved_spec.to_dict(),
            "research_objective": self.research_objective,
            "hypothesis": self.hypothesis,
            "experiment_type": self.experiment_type,
            "baseline": self.baseline,
            "comparison_policy": self.comparison_policy,
            "planned_changes": list(self.planned_changes),
            "controlled_variables": list(self.controlled_variables),
            "success_criteria": list(self.success_criteria),
            "primary_metric": self.primary_metric,
            "secondary_metrics": list(self.secondary_metrics),
            "required_diagnostics": list(self.required_diagnostics),
            "expected_evidence": list(self.expected_evidence),
            "risks": list(self.risks),
            "failure_modes_to_monitor": list(self.failure_modes_to_monitor),
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
            "requires_confirmation": self.requires_confirmation,
            "is_executable_ready": self.is_executable_ready,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

def summarise_experiment_plan(plan: ExperimentPlan) -> str:
    """Return a concise, human-readable summary suitable for user review.

    Covers: objective, hypothesis, baseline, planned changes, success criteria,
    required diagnostics, risks, and confirmation requirement.
    """
    lines: list[str] = [
        f"Experiment Plan: {plan.name}",
        f"ID: {plan.plan_id}",
        "",
        f"Objective:       {plan.research_objective}",
        f"Hypothesis:      {plan.hypothesis}",
        f"Experiment type: {plan.experiment_type}",
        f"Baseline:        {plan.baseline or 'none'}",
        "",
        "Planned changes:",
    ]
    for ch in plan.planned_changes:
        lines.append(f"  - {ch}")

    lines += ["", "Success criteria:"]
    for cr in plan.success_criteria:
        lines.append(f"  - {cr}")

    lines += [
        "",
        f"Primary metric: {plan.primary_metric or 'not set'}",
        "",
        "Required diagnostics:",
    ]
    for d in plan.required_diagnostics:
        lines.append(f"  - {d}")

    lines += ["", "Risks:"]
    for r in plan.risks:
        lines.append(f"  - {r}")

    if plan.failure_modes_to_monitor:
        lines += ["", "Failure modes to monitor:"]
        for fm in plan.failure_modes_to_monitor:
            lines.append(f"  - {fm}")

    if plan.warnings:
        lines += ["", "Warnings:"]
        for w in plan.warnings:
            lines.append(f"  ! {w}")

    lines += [
        "",
        f"Requires confirmation: {plan.requires_confirmation}",
        f"Executable ready:      {plan.is_executable_ready}",
    ]
    return "\n".join(lines)
