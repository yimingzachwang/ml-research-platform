"""PlanDraftResult — envelope for create_draft_from_plan() outcome."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.agent.planner.experiment_plan import ExperimentPlan
    from src.orchestration.config_generation.draft_schema import (
        DraftValidationResult,
        ExperimentDraft,
    )


@dataclass
class PlanDraftResult:
    """Outcome of translating an ExperimentPlan into an ExperimentDraft.

    Status values:
      draft_created         — ExperimentDraft assembled and validate_draft() passed.
      needs_base_experiment — No base experiment config identified or found on disk.
      unsupported_plan      — Plan contains user-specified fields (e.g. universe,
                              tickers) outside the current change-path whitelist.
      invalid_plan          — Plan is not complete/valid (resolver pre-flight guard).
      validation_failed     — Draft assembled but validate_draft() rejected it.
    """

    status: Literal[
        "draft_created",
        "needs_base_experiment",
        "unsupported_plan",
        "invalid_plan",
        "validation_failed",
    ]
    plan: ExperimentPlan | None

    draft: ExperimentDraft | None = None
    base_experiment: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unsupported_fields: list[str] = field(default_factory=list)
    validation_result: DraftValidationResult | None = None
    summary: str = ""

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    @property
    def is_draft_created(self) -> bool:
        return self.status == "draft_created"

    @property
    def is_needs_base_experiment(self) -> bool:
        return self.status == "needs_base_experiment"

    @property
    def is_unsupported(self) -> bool:
        return self.status == "unsupported_plan"

    @property
    def is_invalid_plan(self) -> bool:
        return self.status == "invalid_plan"

    @property
    def is_validation_failed(self) -> bool:
        return self.status == "validation_failed"


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------


def summarise_plan_draft_result(result: PlanDraftResult) -> str:
    """Return a concise, human-readable summary of a PlanDraftResult."""
    lines: list[str] = [f"Plan Draft Result: {result.status}", ""]

    if result.plan:
        lines.append(f"Plan:            {result.plan.name}")
        lines.append(f"Experiment type: {result.plan.experiment_type}")
    if result.base_experiment:
        lines.append(f"Base experiment: {result.base_experiment}")
    if result.draft:
        lines.append(f"Draft:           {result.draft.proposed_name}")
        lines.append(f"Changes:         {len(result.draft.changes)}")
        lines.append(f"Approved:        {result.draft.approved}")
    if result.errors:
        lines += ["", "Errors:"]
        for e in result.errors:
            lines.append(f"  ! {e}")
    if result.warnings:
        lines += ["", "Warnings:"]
        for w in result.warnings:
            lines.append(f"  ~ {w}")
    if result.unsupported_fields:
        lines += ["", f"Unsupported fields: {', '.join(result.unsupported_fields)}"]

    if result.summary:
        lines += ["", result.summary]

    return "\n".join(lines)
