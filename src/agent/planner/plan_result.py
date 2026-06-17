"""ExperimentPlanResult — envelope returned by create_experiment_plan()."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.agent.planner.experiment_plan import ExperimentPlan

PlanStatus = Literal["planned", "needs_review", "invalid"]


@dataclass
class ExperimentPlanResult:
    """Full result of create_experiment_plan().

    Status rules:
        planned:      Plan created; no blocking critic errors and no review warnings.
        needs_review: Plan created; critic or resolver warnings present, or
                      requires_confirmation=True from defaults.
        invalid:      Plan could not be created (incomplete/invalid resolved spec)
                      or failed a blocking critic check.
    """

    status: PlanStatus
    plan: ExperimentPlan | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    critic_errors: list[str] = field(default_factory=list)
    critic_warnings: list[str] = field(default_factory=list)
    summary: str = ""

    # -----------------------------------------------------------------

    @property
    def is_planned(self) -> bool:
        return self.status == "planned"

    @property
    def needs_review(self) -> bool:
        return self.status == "needs_review"

    @property
    def is_invalid(self) -> bool:
        return self.status == "invalid"
