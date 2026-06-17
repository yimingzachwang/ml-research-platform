"""Agent drafting layer — ExperimentPlan → ExperimentDraft bridge (Phase 3B).

Pipeline:
  create_draft_from_plan()  — ExperimentPlan → PlanDraftResult (validated, unapproved)

The draft is not approved or rendered here. Call approve_draft() then
render_to_yaml() (Phase 3C) to write the final YAML config.

Governance:
  detect_hard_unsupported_fields() — fields that block draft creation
  detect_unsupported_fields()      — all unreportable fields (hard + soft dates)
"""

from src.agent.drafting.draft_mapping import (
    detect_hard_unsupported_fields,
    detect_unsupported_fields,
    map_spec_to_changes,
)
from src.agent.drafting.draft_result import PlanDraftResult, summarise_plan_draft_result
from src.agent.drafting.plan_to_draft import create_draft_from_plan

__all__ = [
    "PlanDraftResult",
    "summarise_plan_draft_result",
    "create_draft_from_plan",
    "detect_unsupported_fields",
    "detect_hard_unsupported_fields",
    "map_spec_to_changes",
]
