"""NextActionRecommendationResult — structured recommendation envelope.

Phase 5B: Memory-Aware Next-Action Planner.

Governance invariants:
  - approval_required   = True   always
  - execution_performed = False  always
  - draft_created       = False  always
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.agent.next_actions.next_action import NextActionCandidate

if TYPE_CHECKING:
    from src.agent.memory.memory_context import ResearchMemoryContext


@dataclass
class NextActionRecommendationResult:
    """Structured result from the memory-aware next-action planner.

    status:
      "recommended"        at least one actionable candidate exists
      "needs_more_evidence" memory context was empty or insufficient
      "invalid_input"       no usable objective/spec/plan was provided
    """

    status: str
    objective: str | None

    candidates: list[NextActionCandidate] = field(default_factory=list)
    top_candidate: NextActionCandidate | None = None
    memory_context: ResearchMemoryContext | None = None

    evidence_summary: str = ""
    rejected_directions: list[str] = field(default_factory=list)
    recommended_cautions: list[str] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary: str = ""

    # Governance invariants — never mutate these
    approval_required: bool = True      # always True
    execution_performed: bool = False   # always False
    draft_created: bool = False         # always False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "objective": self.objective,
            "candidates": [c.to_dict() for c in self.candidates],
            "top_candidate": (
                self.top_candidate.to_dict() if self.top_candidate is not None else None
            ),
            "memory_context": (
                self.memory_context.to_dict() if self.memory_context is not None else None
            ),
            "evidence_summary": self.evidence_summary,
            "rejected_directions": list(self.rejected_directions),
            "recommended_cautions": list(self.recommended_cautions),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "summary": self.summary,
            "approval_required": self.approval_required,
            "execution_performed": self.execution_performed,
            "draft_created": self.draft_created,
        }


def summarise_next_action_result(result: NextActionRecommendationResult) -> str:
    """Deterministic plain-text summary of the recommendation result.

    No LLM.  Suitable for inclusion in an agent prompt or MCP tool display.
    """
    lines: list[str] = []

    if result.objective:
        lines.append(f"Objective: {result.objective}")

    if result.status == "needs_more_evidence":
        lines.append(
            "Status: needs_more_evidence. "
            "No prior memory evidence was found or retrieved. "
            "A baseline experiment or diagnostic evidence run is recommended before "
            "pursuing specific design changes."
        )
    elif result.status == "invalid_input":
        lines.append(
            "Status: invalid_input. "
            "No usable objective, spec, plan, or memory context was provided."
        )
    elif result.candidates:
        top = result.candidates[0]
        lines.append(
            f"Top recommendation: {top.title} "
            f"(priority={top.priority}, confidence={top.confidence}, "
            f"type={top.action_type})."
        )
        lines.append(f"Why: {top.rationale[:200]}")
        if top.supporting_evidence_ids:
            ids = ", ".join(top.supporting_evidence_ids[:3])
            lines.append(f"Supporting evidence: {ids}")
        if top.main_risks:
            lines.append(f"Main caution: {top.main_risks[0]}")
    else:
        lines.append(f"Status: {result.status}. No actionable candidates produced.")

    if result.rejected_directions:
        lines.append(f"Avoid: {result.rejected_directions[0][:160]}")

    if result.recommended_cautions:
        lines.append(f"Advisory: {result.recommended_cautions[0][:160]}")

    lines.append(
        "User confirmation required before creating any experiment spec. "
        "No drafts created. No experiments executed."
    )

    return "\n".join(lines)
