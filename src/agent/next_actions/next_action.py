"""NextActionCandidate — one possible next research action.

Phase 5B: Memory-Aware Next-Action Planner.

Governance invariants (enforced in Phase 5B):
  - creates_draft     = False  always
  - executes_experiment = False  always
  - requires_user_confirmation = True  always
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_ACTION_TYPES: frozenset[str] = frozenset(
    {
        "feature_ablation",
        "model_comparison",
        "cost_sensitivity",
        "validation_robustness",
        "portfolio_construction_test",
        "universe_review",
        "diagnostic_review",
        "repeat_baseline",
        "stop_or_deprioritise",
        "needs_more_evidence",
        "custom",
    }
)

VALID_PRIORITIES: frozenset[str] = frozenset({"high", "medium", "low"})
VALID_CONFIDENCES: frozenset[str] = frozenset({"high", "medium", "low"})


@dataclass
class NextActionCandidate:
    """One candidate next research action.

    Recommendations only — never creates drafts, never executes experiments.
    Every candidate requires explicit researcher confirmation before any
    follow-on spec, plan, or draft is created.
    """

    action_id: str
    title: str
    action_type: str
    description: str

    rationale: str
    supporting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)

    suggested_spec_changes: dict[str, Any] = field(default_factory=dict)
    suggested_plan_focus: str | None = None

    expected_benefit: str | None = None
    main_risks: list[str] = field(default_factory=list)
    failure_modes_to_monitor: list[str] = field(default_factory=list)

    priority: str = "medium"     # "high" | "medium" | "low"
    confidence: str = "medium"   # "high" | "medium" | "low"

    requires_new_spec: bool = True
    requires_user_confirmation: bool = True   # Phase 5B: always True
    creates_draft: bool = False               # Phase 5B: always False
    executes_experiment: bool = False         # Phase 5B: always False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "title": self.title,
            "action_type": self.action_type,
            "description": self.description,
            "rationale": self.rationale,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "contradicting_evidence_ids": list(self.contradicting_evidence_ids),
            "suggested_spec_changes": dict(self.suggested_spec_changes),
            "suggested_plan_focus": self.suggested_plan_focus,
            "expected_benefit": self.expected_benefit,
            "main_risks": list(self.main_risks),
            "failure_modes_to_monitor": list(self.failure_modes_to_monitor),
            "priority": self.priority,
            "confidence": self.confidence,
            "requires_new_spec": self.requires_new_spec,
            "requires_user_confirmation": self.requires_user_confirmation,
            "creates_draft": self.creates_draft,
            "executes_experiment": self.executes_experiment,
        }
