"""WorkflowResult — serialisable envelope for the agent workflow façade.

Every public function in research_workflow.py returns one of these.
All serialised fields are MCP-safe (no Python objects, no dataclasses, no Path).

experiment_draft_object is an optional non-serialised reference to the raw
ExperimentDraft used by the draft_persistence bridge (Phase 3D).  It is never
included in to_dict() output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowResult:
    """Serialisable result of one deterministic agent pipeline step.

    Governance flags (always present):
        approval_required:   True whenever a draft is present.  Draft approval
                             is always a separate, explicit human action.
        execution_performed: Always False — the workflow never runs experiments.
        draft_approved:      Always False — approval is never automatic.

    Unsupported field language:
        hard_unsupported_fields: universe / tickers / frequency / data_source /
                                 asset_class — block draft creation.
        unsupported_fields:      start_date / end_date — reported but do not
                                 block a draft_created result.
    """

    # Status: one of resolved | planned | needs_review | draft_created |
    #         needs_user_input | invalid | unsupported_plan | validation_failed | error
    status: str
    summary: str

    # Structured payloads — plain dicts, safe to serialise
    resolution: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    draft: dict[str, Any] | None = None

    # Provenance / advisory
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    defaulted_fields: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    requires_confirmation: bool = False

    # Governance
    unsupported_fields: list[str] = field(default_factory=list)
    hard_unsupported_fields: list[str] = field(default_factory=list)
    approval_required: bool = True
    execution_performed: bool = False
    draft_approved: bool = False

    # Draft detail (only populated when draft is present)
    mapped_changes: list[dict[str, Any]] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    proposed_name: str | None = None
    base_experiment: str | None = None

    # Non-serialised: raw ExperimentDraft object for draft_persistence bridge.
    # Not included in to_dict(). Set by research_workflow._draft_to_result when
    # status == "draft_created".
    experiment_draft_object: Any = field(default=None, repr=False, compare=False)

    # Phase 4B: parse metadata propagated from IntentParseResult (NL workflow only).
    parse: dict[str, Any] | None = None
    # Phase 4B: persistence record from SavedAgentDraftResult (NL workflow save_draft=True).
    saved_draft: dict[str, Any] | None = None

    # Phase 5C: next-action recommendation payload (next_action_workflow only).
    next_actions: dict[str, Any] | None = None
    # Phase 5C: compact memory context summary (next_action_workflow only).
    memory_context_summary: dict[str, Any] | None = None

    # Phase 5E: spec proposal payload and selected candidate (spec_proposal_workflow only).
    spec_proposal: dict[str, Any] | None = None
    selected_candidate: dict[str, Any] | None = None

    # Phase 5F: True when save_agent_draft_result() successfully persisted the draft.
    draft_saved: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (safe for MCP transport)."""
        return {
            "status": self.status,
            "summary": self.summary,
            "resolution": self.resolution,
            "plan": self.plan,
            "draft": self.draft,
            "warnings": self.warnings,
            "errors": self.errors,
            "missing_fields": self.missing_fields,
            "defaulted_fields": self.defaulted_fields,
            "assumptions": self.assumptions,
            "requires_confirmation": self.requires_confirmation,
            "unsupported_fields": self.unsupported_fields,
            "hard_unsupported_fields": self.hard_unsupported_fields,
            "approval_required": self.approval_required,
            "execution_performed": self.execution_performed,
            "draft_approved": self.draft_approved,
            "mapped_changes": self.mapped_changes,
            "validation_errors": self.validation_errors,
            "proposed_name": self.proposed_name,
            "base_experiment": self.base_experiment,
            # Phase 4B additions
            "parse": self.parse,
            "saved_draft": self.saved_draft,
            # Phase 5C additions
            "next_actions": self.next_actions,
            "memory_context": self.memory_context_summary,
            # Phase 5E additions
            "spec_proposal": self.spec_proposal,
            "selected_candidate": self.selected_candidate,
            "governance": {
                "approval_required": self.approval_required,
                "requires_user_confirmation": True,
                "execution_performed": self.execution_performed,
                "draft_approved": self.draft_approved,
                "draft_created": self.status == "draft_created",
                "draft_saved": self.draft_saved,
            },
        }
