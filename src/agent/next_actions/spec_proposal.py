"""ResearchSpecProposal — Phase 5D.

A proposed ResearchSpec derived from a NextActionCandidate.

Governance invariants (always enforced):
  requires_user_confirmation = True
  creates_draft              = False
  executes_experiment        = False
  approval_required          = True

These values are set at construction and never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from src.agent.intent.research_spec import ResearchSpec
    from src.agent.resolver.resolution_result import ResearchSpecResolutionResult

ProposalStatus = Literal[
    "proposal_created",
    "needs_user_input",
    "unsupported_action",
    "invalid_candidate",
    "resolution_failed",
]


@dataclass
class ResearchSpecProposal:
    """A proposed ResearchSpec derived from one NextActionCandidate.

    Provenance is fully preserved:
      - ``source_candidate_id`` / ``source_action_type`` link back to the planner output
      - ``proposed_changes`` contains only the fields changed by the action
      - ``unchanged_fields`` contains the base-spec fields that were carried forward
      - ``supporting_evidence_ids`` come directly from the candidate
      - ``unsupported_changes`` lists any candidate suggestions that could not be applied

    Governance invariants (read-only, always True/False as stated):
      requires_user_confirmation = True
      creates_draft              = False
      executes_experiment        = False
      approval_required          = True
    """

    proposal_id: str
    source_candidate_id: str
    source_action_type: str

    # The spec we started from (None if derived from resolved_spec or nothing).
    base_spec: ResearchSpec | None
    # The proposed spec after applying candidate.suggested_spec_changes.
    proposed_spec: ResearchSpec

    # Diff: what changed vs. what was preserved
    proposed_changes: dict[str, Any] = field(default_factory=dict)
    unchanged_fields: dict[str, Any] = field(default_factory=dict)

    # Provenance from the source candidate
    rationale: str = ""
    supporting_evidence_ids: list[str] = field(default_factory=list)

    # Advisory
    warnings: list[str] = field(default_factory=list)
    unsupported_changes: list[str] = field(default_factory=list)

    # From running resolve_research_spec() on the proposed spec
    resolution_status: str | None = None
    resolution_summary: str | None = None

    # Governance — always these values; never mutated
    requires_user_confirmation: bool = True
    creates_draft: bool = False
    executes_experiment: bool = False
    approval_required: bool = True

    def __post_init__(self) -> None:
        # Governance hard enforcement: never allow mutation of safety fields.
        object.__setattr__(self, "requires_user_confirmation", True)
        object.__setattr__(self, "creates_draft", False)
        object.__setattr__(self, "executes_experiment", False)
        object.__setattr__(self, "approval_required", True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchSpecProposal":
        """Reconstruct a ResearchSpecProposal from a to_dict() output.

        Governance invariants are always re-enforced by __post_init__ regardless
        of what the input dict claims for creates_draft / executes_experiment etc.
        """
        from src.agent.intent.research_spec import ResearchSpec as _RS

        proposed_spec_dict = data.get("proposed_spec") or {}
        base_spec_dict = data.get("base_spec")
        proposed_spec = _RS.from_dict(proposed_spec_dict)
        base_spec = _RS.from_dict(base_spec_dict) if base_spec_dict else None
        return cls(
            proposal_id=str(data.get("proposal_id") or ""),
            source_candidate_id=str(data.get("source_candidate_id") or ""),
            source_action_type=str(data.get("source_action_type") or ""),
            base_spec=base_spec,
            proposed_spec=proposed_spec,
            proposed_changes=dict(data.get("proposed_changes") or {}),
            unchanged_fields=dict(data.get("unchanged_fields") or {}),
            rationale=str(data.get("rationale") or ""),
            supporting_evidence_ids=list(data.get("supporting_evidence_ids") or []),
            warnings=list(data.get("warnings") or []),
            unsupported_changes=list(data.get("unsupported_changes") or []),
            resolution_status=data.get("resolution_status"),
            resolution_summary=data.get("resolution_summary"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (MCP-safe)."""
        return {
            "proposal_id": self.proposal_id,
            "source_candidate_id": self.source_candidate_id,
            "source_action_type": self.source_action_type,
            "base_spec": self.base_spec.to_dict() if self.base_spec is not None else None,
            "proposed_spec": self.proposed_spec.to_dict(),
            "proposed_changes": dict(self.proposed_changes),
            "unchanged_fields": dict(self.unchanged_fields),
            "rationale": self.rationale,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "warnings": list(self.warnings),
            "unsupported_changes": list(self.unsupported_changes),
            "resolution_status": self.resolution_status,
            "resolution_summary": self.resolution_summary,
            "requires_user_confirmation": self.requires_user_confirmation,
            "creates_draft": self.creates_draft,
            "executes_experiment": self.executes_experiment,
            "approval_required": self.approval_required,
        }


@dataclass
class ResearchSpecProposalResult:
    """Result envelope for propose_spec_from_next_action().

    Status values:
        proposal_created  — proposal built and proposed_spec resolves cleanly
        needs_user_input  — resolver needs more information (missing fields)
        unsupported_action — action type or requested changes cannot be mapped
        invalid_candidate  — the input NextActionCandidate is structurally invalid
        resolution_failed  — proposed spec failed compatibility validation
    """

    status: ProposalStatus
    proposal: ResearchSpecProposal | None
    resolution_result: ResearchSpecResolutionResult | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (MCP-safe)."""
        res = self.resolution_result
        return {
            "status": self.status,
            "proposal": self.proposal.to_dict() if self.proposal is not None else None,
            "resolution_result": (
                {
                    "status": res.status,
                    "missing_fields": res.missing_fields,
                    "errors": res.errors,
                    "warnings": res.warnings,
                    "summary": res.summary(),
                }
                if res is not None
                else None
            ),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Summary function
# ---------------------------------------------------------------------------


def summarise_spec_proposal(result: ResearchSpecProposalResult) -> str:
    """Deterministic plain-text summary of a proposal result. No LLM.

    Always ends with the governance statement (no draft, no execution).
    """
    parts: list[str] = [f"Phase 5D Spec Proposal [{result.status}]."]

    p = result.proposal
    if p is not None:
        parts.append(f"Action type: {p.source_action_type}.")
        if p.proposed_changes:
            change_strs = [
                f"{k}={v!r}" for k, v in list(p.proposed_changes.items())[:5]
            ]
            parts.append(f"Proposed changes: {', '.join(change_strs)}.")
        else:
            parts.append("Proposed changes: none (repeat baseline).")
        if p.rationale:
            parts.append(f"Rationale: {p.rationale[:120]}.")
        if p.supporting_evidence_ids:
            ids = ", ".join(p.supporting_evidence_ids[:3])
            parts.append(f"Supporting evidence: {ids}.")
        if p.unsupported_changes:
            parts.append(
                f"Unsupported changes ({len(p.unsupported_changes)}): "
                + "; ".join(p.unsupported_changes[:2])
                + ("..." if len(p.unsupported_changes) > 2 else "")
                + "."
            )
        if p.resolution_status:
            parts.append(f"Resolution: {p.resolution_status}.")
        if p.resolution_summary:
            parts.append(f"Resolution summary: {p.resolution_summary[:80]}.")

    if result.errors:
        parts.append(f"Errors: {'; '.join(result.errors[:3])}.")
    if result.warnings:
        parts.append(f"Warnings ({len(result.warnings)}).")

    parts.append(
        "User confirmation required before creating any experiment spec. "
        "No draft created. No approval. No execution performed."
    )
    return " ".join(parts)
