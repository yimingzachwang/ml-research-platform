"""Spec proposal workflow façade (Phase 5E).

Connects the Phase 5D candidate-to-spec proposal to both:
  - direct NextActionCandidate input
  - natural-language recommendation request (→ select candidate → propose)

Governance invariants (always enforced):
  - Does not create drafts.
  - Does not execute experiments.
  - Does not approve anything.
  - approval_required          = True  always.
  - requires_user_confirmation = True  always.
  - draft_created              = False always.
  - execution_performed        = False always.
  - draft_approved             = False always.
"""

from __future__ import annotations

from typing import Any

from src.agent.drafting.draft_mapping import _HARD_UNSUPPORTED_SPEC_FIELDS
from src.agent.intent.llm_settings import resolve_llm_settings
from src.agent.intent.research_spec import ResearchSpec
from src.agent.next_actions.candidate_to_spec import propose_spec_from_next_action
from src.agent.next_actions.next_action import NextActionCandidate
from src.agent.workflow.next_action_workflow import run_recommend_next_actions_from_request
from src.agent.workflow.workflow_result import WorkflowResult
from src.orchestration.llm.review_schema import PROVIDER_STUB  # noqa: F401 — re-exported

# Minimum fields required to build a NextActionCandidate from a dict.
_REQUIRED_CANDIDATE_FIELDS: frozenset[str] = frozenset(
    {"action_id", "action_type", "title", "description"}
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_candidate_from_dict(
    d: dict[str, Any],
) -> tuple[NextActionCandidate | None, str | None]:
    """Parse a NextActionCandidate from a plain dict.

    Returns (candidate, None) on success or (None, error_message) on failure.
    Governance flags are never read from the dict — they are always enforced
    by the NextActionCandidate dataclass invariants.
    """
    missing = [f for f in sorted(_REQUIRED_CANDIDATE_FIELDS) if not d.get(f)]
    if missing:
        return None, f"Candidate dict missing required fields: {missing}"
    try:
        rationale = str(d.get("rationale") or d.get("description") or "")
        candidate = NextActionCandidate(
            action_id=str(d["action_id"]),
            title=str(d["title"]),
            action_type=str(d["action_type"]),
            description=str(d["description"]),
            rationale=rationale,
            supporting_evidence_ids=list(d.get("supporting_evidence_ids") or []),
            contradicting_evidence_ids=list(d.get("contradicting_evidence_ids") or []),
            suggested_spec_changes=dict(d.get("suggested_spec_changes") or {}),
            suggested_plan_focus=d.get("suggested_plan_focus"),
            expected_benefit=d.get("expected_benefit"),
            main_risks=list(d.get("main_risks") or []),
            failure_modes_to_monitor=list(d.get("failure_modes_to_monitor") or []),
            priority=str(d.get("priority") or "medium"),
            confidence=str(d.get("confidence") or "medium"),
        )
    except Exception as exc:
        return None, f"Failed to construct NextActionCandidate: {exc}"
    return candidate, None


def _compact_proposal(proposal_result: Any) -> dict[str, Any]:
    """Compact serialisation of a ResearchSpecProposalResult for MCP transport."""
    d = proposal_result.to_dict()
    proposal = d.get("proposal") or {}
    return {
        "status": d.get("status"),
        "summary": d.get("summary"),
        "proposal": proposal if proposal else None,
        "proposed_spec": proposal.get("proposed_spec") if proposal else None,
        "proposed_changes": proposal.get("proposed_changes") if proposal else None,
        "unchanged_fields": proposal.get("unchanged_fields") if proposal else None,
        "unsupported_changes": proposal.get("unsupported_changes") if proposal else [],
        "supporting_evidence_ids": proposal.get("supporting_evidence_ids") if proposal else [],
        "resolution_status": proposal.get("resolution_status") if proposal else None,
        "rationale": proposal.get("rationale") if proposal else None,
        "warnings": d.get("warnings") or [],
        "errors": d.get("errors") or [],
        "governance": {
            "approval_required": True,
            "requires_user_confirmation": True,
            "draft_created": False,
            "execution_performed": False,
            "draft_approved": False,
        },
    }


def _error_result(message: str, errors: list[str] | None = None) -> WorkflowResult:
    return WorkflowResult(
        status="error",
        summary=message,
        errors=errors or [message],
        approval_required=True,
        execution_performed=False,
        draft_approved=False,
    )


# ---------------------------------------------------------------------------
# Public workflow functions
# ---------------------------------------------------------------------------


def run_propose_spec_from_candidate(
    candidate: NextActionCandidate,
    base_spec: ResearchSpec | None = None,
    resolved_spec: Any = None,
    mode: str = "recommended",
) -> WorkflowResult:
    """Convert a NextActionCandidate into a ResearchSpecProposal.

    Governance:
        approval_required          = True  (always)
        requires_user_confirmation = True  (always)
        execution_performed        = False (always)
        draft_created              = False (always)
        draft_approved             = False (always)

    Returns WorkflowResult with status matching ResearchSpecProposalResult.status:
        proposal_created   — spec proposal built and resolved
        needs_user_input   — candidate lacks required spec changes
        unsupported_action — action type or registry values unsupported
        invalid_candidate  — candidate failed structural validation
        resolution_failed  — proposal built but resolver failed
        error              — unexpected exception
    """
    try:
        from src.agent.registry.registry_loader import load_registry
        registry = load_registry()
    except Exception as exc:
        return _error_result(f"Registry load failed: {exc}")

    try:
        proposal_result = propose_spec_from_next_action(
            candidate=candidate,
            base_spec=base_spec,
            resolved_spec=resolved_spec,
            registry=registry,
            mode=mode,
        )
    except Exception as exc:
        return _error_result(f"Spec proposal failed: {exc}")

    compact = _compact_proposal(proposal_result)
    summary = (
        f"Spec proposal [{proposal_result.status}]. "
        + (proposal_result.summary or "")
        + " No draft created. No execution. Approval required."
    )

    return WorkflowResult(
        status=proposal_result.status,
        summary=summary,
        warnings=list(proposal_result.warnings),
        errors=list(proposal_result.errors),
        selected_candidate=candidate.to_dict(),
        spec_proposal=compact,
        approval_required=True,
        execution_performed=False,
        draft_approved=False,
    )


def run_propose_spec_from_request(
    request: str,
    candidate_index: int = 0,
    provider: str | None = None,
    mode: str = "recommended",
    model: str | None = None,
    base_url: str | None = None,
    max_candidates: int = 3,
    max_memory_results: int = 8,
) -> WorkflowResult:
    """Parse a natural-language request, obtain recommendation candidates, select
    one by index, and produce a ResearchSpecProposal.

    Governance:
        approval_required          = True  (always)
        requires_user_confirmation = True  (always)
        execution_performed        = False (always)
        draft_created              = False (always)
        draft_approved             = False (always)

    Returns WorkflowResult with status:
        proposal_created          — spec proposal built successfully
        needs_user_input          — resolution or candidate lacking info
        needs_clarification       — NL intent could not be parsed
        needs_more_evidence       — no candidates returned
        unsupported_action        — action type or registry values unsupported
        invalid_candidate_index   — candidate_index is out of range
        invalid_candidate         — selected candidate failed to reconstruct
        error                     — unexpected exception
    """
    _s = resolve_llm_settings(provider, base_url, model)
    # Step 1: obtain candidates via the NL recommendation workflow.
    na_wr = run_recommend_next_actions_from_request(
        request=request,
        provider=_s.provider,
        mode=mode,
        model=_s.model,
        base_url=_s.base_url,
        max_candidates=max_candidates,
        max_memory_results=max_memory_results,
    )

    # Propagate non-recommendation failures immediately (parsing / resolution errors).
    if na_wr.status not in ("recommended", "needs_more_evidence"):
        return na_wr

    na = na_wr.next_actions or {}
    candidates_raw = na.get("candidates") or []

    if not candidates_raw:
        return WorkflowResult(
            status="needs_more_evidence",
            summary=(
                "No candidates were returned by the recommendation step. "
                "A baseline or diagnostic experiment is recommended before proposing spec changes."
            ),
            warnings=list(na_wr.warnings),
            errors=list(na_wr.errors),
            next_actions=na_wr.next_actions,
            memory_context_summary=na_wr.memory_context_summary,
            approval_required=True,
            execution_performed=False,
            draft_approved=False,
        )

    # Step 2: validate and select by index.
    n = len(candidates_raw)
    if candidate_index < 0 or candidate_index >= n:
        return WorkflowResult(
            status="invalid_candidate_index",
            summary=(
                f"candidate_index={candidate_index} is out of range. "
                f"{n} candidate(s) available (indices 0–{n - 1})."
            ),
            errors=[
                f"candidate_index={candidate_index} out of range (0–{n - 1})."
            ],
            warnings=list(na_wr.warnings),
            next_actions=na_wr.next_actions,
            memory_context_summary=na_wr.memory_context_summary,
            approval_required=True,
            execution_performed=False,
            draft_approved=False,
        )

    selected_raw = candidates_raw[candidate_index]
    candidate, parse_error = _parse_candidate_from_dict(selected_raw)
    if parse_error:
        return WorkflowResult(
            status="invalid_candidate",
            summary=f"Could not reconstruct candidate from recommendation output: {parse_error}",
            errors=[parse_error],
            warnings=list(na_wr.warnings),
            next_actions=na_wr.next_actions,
            memory_context_summary=na_wr.memory_context_summary,
            approval_required=True,
            execution_performed=False,
            draft_approved=False,
        )

    # Step 3: extract a base ResearchSpec from the resolution result if available.
    base_spec: ResearchSpec | None = None
    if na_wr.resolution:
        resolved_fields = (na_wr.resolution or {}).get("resolved_spec") or {}
        if resolved_fields:
            try:
                # Strip hard-unsupported structural fields; the resolver
                # re-populates them from the preset, and including them here
                # would mark them as user_specified → unsupported_plan.
                filtered = {k: v for k, v in resolved_fields.items()
                            if k not in _HARD_UNSUPPORTED_SPEC_FIELDS}
                base_spec = ResearchSpec.from_dict(filtered)
            except Exception:
                pass  # Non-fatal — base_spec stays None

    # Step 4: propose spec from the selected candidate.
    proposal_wr = run_propose_spec_from_candidate(
        candidate=candidate,
        base_spec=base_spec,
        mode=mode,
    )

    # Attach full recommendation context for the caller.
    proposal_wr.next_actions = na_wr.next_actions
    proposal_wr.memory_context_summary = na_wr.memory_context_summary
    proposal_wr.resolution = na_wr.resolution
    proposal_wr.parse = na_wr.parse

    return proposal_wr
