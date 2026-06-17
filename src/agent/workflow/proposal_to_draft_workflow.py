"""Proposal-to-draft workflow façade (Phase 5F).

Converts an explicitly confirmed ResearchSpecProposal into a validated draft
preview, and optionally saves it as an unapproved draft.

The principal invariant: a draft is created only when the caller explicitly
sets confirm_proposal=True.  No confirmation can be inferred from natural
language or any other implicit signal.

Governance invariants (always enforced):
  - Requires explicit confirm_proposal=True before creating any draft.
  - Does not approve the draft.
  - Does not execute experiments.
  - approval_required          = True  always.
  - requires_user_confirmation = True  always.
  - draft_approved             = False always.
  - execution_performed        = False always.
  - draft_saved reflects whether save_agent_draft_result() succeeded.
"""

from __future__ import annotations

from typing import Any

from src.agent.next_actions.spec_proposal import ResearchSpecProposal
from src.agent.workflow.draft_persistence import save_agent_draft_result
from src.agent.workflow.research_workflow import run_draft_from_spec
from src.agent.workflow.workflow_result import WorkflowResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_proposal_governance(proposal: ResearchSpecProposal) -> str | None:
    """Return an error string if proposal governance invariants are violated.

    These checks defend against proposals that have somehow been constructed
    outside the normal path.  Under normal operation, ResearchSpecProposal
    __post_init__ enforces all four invariants; this boundary guard ensures
    the workflow never silently bypasses them.
    """
    if not proposal.approval_required:
        return "approval_required must be True"
    if not proposal.requires_user_confirmation:
        return "requires_user_confirmation must be True"
    if proposal.creates_draft:
        return "creates_draft must be False"
    if proposal.executes_experiment:
        return "executes_experiment must be False"
    return None


# ---------------------------------------------------------------------------
# Public workflow function
# ---------------------------------------------------------------------------


def run_draft_from_spec_proposal(
    proposal: ResearchSpecProposal,
    confirm_proposal: bool = False,
    base_experiment: str | None = None,
    save_draft: bool = False,
    mode: str = "recommended",
) -> WorkflowResult:
    """Convert a ResearchSpecProposal into a validated draft preview.

    A draft is only created when confirm_proposal=True.  Without this flag
    the function returns status="confirmation_required" and takes no action.

    Saving an unapproved draft requires both confirm_proposal=True and
    save_draft=True.  The draft is persisted but never approved.

    Parameters
    ----------
    proposal:
        A ResearchSpecProposal (from Phase 5D/5E).
    confirm_proposal:
        Must be True to proceed past the confirmation gate.
    base_experiment:
        Optional config name to use as the baseline experiment in the plan.
    save_draft:
        If True, persist the draft using the Phase 3D bridge (unapproved only).
    mode:
        Resolver mode; one of "strict", "recommended", "demo".

    Governance
    ----------
        approval_required          = True  (always)
        requires_user_confirmation = True  (always)
        draft_approved             = False (always)
        execution_performed        = False (always)

    Returns WorkflowResult with status:
        confirmation_required — confirm_proposal was False; no action taken
        invalid_proposal      — proposal governance invariants violated
        unsupported_proposal  — proposal.unsupported_changes is non-empty
        draft_created         — validated draft preview produced
        unsupported_plan      — hard-unsupported fields blocked draft creation
        validation_failed     — draft failed schema validation
        needs_user_input      — spec incomplete or base experiment config missing
        invalid               — spec/plan failed registry or compatibility validation
        error                 — unexpected exception
    """
    # --- 1. Require explicit confirmation ---
    if not confirm_proposal:
        return WorkflowResult(
            status="confirmation_required",
            summary=(
                "Explicit confirmation is required before creating a draft from this proposal. "
                "Set confirm_proposal=True to proceed. "
                "No draft created. No spec saved. No execution performed."
            ),
            spec_proposal=proposal.to_dict(),
            approval_required=True,
            execution_performed=False,
            draft_approved=False,
        )

    # --- 2. Validate proposal governance boundary ---
    gov_error = _validate_proposal_governance(proposal)
    if gov_error:
        return WorkflowResult(
            status="invalid_proposal",
            summary=f"Proposal has invalid governance: {gov_error}. No draft created.",
            errors=[f"Invalid proposal governance: {gov_error}"],
            spec_proposal=proposal.to_dict(),
            approval_required=True,
            execution_performed=False,
            draft_approved=False,
        )

    # --- 3. Reject proposals with unsupported changes ---
    if proposal.unsupported_changes:
        unsupported_list = list(proposal.unsupported_changes)
        return WorkflowResult(
            status="unsupported_proposal",
            summary=(
                f"Proposal has {len(unsupported_list)} unsupported change(s) that cannot "
                "be applied to a draft. Revise the proposal before creating a draft. "
                f"Unsupported: {'; '.join(unsupported_list)}"
            ),
            errors=unsupported_list,
            warnings=[f"Unsupported change: {c}" for c in unsupported_list],
            spec_proposal=proposal.to_dict(),
            approval_required=True,
            execution_performed=False,
            draft_approved=False,
        )

    # --- 4. Run the deterministic draft pipeline ---
    spec_dict = proposal.proposed_spec.to_dict()
    result = run_draft_from_spec(
        spec_dict,
        mode=mode,
        base_experiment=base_experiment,
    )

    # Attach the proposal for provenance / traceability.
    result.spec_proposal = proposal.to_dict()

    # --- 5. Optionally save unapproved draft ---
    if save_draft and result.status == "draft_created":
        persist = save_agent_draft_result(result)
        result.saved_draft = persist.to_dict()
        result.draft_saved = persist.status == "saved"

    return result
