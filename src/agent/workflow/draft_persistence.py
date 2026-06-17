"""Phase 3D — Persistence bridge: save a validated agent-workflow draft.

Bridges WorkflowResult (from research_workflow) into the existing draft storage
mechanism (src/orchestration/config_generation/draft_generator.save_draft).

Governance:
  - Only saves when status == "draft_created" and no hard-unsupported fields.
  - Does NOT approve the draft.
  - Does NOT render to YAML.
  - Does NOT execute any experiment.
  - approval_required / execution_performed / draft_approved are always set to
    True / False / False respectively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.orchestration.config_generation.draft_generator import save_draft as _save_draft

_SOFT_DATE_NOTE = (
    "The requested {fields} were recorded but not applied to the draft because "
    "date-range changes are not currently supported by the draft validator."
)


@dataclass
class SavedAgentDraftResult:
    """Result of save_agent_draft_result().

    status:
        "saved"     — draft written to disk; awaiting human approval.
        "not_saved" — conditions not met (wrong workflow status, hard-unsupported
                      fields, or no ExperimentDraft object available).
        "error"     — an unexpected exception occurred during disk persistence.
    """

    status: str  # "saved" | "not_saved" | "error"
    draft_id: str | None = None
    draft_path: str | None = None
    base_experiment: str | None = None
    proposed_name: str | None = None
    approval_required: bool = True    # always True
    execution_performed: bool = False  # always False
    draft_approved: bool = False      # always False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "draft_id": self.draft_id,
            "draft_path": self.draft_path,
            "base_experiment": self.base_experiment,
            "proposed_name": self.proposed_name,
            "approval_required": self.approval_required,
            "execution_performed": self.execution_performed,
            "draft_approved": self.draft_approved,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "summary": self.summary,
        }


def save_agent_draft_result(
    workflow_result: Any,
    base_dir: Path | str | None = None,
) -> SavedAgentDraftResult:
    """Persist a validated agent-workflow draft using the existing draft storage.

    Accepts a WorkflowResult (from run_draft_from_spec).  Only persists when all
    of the following hold:
      1. workflow_result.status == "draft_created"
      2. workflow_result.hard_unsupported_fields is empty
      3. workflow_result.experiment_draft_object is not None

    Returns SavedAgentDraftResult with status "saved", "not_saved", or "error".
    The returned result is safe to serialise to MCP transport (all fields are
    plain Python types).
    """
    # Guard 1: status must be draft_created
    status = getattr(workflow_result, "status", None)
    if status != "draft_created":
        return SavedAgentDraftResult(
            status="not_saved",
            summary=(
                f"Draft not saved: workflow status is {status!r}, "
                "expected 'draft_created'."
            ),
            errors=[f"Cannot persist draft with status={status!r}."],
        )

    # Guard 2: hard-unsupported fields must be absent
    hard_fields = list(getattr(workflow_result, "hard_unsupported_fields", []) or [])
    if hard_fields:
        return SavedAgentDraftResult(
            status="not_saved",
            summary=f"Draft not saved: hard-unsupported fields present: {hard_fields}.",
            errors=[f"Hard-unsupported fields block persistence: {hard_fields}."],
        )

    # Guard 3: ExperimentDraft object must be available
    experiment_draft = getattr(workflow_result, "experiment_draft_object", None)
    if experiment_draft is None:
        return SavedAgentDraftResult(
            status="not_saved",
            summary="Draft not saved: no ExperimentDraft object available for persistence.",
            errors=["experiment_draft_object is None — cannot persist without the raw draft."],
        )

    # Soft-unsupported date fields: warned but not blocking
    warnings: list[str] = []
    soft_fields = list(getattr(workflow_result, "unsupported_fields", []) or [])
    if soft_fields:
        warnings.append(_SOFT_DATE_NOTE.format(fields=soft_fields))

    # Persist
    try:
        path = _save_draft(experiment_draft, llm_base=base_dir)
    except Exception as exc:
        return SavedAgentDraftResult(
            status="error",
            summary=f"Draft persistence failed: {exc}",
            errors=[str(exc)],
        )

    return SavedAgentDraftResult(
        status="saved",
        draft_id=str(experiment_draft.draft_id),
        draft_path=str(path),
        base_experiment=str(experiment_draft.base_experiment),
        proposed_name=str(experiment_draft.proposed_name),
        approval_required=True,
        execution_performed=False,
        draft_approved=False,
        warnings=warnings,
        summary=(
            f"Draft '{experiment_draft.proposed_name}' "
            f"(id={experiment_draft.draft_id}) saved to {path}. "
            "Awaiting human approval before any execution."
        ),
    )
