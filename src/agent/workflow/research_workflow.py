"""Research workflow façade — deterministic resolver → planner → draft pipeline.

Three public functions, each building on the previous:

    run_resolve_spec()    — ResearchSpec dict → WorkflowResult (resolved/needs_user_input/invalid)
    run_plan_from_spec()  — ResearchSpec dict → WorkflowResult (planned/needs_review/invalid)
    run_draft_from_spec() — ResearchSpec dict → WorkflowResult (draft_created/…)

Governance:
  - No LLM calls.
  - No experiment execution.
  - No draft approval.
  - No config writes.
  - Hard-unsupported fields (universe, tickers, frequency, data_source, asset_class)
    block draft creation.
  - Soft-unsupported fields (start_date, end_date) are reported in
    result.unsupported_fields but do not block draft_created.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agent.drafting.draft_mapping import (
    detect_hard_unsupported_fields,
)
from src.agent.drafting.plan_to_draft import create_draft_from_plan
from src.agent.intent.research_spec import ResearchSpec
from src.agent.planner.create_experiment_plan import create_experiment_plan
from src.agent.planner.experiment_plan import ExperimentPlan
from src.agent.planner.plan_result import ExperimentPlanResult
from src.agent.resolver.resolution_result import ResearchSpecResolutionResult
from src.agent.resolver.resolve_research_spec import resolve_research_spec
from src.agent.workflow.workflow_result import WorkflowResult

_VALID_MODES: frozenset[str] = frozenset({"strict", "recommended", "demo"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_resolve_spec(
    spec_dict: dict[str, Any],
    mode: str = "recommended",
) -> WorkflowResult:
    """Resolve a ResearchSpec dict using the deterministic resolver.

    Returns WorkflowResult with status:
        resolved         — all required fields filled; spec is valid.
        needs_user_input — required fields remain missing after defaults.
        invalid          — compatibility errors prevent the spec from being used.
        error            — unexpected exception (bad input, registry load failure).
    """
    res, exc = _resolve_raw(spec_dict, mode)
    if exc is not None:
        return WorkflowResult(
            status="error",
            summary=f"Resolution failed: {exc}",
            errors=[str(exc)],
        )
    return _resolution_to_result(res)


def run_plan_from_spec(
    spec_dict: dict[str, Any],
    mode: str = "recommended",
    baseline: str | None = None,
) -> WorkflowResult:
    """Resolve a ResearchSpec dict and create an ExperimentPlan.

    Returns WorkflowResult with status:
        planned          — plan created; no critic errors and no review warnings.
        needs_review     — plan created; critic warnings or requires_confirmation.
        needs_user_input — spec is incomplete; no plan created.
        invalid          — spec or plan failed validation; no plan created.
        error            — unexpected exception.
    """
    res, exc = _resolve_raw(spec_dict, mode)
    if exc is not None:
        return WorkflowResult(
            status="error",
            summary=f"Resolution failed: {exc}",
            errors=[str(exc)],
        )

    resolve_wr = _resolution_to_result(res)
    if resolve_wr.status in ("needs_user_input", "invalid", "error"):
        return resolve_wr

    plan_result, exc = _plan_raw(res.resolved_spec, baseline)
    if exc is not None:
        return WorkflowResult(
            status="error",
            summary=f"Planning failed: {exc}",
            resolution=resolve_wr.resolution,
            errors=[str(exc)],
        )
    return _plan_to_result(resolve_wr, plan_result)


def run_draft_from_spec(
    spec_dict: dict[str, Any],
    mode: str = "recommended",
    baseline: str | None = None,
    base_experiment: str | None = None,
    configs_base: Path | str | None = None,
) -> WorkflowResult:
    """Resolve → plan → create validated draft from a ResearchSpec dict.

    Returns WorkflowResult with status:
        draft_created    — draft assembled and validated; awaiting human approval.
                           result.unsupported_fields may contain date fields.
        needs_review     — plan has warnings; draft still created if possible.
        needs_user_input — spec incomplete or base config not found.
        unsupported_plan — hard-unsupported fields blocked draft creation.
        validation_failed — draft assembled but failed schema validation.
        invalid          — spec or plan failed validation.
        error            — unexpected exception.

    Governance:
        approval_required  is always True when status == draft_created.
        execution_performed is always False.
        draft_approved     is always False.
    """
    res, exc = _resolve_raw(spec_dict, mode)
    if exc is not None:
        return WorkflowResult(
            status="error",
            summary=f"Resolution failed: {exc}",
            errors=[str(exc)],
        )

    resolve_wr = _resolution_to_result(res)
    if resolve_wr.status in ("needs_user_input", "invalid", "error"):
        return resolve_wr

    plan_result, exc = _plan_raw(res.resolved_spec, baseline)
    if exc is not None:
        return WorkflowResult(
            status="error",
            summary=f"Planning failed: {exc}",
            resolution=resolve_wr.resolution,
            errors=[str(exc)],
        )

    if plan_result.is_invalid:
        return _plan_to_result(resolve_wr, plan_result)

    # Optionally override the baseline with an explicit base experiment config name
    plan: ExperimentPlan = plan_result.plan
    if base_experiment is not None:
        plan.baseline = base_experiment

    try:
        draft_result = create_draft_from_plan(plan, configs_base=configs_base)
    except Exception as exc:
        return WorkflowResult(
            status="error",
            summary=f"Draft creation failed: {exc}",
            resolution=resolve_wr.resolution,
            plan=_plan_to_dict(plan),
            errors=[str(exc)],
        )

    plan_wr = _plan_to_result(resolve_wr, plan_result)
    return _draft_to_result(plan_wr, draft_result, res.resolved_spec)


# ---------------------------------------------------------------------------
# Internal: raw pipeline steps
# ---------------------------------------------------------------------------


def _resolve_raw(
    spec_dict: dict[str, Any],
    mode: str,
) -> tuple[ResearchSpecResolutionResult | None, Exception | None]:
    try:
        spec = ResearchSpec.from_dict(spec_dict)
        return resolve_research_spec(spec, mode=_validated_mode(mode)), None
    except Exception as exc:
        return None, exc


def _plan_raw(
    resolved_spec: Any,
    baseline: str | None,
) -> tuple[ExperimentPlanResult | None, Exception | None]:
    try:
        return create_experiment_plan(resolved_spec, baseline=baseline), None
    except Exception as exc:
        return None, exc


# ---------------------------------------------------------------------------
# Internal: result converters
# ---------------------------------------------------------------------------


def _resolution_to_result(res: ResearchSpecResolutionResult) -> WorkflowResult:
    rs = res.resolved_spec
    resolution_dict: dict[str, Any] = {
        "status": res.status,
        "summary": res.summary(),
        "resolved_spec": rs.to_dict() if rs is not None else None,
        "missing_fields": list(res.missing_fields),
        "field_options": {k: list(v) for k, v in res.field_options.items()},
        "recommended_defaults": dict(res.recommended_defaults),
        "assumptions": list(res.assumptions),
        "defaulted_fields": dict(res.defaulted_fields),
        "warnings": list(res.warnings),
        "errors": list(res.errors),
        "requires_confirmation": res.requires_confirmation,
    }
    return WorkflowResult(
        status=res.status,
        summary=res.summary(),
        resolution=resolution_dict,
        warnings=list(res.warnings),
        errors=list(res.errors),
        missing_fields=list(res.missing_fields),
        defaulted_fields=dict(res.defaulted_fields),
        assumptions=list(res.assumptions),
        requires_confirmation=res.requires_confirmation,
    )


def _plan_to_result(
    resolve_wr: WorkflowResult,
    plan_result: ExperimentPlanResult,
) -> WorkflowResult:
    plan_dict = _plan_to_dict(plan_result.plan) if plan_result.plan else None
    all_warnings = (
        list(resolve_wr.warnings)
        + list(plan_result.warnings)
        + list(plan_result.critic_warnings)
    )
    all_errors = list(plan_result.errors) + list(plan_result.critic_errors)

    if plan_result.is_invalid:
        return WorkflowResult(
            status="invalid",
            summary=plan_result.summary,
            resolution=resolve_wr.resolution,
            plan=plan_dict,
            warnings=all_warnings,
            errors=all_errors,
            defaulted_fields=resolve_wr.defaulted_fields,
            assumptions=resolve_wr.assumptions,
            requires_confirmation=resolve_wr.requires_confirmation,
        )

    plan = plan_result.plan
    status = "planned" if plan_result.is_planned else "needs_review"
    return WorkflowResult(
        status=status,
        summary=plan_result.summary,
        resolution=resolve_wr.resolution,
        plan=plan_dict,
        warnings=all_warnings,
        errors=[],
        defaulted_fields=resolve_wr.defaulted_fields,
        assumptions=resolve_wr.assumptions,
        requires_confirmation=plan.requires_confirmation if plan else False,
    )


def _draft_to_result(
    plan_wr: WorkflowResult,
    draft_result: Any,
    resolved_spec: Any,
) -> WorkflowResult:
    all_warnings = list(plan_wr.warnings) + list(draft_result.warnings)
    unsupported = list(draft_result.unsupported_fields)
    hard_unsupported = _hard_unsupported_from_spec(resolved_spec)

    mapped_changes: list[dict[str, Any]] = []
    draft_dict: dict[str, Any] | None = None
    if draft_result.draft is not None:
        draft_dict = _draft_to_dict(draft_result.draft)
        for c in draft_result.draft.changes:
            mapped_changes.append({
                "section": c.section,
                "field": c.field,
                "current_value": c.current_value,
                "proposed_value": c.proposed_value,
                "rationale": c.rationale,
            })

    if draft_result.is_draft_created:
        date_note = ""
        if unsupported:
            date_note = (
                f" Note: the requested {unsupported} were recorded but not applied "
                "to the generated draft because date-range changes are not currently "
                "supported by the draft validator. These fields appear in "
                "result.unsupported_fields."
            )
        return WorkflowResult(
            status="draft_created",
            summary=draft_result.summary + date_note,
            resolution=plan_wr.resolution,
            plan=plan_wr.plan,
            draft=draft_dict,
            warnings=all_warnings,
            errors=[],
            unsupported_fields=unsupported,
            hard_unsupported_fields=hard_unsupported,
            defaulted_fields=plan_wr.defaulted_fields,
            assumptions=plan_wr.assumptions,
            requires_confirmation=plan_wr.requires_confirmation,
            mapped_changes=mapped_changes,
            validation_errors=[],
            proposed_name=draft_result.draft.proposed_name if draft_result.draft else None,
            base_experiment=draft_result.base_experiment,
            approval_required=True,
            execution_performed=False,
            draft_approved=False,
            # Non-serialised reference used by the draft_persistence bridge (Phase 3D).
            experiment_draft_object=draft_result.draft,
        )

    if draft_result.is_unsupported:
        return WorkflowResult(
            status="unsupported_plan",
            summary=draft_result.summary,
            resolution=plan_wr.resolution,
            plan=plan_wr.plan,
            warnings=all_warnings,
            errors=list(draft_result.errors),
            unsupported_fields=unsupported,
            hard_unsupported_fields=hard_unsupported,
            approval_required=False,
            execution_performed=False,
            draft_approved=False,
        )

    if draft_result.is_needs_base_experiment:
        return WorkflowResult(
            status="needs_user_input",
            summary=draft_result.summary,
            resolution=plan_wr.resolution,
            plan=plan_wr.plan,
            warnings=all_warnings,
            errors=list(draft_result.errors),
            unsupported_fields=unsupported,
        )

    if draft_result.is_validation_failed:
        val_errors = (
            list(draft_result.validation_result.errors)
            if draft_result.validation_result else []
        )
        return WorkflowResult(
            status="validation_failed",
            summary=draft_result.summary,
            resolution=plan_wr.resolution,
            plan=plan_wr.plan,
            draft=draft_dict,
            warnings=all_warnings,
            errors=list(draft_result.errors),
            validation_errors=val_errors,
            unsupported_fields=unsupported,
            hard_unsupported_fields=hard_unsupported,
            mapped_changes=mapped_changes,
            approval_required=False,
            execution_performed=False,
            draft_approved=False,
        )

    if draft_result.is_invalid_plan:
        return WorkflowResult(
            status="invalid",
            summary=draft_result.summary,
            resolution=plan_wr.resolution,
            plan=plan_wr.plan,
            warnings=all_warnings,
            errors=list(draft_result.errors),
        )

    return WorkflowResult(
        status="error",
        summary=f"Unexpected draft result status: {draft_result.status!r}",
        errors=[f"Unknown PlanDraftResult status: {draft_result.status!r}"],
    )


# ---------------------------------------------------------------------------
# Internal: serialisation helpers
# ---------------------------------------------------------------------------


def _validated_mode(mode: str) -> str:
    return mode if mode in _VALID_MODES else "recommended"


def _hard_unsupported_from_spec(resolved_spec: Any) -> list[str]:
    try:
        return detect_hard_unsupported_fields(resolved_spec)
    except Exception:
        return []


def _plan_to_dict(plan: ExperimentPlan | None) -> dict[str, Any]:
    if plan is None:
        return {}
    return {
        "plan_id": plan.plan_id,
        "name": plan.name,
        "description": plan.description,
        "experiment_type": plan.experiment_type,
        "baseline": plan.baseline,
        "research_objective": plan.research_objective,
        "hypothesis": plan.hypothesis,
        "comparison_policy": plan.comparison_policy,
        "planned_changes": list(plan.planned_changes),
        "controlled_variables": list(plan.controlled_variables),
        "success_criteria": list(plan.success_criteria),
        "primary_metric": plan.primary_metric,
        "secondary_metrics": list(plan.secondary_metrics),
        "required_diagnostics": list(plan.required_diagnostics),
        "expected_evidence": list(plan.expected_evidence),
        "risks": list(plan.risks),
        "failure_modes_to_monitor": list(plan.failure_modes_to_monitor),
        "assumptions": list(plan.assumptions),
        "warnings": list(plan.warnings),
        "requires_confirmation": plan.requires_confirmation,
        "is_executable_ready": plan.is_executable_ready,
        "created_at": plan.created_at,
    }


def _draft_to_dict(draft: Any) -> dict[str, Any]:
    if draft is None:
        return {}
    return {
        "draft_id": draft.draft_id,
        "draft_hash": draft.draft_hash,
        "base_experiment": draft.base_experiment,
        "proposed_name": draft.proposed_name,
        "changes": [
            {
                "section": c.section,
                "field": c.field,
                "current_value": c.current_value,
                "proposed_value": c.proposed_value,
                "rationale": c.rationale,
            }
            for c in draft.changes
        ],
        "generated_at": draft.generated_at,
        "approved": draft.approved,
    }
