"""Next-action recommendation workflow façade (Phase 5C).

Connects the Phase 5B next-action planner to both structured-spec and
natural-language pipeline entry points.

    structured spec / ResearchSpec
    → resolve_research_spec()       (deterministic resolver)
    → create_experiment_plan()      (deterministic planner)
    → retrieve_research_memory_context()  (Phase 5A connector)
    → recommend_next_actions()      (Phase 5B planner)
    → WorkflowResult with next_actions payload

    NL request
    → parse_research_intent()       (Phase 4A intent parser)
    → structured path above

Governance invariants (always enforced):
  - Does not create drafts.
  - Does not execute experiments.
  - Does not approve anything.
  - approval_required = True always.
  - execution_performed = False always.
  - draft_created = False always (never status="draft_created").
  - Execution/approval keywords in NL request → warning, never acted on.
  - needs_user_input from resolver propagated as-is (caller must provide more info).
  - max_memory_results controls top_k on the memory query.
  - Truncation beyond max_candidates is reported in warnings.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from src.agent.intent.research_spec import ResearchSpec
from src.agent.memory.memory_query import (
    ResearchMemoryQuery,
    query_from_experiment_plan,
    query_from_research_spec,
    query_from_resolved_spec,
)
from src.agent.next_actions.next_action_planner import recommend_next_actions
from src.agent.next_actions.next_action_result import (
    NextActionRecommendationResult,
    summarise_next_action_result,
)
from src.agent.planner.create_experiment_plan import create_experiment_plan
from src.agent.resolver.resolve_research_spec import (
    resolve_research_spec as _resolve_fn,
)
from src.agent.workflow.natural_language_workflow import (
    _check_execution_keywords,
    _make_parse_dict,
    _make_parse_error_result,
    _merge_parse_warnings,
    _prepend_warning,
)
from src.agent.workflow.research_workflow import (
    _plan_to_dict as _plan_to_serial,
    _resolution_to_result,
    _validated_mode,
)
from src.agent.intent.llm_settings import resolve_llm_settings
from src.agent.workflow.workflow_result import WorkflowResult
from src.orchestration.llm.review_schema import PROVIDER_STUB  # noqa: F401 — re-exported

# Request all candidates from the planner to detect truncation.
_CANDIDATE_CEILING: int = 64


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_memory_context(
    native_spec: ResearchSpec | None,
    native_resolved: Any,
    native_plan: Any,
    objective: str | None,
    max_memory_results: int,
) -> Any | None:
    """Build a ResearchMemoryContext using the Phase 5A connector.

    Returns None on any error (non-fatal; the planner handles empty context).
    """
    try:
        from src.agent.memory.memory_connector import retrieve_research_memory_context

        if native_plan is not None:
            query = query_from_experiment_plan(
                native_plan,
                top_k_keyword=max_memory_results,
                top_k_semantic=max_memory_results,
            )
        elif native_resolved is not None:
            query = query_from_resolved_spec(
                native_resolved,
                top_k_keyword=max_memory_results,
                top_k_semantic=max_memory_results,
            )
        elif native_spec is not None:
            query = query_from_research_spec(
                native_spec,
                top_k_keyword=max_memory_results,
                top_k_semantic=max_memory_results,
            )
        elif objective:
            tokens = list(
                dict.fromkeys(re.findall(r"[a-z0-9_]+", objective.lower()))
            )
            query = ResearchMemoryQuery(
                keyword_query=" ".join(tokens) or None,
                semantic_query=objective,
                source="objective",
                top_k_keyword=max_memory_results,
                top_k_semantic=max_memory_results,
            )
        else:
            return None

        return retrieve_research_memory_context(query)
    except Exception:  # noqa: BLE001 — memory retrieval is always non-fatal
        return None


def _compact_memory_context(na_result: NextActionRecommendationResult) -> dict[str, Any] | None:
    """Compact serialisation of the memory context (no evidence item bodies)."""
    ctx = na_result.memory_context
    if ctx is None:
        return None
    return {
        "keyword_count": ctx.keyword_count,
        "semantic_count": ctx.semantic_count,
        "total_items": len(ctx.evidence_items),
        "experiment_summary": ctx.experiment_summary,
        "failure_mode_summary": ctx.failure_mode_summary,
        "warnings": list(ctx.warnings),
        "is_empty": ctx.is_empty(),
    }


def _compact_na_result(na_result: NextActionRecommendationResult) -> dict[str, Any]:
    """Serialise NextActionRecommendationResult without the full memory_context."""
    return {
        "status": na_result.status,
        "objective": na_result.objective,
        "candidates": [c.to_dict() for c in na_result.candidates],
        "top_candidate": (
            na_result.top_candidate.to_dict() if na_result.top_candidate is not None else None
        ),
        "evidence_summary": na_result.evidence_summary,
        "rejected_directions": list(na_result.rejected_directions),
        "recommended_cautions": list(na_result.recommended_cautions),
        "warnings": list(na_result.warnings),
        "errors": list(na_result.errors),
        "summary": na_result.summary,
        "approval_required": na_result.approval_required,
        "execution_performed": na_result.execution_performed,
        "draft_created": na_result.draft_created,
    }


def _apply_truncation(
    na_result: NextActionRecommendationResult,
    max_candidates: int,
) -> list[str]:
    """Truncate na_result.candidates in-place; return warnings generated."""
    total = len(na_result.candidates)
    if total <= max_candidates:
        return []
    na_result.candidates = na_result.candidates[:max_candidates]
    na_result.top_candidate = na_result.candidates[0] if na_result.candidates else None
    na_result.summary = summarise_next_action_result(na_result)
    return [
        f"{total} candidate(s) were generated; "
        f"{max_candidates} returned due to max_candidates={max_candidates}."
    ]


def _wrap_into_workflow_result(
    na_result: NextActionRecommendationResult,
    *,
    resolve_wr: WorkflowResult | None,
    plan_wr: WorkflowResult | None,
    parse: dict[str, Any] | None,
    extra_warnings: list[str],
) -> WorkflowResult:
    """Package a NextActionRecommendationResult into a WorkflowResult."""
    all_warnings = extra_warnings + list(na_result.warnings)
    return WorkflowResult(
        status=na_result.status,
        summary=na_result.summary,
        resolution=resolve_wr.resolution if resolve_wr else None,
        plan=plan_wr.plan if plan_wr else None,
        parse=parse,
        warnings=all_warnings,
        errors=list(na_result.errors),
        next_actions=_compact_na_result(na_result),
        memory_context_summary=_compact_memory_context(na_result),
        approval_required=True,
        execution_performed=False,
        draft_approved=False,
    )


# ---------------------------------------------------------------------------
# Public workflow functions
# ---------------------------------------------------------------------------


def run_recommend_next_actions(
    objective: str | None = None,
    spec: ResearchSpec | None = None,
    mode: str = "recommended",
    max_candidates: int = 3,
    max_memory_results: int = 8,
    memory_context: Any = None,
) -> WorkflowResult:
    """Resolve a ResearchSpec (if given), create an ExperimentPlan, retrieve
    memory evidence, and recommend deterministic next research actions.

    Governance:
        approval_required  = True  (always)
        execution_performed = False (always)
        draft_created       = False (always — never returns status "draft_created")

    Returns WorkflowResult with status:
        recommended         — actionable candidates produced
        needs_more_evidence — memory was empty; baseline / diagnostic run recommended
        needs_user_input    — spec resolution requires more information from the user
        invalid             — spec failed registry / compatibility validation
        error               — unexpected exception
    """
    native_spec = spec
    native_resolved: Any = None
    native_plan: Any = None
    resolve_wr: WorkflowResult | None = None
    plan_wr: WorkflowResult | None = None

    if spec is not None:
        spec_dict = {k: v for k, v in spec.to_dict().items() if v is not None}
        try:
            res = _resolve_fn(spec, mode=_validated_mode(mode))
            resolve_wr = _resolution_to_result(res)
        except Exception as exc:
            return WorkflowResult(
                status="error",
                summary=f"Resolution failed: {exc}",
                errors=[str(exc)],
                approval_required=True,
                execution_performed=False,
                draft_approved=False,
            )

        if resolve_wr.status in ("needs_user_input", "invalid", "error"):
            return resolve_wr

        native_resolved = res.resolved_spec

        # Planning is best-effort; a plan failure never blocks recommendations.
        if native_resolved is not None:
            try:
                plan_result = create_experiment_plan(native_resolved)
                native_plan = plan_result.plan if plan_result else None
                if native_plan is not None:
                    plan_dict = _plan_to_serial(native_plan)
                    plan_wr = WorkflowResult(
                        status="planned",
                        summary=plan_result.summary if plan_result else "",
                        plan=plan_dict,
                        approval_required=True,
                        execution_performed=False,
                        draft_approved=False,
                    )
            except Exception:
                pass  # Non-fatal

    # Build memory context with max_memory_results cap (unless caller provides one).
    if memory_context is None:
        memory_context = _build_memory_context(
            native_spec=native_spec,
            native_resolved=native_resolved,
            native_plan=native_plan,
            objective=objective,
            max_memory_results=max_memory_results,
        )

    # Run the planner with ceiling max_candidates to detect truncation.
    try:
        na_result = recommend_next_actions(
            objective=objective,
            research_spec=native_spec,
            resolved_spec=native_resolved,
            experiment_plan=native_plan,
            memory_context=memory_context,
            max_candidates=_CANDIDATE_CEILING,
        )
    except Exception as exc:
        return WorkflowResult(
            status="error",
            summary=f"Next-action recommendation failed: {exc}",
            resolution=resolve_wr.resolution if resolve_wr else None,
            errors=[str(exc)],
            approval_required=True,
            execution_performed=False,
            draft_approved=False,
        )

    extra_warnings = _apply_truncation(na_result, max_candidates)

    return _wrap_into_workflow_result(
        na_result,
        resolve_wr=resolve_wr,
        plan_wr=plan_wr,
        parse=None,
        extra_warnings=extra_warnings,
    )


def run_recommend_next_actions_from_request(
    request: str,
    provider: str | None = None,
    mode: str = "recommended",
    model: str | None = None,
    base_url: str | None = None,
    llm_client: Callable[[str], str] | None = None,
    max_candidates: int = 3,
    max_memory_results: int = 8,
) -> WorkflowResult:
    """Parse a natural-language request, resolve, plan, and recommend next actions.

    Governance:
        approval_required   = True  (always)
        execution_performed = False (always)
        draft_created       = False (always)
        Execution/approval keywords in request → warning prepended, never acted on.
        save_draft is never inferred from natural language.

    Returns WorkflowResult with status:
        recommended         — actionable candidates produced
        needs_more_evidence — memory was empty; baseline recommended
        needs_clarification — intent parser could not extract enough intent
        needs_user_input    — resolver needs more information from the user
        invalid             — spec failed validation
        error               — unexpected exception
    """
    from src.agent.intent.intent_parser import parse_research_intent

    _s = resolve_llm_settings(provider, base_url, model)
    exec_warning = _check_execution_keywords(request)

    try:
        parse_result = parse_research_intent(
            user_request=request,
            provider=_s.provider,
            model=_s.model,
            base_url=_s.base_url,
            llm_client=llm_client,
            mode=mode,
        )
    except Exception as exc:
        wr = WorkflowResult(
            status="error",
            summary=f"Intent parsing failed: {exc}",
            errors=[str(exc)],
            warnings=[exec_warning] if exec_warning else [],
            approval_required=True,
            execution_performed=False,
            draft_approved=False,
        )
        return wr

    parse_dict = _make_parse_dict(parse_result)

    if parse_result.status != "parsed" or parse_result.research_spec is None:
        wr = _make_parse_error_result(parse_result)
        _prepend_warning(wr, exec_warning)
        wr.parse = parse_dict
        return wr

    # Delegate to the structured path.
    wr = run_recommend_next_actions(
        objective=None,
        spec=parse_result.research_spec,
        mode=mode,
        max_candidates=max_candidates,
        max_memory_results=max_memory_results,
    )

    # Attach parse metadata and execution-keyword warning.
    wr.parse = parse_dict
    _prepend_warning(wr, exec_warning)
    _merge_parse_warnings(wr, parse_result)

    return wr
