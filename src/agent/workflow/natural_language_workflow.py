"""Natural-language research workflow façade (Phase 4B).

Connects the Phase 4A intent parser to the deterministic agent pipeline:

    NL request
    → parse_research_intent()   (Phase 4A — registry-constrained LLM extraction)
    → run_resolve_spec()         (deterministic resolver + critic)
    → run_plan_from_spec()       (deterministic planner)
    → run_draft_from_spec()      (deterministic drafter + validator)
    → save_agent_draft_result()  (optional explicit persistence, Phase 3D)

Governance (invariants this module enforces):
  - The LLM is an intent extractor only; all research decisions are deterministic.
  - save_draft=True must be passed EXPLICITLY — never inferred from NL request.
  - No approval. No execution. No YAML writes.
  - Execution/approval keywords detected in request → warning added, not acted on.
  - needs_user_input from the resolver is a normal structured status, not an error.
  - needs_clarification from the parser is a normal structured status, not an error.
  - Parse metadata (unrecognised_terms, confidence, registry_warnings) always
    propagated to WorkflowResult.parse.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from src.agent.intent.intent_parser import parse_research_intent
from src.agent.intent.intent_result import IntentParseResult
from src.agent.workflow.draft_persistence import save_agent_draft_result
from src.agent.workflow.research_workflow import (
    run_draft_from_spec,
    run_plan_from_spec,
    run_resolve_spec,
)
from src.agent.intent.llm_settings import resolve_llm_settings
from src.agent.workflow.workflow_result import WorkflowResult
from src.orchestration.llm.review_schema import PROVIDER_STUB  # noqa: F401 — re-exported

# ---------------------------------------------------------------------------
# Execution-intent detection
# ---------------------------------------------------------------------------

_EXEC_KEYWORD_SET: frozenset[str] = frozenset({
    "run", "execute", "approve", "trade", "deploy", "live", "launch",
})

_EXEC_WARNING_TEMPLATE = (
    "Execution or approval was requested in natural language "
    "(detected: {found}), but this workflow does not execute or approve "
    "experiments. A draft can only be saved as unapproved when "
    "save_draft=True is explicitly passed."
)


def _check_execution_keywords(request: str) -> str | None:
    """Return a warning string if execution/approval keywords are detected; else None."""
    words = {re.sub(r"[.,!?;:]", "", w) for w in request.lower().split()}
    found = sorted(words & _EXEC_KEYWORD_SET)
    if found:
        return _EXEC_WARNING_TEMPLATE.format(found=found)
    return None


# ---------------------------------------------------------------------------
# Public workflow functions
# ---------------------------------------------------------------------------


def run_resolve_from_request(
    request: str,
    provider: str | None = None,
    mode: str = "recommended",
    model: str | None = None,
    base_url: str | None = None,
    llm_client: Callable[[str], str] | None = None,
    registry: Any = None,
) -> WorkflowResult:
    """NL request → parse → resolve.

    Stops at resolution.  No plan, no draft, no save, no execution.
    Governance flags are always present in the result.

    Returns:
        WorkflowResult.  Possible statuses:
          - resolved         — spec resolved successfully
          - needs_user_input — resolver needs more information from the user
          - needs_clarification — parser could not extract enough intent
          - invalid          — resolver rejected the spec (compatibility error)
          - error            — parser or resolver encountered an unexpected error
    """
    _s = resolve_llm_settings(provider, base_url, model)
    exec_warning = _check_execution_keywords(request)

    parse_result = parse_research_intent(
        user_request=request,
        registry=registry,
        provider=_s.provider,
        model=_s.model,
        base_url=_s.base_url,
        llm_client=llm_client,
        mode=mode,
    )

    if parse_result.status != "parsed" or parse_result.research_spec is None:
        wr = _make_parse_error_result(parse_result)
        _prepend_warning(wr, exec_warning)
        return wr

    spec_dict = _spec_to_dict(parse_result.research_spec)
    result = run_resolve_spec(spec_dict, mode=mode)

    result.parse = _make_parse_dict(parse_result)
    _prepend_warning(result, exec_warning)
    _merge_parse_warnings(result, parse_result)

    return result


def run_plan_from_request(
    request: str,
    provider: str | None = None,
    mode: str = "recommended",
    baseline: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    llm_client: Callable[[str], str] | None = None,
    registry: Any = None,
) -> WorkflowResult:
    """NL request → parse → resolve → plan.

    No draft, no save, no execution.

    Returns:
        WorkflowResult.  Possible statuses:
          - planned          — experiment plan created
          - needs_review     — plan created with warnings
          - needs_user_input — resolver needs more information
          - needs_clarification — parser could not extract enough intent
          - invalid          — resolver or plan critic rejected the spec
          - error            — unexpected error
    """
    _s = resolve_llm_settings(provider, base_url, model)
    exec_warning = _check_execution_keywords(request)

    parse_result = parse_research_intent(
        user_request=request,
        registry=registry,
        provider=_s.provider,
        model=_s.model,
        base_url=_s.base_url,
        llm_client=llm_client,
        mode=mode,
    )

    if parse_result.status != "parsed" or parse_result.research_spec is None:
        wr = _make_parse_error_result(parse_result)
        _prepend_warning(wr, exec_warning)
        return wr

    spec_dict = _spec_to_dict(parse_result.research_spec)
    result = run_plan_from_spec(spec_dict, mode=mode, baseline=baseline)

    result.parse = _make_parse_dict(parse_result)
    _prepend_warning(result, exec_warning)
    _merge_parse_warnings(result, parse_result)

    return result


def run_draft_from_request(
    request: str,
    provider: str | None = None,
    mode: str = "recommended",
    baseline: str | None = None,
    base_experiment: str | None = None,
    save_draft: bool = False,
    model: str | None = None,
    base_url: str | None = None,
    llm_client: Callable[[str], str] | None = None,
    registry: Any = None,
) -> WorkflowResult:
    """NL request → parse → resolve → plan → validated draft.

    save_draft must be EXPLICITLY True to persist.  Never approves, never executes.

    Governance:
      - approval_required: True (always)
      - execution_performed: False (always)
      - draft_approved: False (always)
      - save_draft is never inferred from the NL request

    Returns:
        WorkflowResult.  Possible statuses:
          - draft_created    — validated draft ready for human review
          - needs_user_input — resolver needs more information
          - needs_clarification — parser could not extract enough intent
          - unsupported_plan — hard-unsupported field blocks draft creation
          - validation_failed — draft assembled but validation rejected it
          - invalid / error  — upstream failure
        When save_draft=True and status == draft_created, result.saved_draft
        contains the SavedAgentDraftResult dict.
    """
    _s = resolve_llm_settings(provider, base_url, model)
    exec_warning = _check_execution_keywords(request)

    parse_result = parse_research_intent(
        user_request=request,
        registry=registry,
        provider=_s.provider,
        model=_s.model,
        base_url=_s.base_url,
        llm_client=llm_client,
        mode=mode,
    )

    if parse_result.status != "parsed" or parse_result.research_spec is None:
        wr = _make_parse_error_result(parse_result)
        _prepend_warning(wr, exec_warning)
        return wr

    spec_dict = _spec_to_dict(parse_result.research_spec)
    result = run_draft_from_spec(
        spec_dict,
        mode=mode,
        baseline=baseline,
        base_experiment=base_experiment,
    )

    result.parse = _make_parse_dict(parse_result)
    _prepend_warning(result, exec_warning)
    _merge_parse_warnings(result, parse_result)

    # Optional persistence — only if explicitly requested and draft was created.
    if save_draft and result.status == "draft_created":
        persist = save_agent_draft_result(result)
        result.saved_draft = persist.to_dict()

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _spec_to_dict(spec: Any) -> dict[str, Any]:
    """Convert a ResearchSpec to a plain dict, dropping None values."""
    return {k: v for k, v in spec.to_dict().items() if v is not None}


def _make_parse_dict(parse_result: IntentParseResult) -> dict[str, Any]:
    """Serialise an IntentParseResult to a plain dict for WorkflowResult.parse."""
    return {
        "status": parse_result.status,
        "summary": parse_result.summary,
        "extracted_fields": dict(parse_result.extracted_fields),
        "unrecognised_terms": list(parse_result.unrecognised_terms),
        "registry_warnings": list(parse_result.registry_warnings),
        "missing_after_parse": list(parse_result.missing_after_parse),
        "confidence": parse_result.confidence,
        "warnings": list(parse_result.warnings),
        "raw_model_output": parse_result.raw_model_output,
    }


def _make_parse_error_result(parse_result: IntentParseResult) -> WorkflowResult:
    """Build a WorkflowResult for a failed or ambiguous parse."""
    status = (
        "needs_clarification"
        if parse_result.status == "needs_clarification"
        else "error"
    )
    return WorkflowResult(
        status=status,
        summary=f"Intent parsing: {parse_result.summary}",
        errors=list(parse_result.errors),
        warnings=list(parse_result.warnings),
        parse=_make_parse_dict(parse_result),
        execution_performed=False,
        draft_approved=False,
    )


def _prepend_warning(result: WorkflowResult, warning: str | None) -> None:
    """Insert an execution-keyword warning at the front of the warnings list."""
    if warning:
        result.warnings.insert(0, warning)


def _merge_parse_warnings(
    result: WorkflowResult, parse_result: IntentParseResult
) -> None:
    """Merge parse-level warnings (e.g. ambiguity) into the workflow result."""
    for w in parse_result.warnings:
        if w not in result.warnings:
            result.warnings.append(w)
