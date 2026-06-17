"""Registry-constrained natural-language intent parser (Phase 4A).

Converts a user's natural-language research request into a ResearchSpec using
an LLM with registry-constrained prompting.  The output is a ResearchSpec only
— no plan, no draft, no approval, no execution.

Governance:
  - The LLM is an intent extractor.  The registry is the source of truth.
  - The deterministic resolver/critic/planner/draft pipeline is unchanged.
  - parse_research_intent() never calls the resolver, planner, or drafter.
  - parse_and_resolve_research_intent() calls resolve only — no plan or draft.
  - No API keys are required in tests (use provider="stub" or pass llm_client).
"""

from __future__ import annotations

import re
from typing import Any, Callable

from src.agent.intent.intent_result import IntentParseResult
from src.agent.intent.prompt_builder import build_intent_parser_prompt
from src.agent.intent.registry_context import build_registry_context
from src.agent.intent.research_spec import ResearchSpec
from src.agent.intent.response_parser import parse_intent_response
from src.agent.registry.registry_loader import load_registry
from src.agent.registry.registry_schema import ResearchConfigRegistry
from src.orchestration.llm.review_schema import (
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    PROVIDER_STUB,
)

_REAL_PROVIDERS: frozenset[str] = frozenset({PROVIDER_ANTHROPIC, PROVIDER_OPENAI})


def parse_research_intent(
    user_request: str,
    registry: ResearchConfigRegistry | None = None,
    provider: str = PROVIDER_STUB,
    model: str | None = None,
    base_url: str | None = None,
    llm_client: Callable[[str], str] | None = None,
    mode: str = "recommended",
) -> IntentParseResult:
    """Parse a natural-language research request into a ResearchSpec.

    Args:
        user_request: Free-form user text describing a research goal.
        registry:     Registry to constrain valid values.  Loaded from disk
                      if not provided.
        provider:     LLM provider: "stub" | "anthropic" | "openai".
                      Defaults to "stub" (no network call, keyword fallback).
        model:        Provider model ID override.
        base_url:     Override base URL for OpenAI-compatible endpoints.
        llm_client:   Optional callable ``(prompt: str) -> str`` that overrides
                      the built-in provider.  Useful for testing.
        mode:         Resolution mode hint (not used by parser itself; passed
                      through for parse_and_resolve_research_intent).

    Returns:
        IntentParseResult with status "parsed", "needs_clarification",
        "invalid_output", or "parser_error".

    Governance:
        The parser NEVER calls the resolver, planner, drafter, or any executor.
        It only calls an LLM (or the keyword fallback) and returns a ResearchSpec.
    """
    if not isinstance(user_request, str):
        return IntentParseResult(
            status="parser_error",
            raw_text=str(user_request),
            errors=["user_request must be a string."],
            summary="Invalid input: user_request is not a string.",
        )

    if not user_request.strip():
        return IntentParseResult(
            status="needs_clarification",
            raw_text=user_request,
            summary="Empty request — please describe your research goal.",
        )

    try:
        if registry is None:
            registry = load_registry()
    except Exception as exc:
        return IntentParseResult(
            status="parser_error",
            raw_text=user_request,
            errors=[f"Registry load failed: {exc}"],
            summary=f"Could not load registry: {exc}",
        )

    registry_context = build_registry_context(registry, compact=True)
    prompt = build_intent_parser_prompt(user_request, registry_context)

    # Obtain raw model output.
    raw_output: str
    if llm_client is not None:
        try:
            raw_output = llm_client(prompt)
        except Exception as exc:
            return IntentParseResult(
                status="parser_error",
                raw_text=user_request,
                errors=[f"llm_client raised: {exc}"],
                summary=f"LLM client call failed: {exc}",
            )
    elif provider == PROVIDER_STUB:
        # Keyword-based deterministic fallback — no network call, no API key needed.
        return _parse_via_keyword_fallback(user_request, registry, prompt)
    else:
        # Real provider via existing call_llm infrastructure.
        from src.orchestration.llm.llm_interface import call_llm
        try:
            response = call_llm(
                prompt=prompt,
                provider=provider,
                model=model,
                base_url=base_url,
                temperature=0.1,
                max_tokens=1024,
            )
            raw_output = response.text
        except Exception as exc:
            return IntentParseResult(
                status="parser_error",
                raw_text=user_request,
                errors=[str(exc)],
                summary=f"LLM call failed: {exc}",
            )

    result = parse_intent_response(raw_output, registry)
    result.raw_text = user_request
    return result


def parse_and_resolve_research_intent(
    user_request: str,
    registry: ResearchConfigRegistry | None = None,
    provider: str = PROVIDER_STUB,
    model: str | None = None,
    base_url: str | None = None,
    llm_client: Callable[[str], str] | None = None,
    mode: str = "recommended",
) -> Any:  # returns WorkflowResult
    """Parse a natural-language request → ResearchSpec → resolve.

    Calls parse_research_intent(), then if status == "parsed", passes the
    ResearchSpec through run_resolve_spec().

    This helper never plans, drafts, saves, approves, or executes.

    Returns:
        WorkflowResult.  If parsing fails, returns a WorkflowResult with
        status="error" and the parse errors.
    """
    from src.agent.workflow.research_workflow import run_resolve_spec

    parse_result = parse_research_intent(
        user_request=user_request,
        registry=registry,
        provider=provider,
        model=model,
        base_url=base_url,
        llm_client=llm_client,
        mode=mode,
    )

    if parse_result.status != "parsed" or parse_result.research_spec is None:
        from src.agent.workflow.workflow_result import WorkflowResult
        # needs_clarification → normal advisory status; other failures → error
        wf_status = (
            "needs_clarification"
            if parse_result.status == "needs_clarification"
            else "error"
        )
        return WorkflowResult(
            status=wf_status,
            summary=(
                f"Intent parsing ({parse_result.status}): "
                f"{parse_result.summary}"
            ),
            errors=list(parse_result.errors) + list(parse_result.registry_warnings),
            warnings=list(parse_result.warnings),
        )

    spec_dict = {
        k: v for k, v in parse_result.research_spec.to_dict().items()
        if v is not None
    }
    return run_resolve_spec(spec_dict, mode=mode)


# ---------------------------------------------------------------------------
# Keyword-based fallback (PROVIDER_STUB path)
# ---------------------------------------------------------------------------


_CAMEL_MODEL_SKIP = frozenset({"Regression", "Classification", "Classifier", "Model"})


def _normalize_for_match(s: str) -> str:
    """Lowercase and collapse hyphens/underscores/extra spaces to single space."""
    return re.sub(r"[-_\s]+", " ", s.lower()).strip()


def _parse_via_keyword_fallback(
    user_request: str,
    registry: ResearchConfigRegistry,
    prompt: str,
) -> IntentParseResult:
    """Deterministic keyword scanner for stub provider.

    Scans the user request for:
      - Preset names (normalised: hyphens/underscores → spaces)
      - Model names (exact + CamelCase prefix matching, e.g. "Lasso" → LassoRegression)
      - 4-digit year ranges ("2015 to 2024") or ISO dates
      - Transaction costs ("<n> bps")

    This is a convenience for tests and demos.  It does not call any LLM.
    """
    text = user_request
    norm_text = _normalize_for_match(text)
    text_lower = text.lower()
    extracted: dict[str, Any] = {}
    unrecognised: list[str] = []
    warnings: list[str] = ["Used keyword fallback (stub provider). Results may be incomplete."]

    # Presets — normalise underscores/hyphens to spaces for matching.
    matched_presets: list[str] = []
    for name in sorted(registry.preset_names()):
        if name in text or _normalize_for_match(name) in norm_text:
            matched_presets.append(name)

    if len(matched_presets) > 1:
        warnings.append(
            f"Ambiguous request: multiple presets detected {matched_presets}. "
            f"Using '{matched_presets[0]}'. Specify one preset for unambiguous results."
        )
    if matched_presets:
        extracted["preset"] = matched_presets[0]

    # Models — exact case-insensitive, then CamelCase prefix matching.
    # Collect ALL matches to detect ambiguity.
    matched_models: list[str] = []
    for name in sorted(registry.model_names()):
        if name.lower() in text_lower:
            if name not in matched_models:
                matched_models.append(name)
            continue
        # e.g. "LassoRegression" → try "lasso" in text
        camel_parts = re.findall(r"[A-Z][a-z]+", name)
        for part in camel_parts:
            if part not in _CAMEL_MODEL_SKIP and part.lower() in text_lower:
                if name not in matched_models:
                    matched_models.append(name)
                break

    if len(matched_models) > 1:
        warnings.append(
            f"Ambiguous request: multiple models detected {matched_models}. "
            f"Using '{matched_models[0]}'. Specify one model for unambiguous results."
        )
    if matched_models:
        extracted["model"] = matched_models[0]

    # Signals — exact case-insensitive.
    for name in sorted(registry.signal_names()):
        if name.lower() in text_lower or _normalize_for_match(name) in norm_text:
            extracted["signal"] = name
            break

    # Labels — exact case-insensitive.
    for name in sorted(registry.label_names()):
        if name.lower() in text_lower or _normalize_for_match(name) in norm_text:
            extracted["label"] = name
            break

    # Date extraction: ISO dates take priority over bare years.
    iso_dates = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if len(iso_dates) >= 2:
        extracted["start_date"] = iso_dates[0]
        extracted["end_date"] = iso_dates[-1]
    elif len(iso_dates) == 1:
        extracted["end_date"] = iso_dates[0]
    else:
        years = re.findall(r"\b(20\d{2})\b", text)
        if len(years) >= 2:
            extracted["start_date"] = f"{years[0]}-01-01"
            extracted["end_date"] = f"{years[-1]}-12-31"
        elif len(years) == 1:
            extracted["end_date"] = f"{years[0]}-12-31"

    # Transaction costs: "<n> bps" or "<n>bps".
    bps_match = re.search(r"(\d+(?:\.\d+)?)\s*bps", text_lower)
    if bps_match:
        try:
            extracted["transaction_cost_bps"] = float(bps_match.group(1))
        except ValueError:
            pass

    stub_output = "[keyword-fallback — stub provider, no LLM call]"

    if not extracted:
        return IntentParseResult(
            status="needs_clarification",
            research_spec=ResearchSpec(),
            raw_text=user_request,
            raw_model_output=stub_output,
            extracted_fields={},
            unrecognised_terms=unrecognised,
            confidence=0.0,
            warnings=warnings,
            summary=(
                "Keyword fallback found no matching registry values. "
                "Provide more detail or use a real LLM provider."
            ),
        )

    spec = ResearchSpec.from_dict(extracted)
    return IntentParseResult(
        status="parsed",
        research_spec=spec,
        raw_text=user_request,
        raw_model_output=stub_output,
        extracted_fields=dict(extracted),
        unrecognised_terms=unrecognised,
        confidence=0.5,
        warnings=warnings,
        summary=f"Keyword fallback extracted: {sorted(extracted.keys())}.",
    )
