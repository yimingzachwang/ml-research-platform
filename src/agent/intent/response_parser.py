"""Response parser for the registry-constrained intent parser.

Parses the LLM's JSON output into an IntentParseResult. Enforces registry
constraints: unsupported values are rejected and placed in unrecognised_terms
or registry_warnings, never silently passed through to ResearchSpec.
"""

from __future__ import annotations

import json
import re
from dataclasses import fields as dataclass_fields
from typing import Any

from src.agent.intent.intent_result import IntentParseResult
from src.agent.intent.research_spec import ResearchSpec
from src.agent.registry.registry_schema import ResearchConfigRegistry

# ResearchSpec field names (constant — computed once)
_SPEC_FIELDS: frozenset[str] = frozenset(f.name for f in dataclass_fields(ResearchSpec))

# Fields whose values must be validated against registry name sets.
# Maps field name → method name on ResearchConfigRegistry that returns the valid names.
_REGISTRY_VALIDATED_FIELDS: dict[str, str] = {
    "preset": "preset_names",
    "asset_class": "asset_class_names",
    "universe": "universe_names",
    "model": "model_names",
    "label": "label_names",
    "signal": "signal_names",
    "validation_protocol": "validation_names",
}

# List fields whose individual elements are registry-validated.
_REGISTRY_VALIDATED_LIST_FIELDS: dict[str, str] = {
    "features": "feature_names",
    "diagnostics": "diagnostic_names",
}


def parse_intent_response(
    model_output: str,
    registry: ResearchConfigRegistry,
) -> IntentParseResult:
    """Parse the LLM's JSON output into an IntentParseResult.

    This function:
    - Parses strict JSON (with optional markdown code-fence stripping)
    - Rejects non-dict or malformed output → "invalid_output"
    - Filters spec fields to only known ResearchSpec field names
    - Validates registry-constrained fields and moves unsupported values to
      unrecognised_terms / registry_warnings (never into ResearchSpec)
    - Preserves raw output for debugging

    Args:
        model_output: Raw text returned by the LLM.
        registry:     Loaded ResearchConfigRegistry for validation.

    Returns:
        IntentParseResult with status "parsed", "needs_clarification",
        "invalid_output", or "parser_error".
    """
    raw = model_output

    # 1. Parse JSON — try direct first, then strip markdown code fences.
    data, parse_error = _try_parse_json(raw)
    if data is None:
        return IntentParseResult(
            status="invalid_output",
            raw_model_output=raw,
            errors=[f"Model output is not valid JSON: {parse_error}"],
            summary="Model output could not be parsed as JSON.",
        )

    # 2. Must be a dict.
    if not isinstance(data, dict):
        return IntentParseResult(
            status="invalid_output",
            raw_model_output=raw,
            errors=[f"Model output is JSON but not an object (got {type(data).__name__})."],
            summary="Model output must be a JSON object.",
        )

    # 3. Extract research_spec sub-object.
    spec_raw = data.get("research_spec")
    if spec_raw is not None and not isinstance(spec_raw, dict):
        return IntentParseResult(
            status="invalid_output",
            raw_model_output=raw,
            errors=["'research_spec' must be a JSON object."],
            summary="research_spec field is not an object.",
        )
    if spec_raw is None:
        spec_raw = {}

    # 4. Validate and filter spec fields.
    validated: dict[str, Any] = {}
    registry_warnings: list[str] = []
    unrecognised_terms: list[str] = list(data.get("unrecognised_terms", []) or [])
    unknown_spec_fields: list[str] = []

    for key, value in spec_raw.items():
        # Skip null values — they are not "specified".
        if value is None:
            continue

        # Reject unknown field names.
        if key not in _SPEC_FIELDS:
            unknown_spec_fields.append(key)
            continue

        # Registry validation for scalar fields.
        if key in _REGISTRY_VALIDATED_FIELDS:
            method_name = _REGISTRY_VALIDATED_FIELDS[key]
            valid_names: frozenset[str] = getattr(registry, method_name)()
            if isinstance(value, str) and value not in valid_names:
                registry_warnings.append(
                    f"'{value}' is not a supported {key} value. "
                    f"Supported: {sorted(valid_names)}. Skipped."
                )
                unrecognised_terms.append(value)
                continue

        # Registry validation for list fields.
        if key in _REGISTRY_VALIDATED_LIST_FIELDS and isinstance(value, list):
            method_name = _REGISTRY_VALIDATED_LIST_FIELDS[key]
            valid_names = getattr(registry, method_name)()
            accepted = []
            for item in value:
                if isinstance(item, str) and item not in valid_names:
                    registry_warnings.append(
                        f"'{item}' is not a supported {key} value. Skipped."
                    )
                    unrecognised_terms.append(item)
                else:
                    accepted.append(item)
            if not accepted:
                continue  # all items rejected — skip field entirely
            value = accepted

        # feature_families: validate against registry families
        if key == "feature_families" and isinstance(value, list):
            from src.agent.intent.registry_context import _extract_feature_families
            valid_families = set(_extract_feature_families(registry))
            accepted_fams = []
            for fam in value:
                if isinstance(fam, str) and fam not in valid_families:
                    registry_warnings.append(
                        f"'{fam}' is not a supported feature_family. Skipped."
                    )
                    unrecognised_terms.append(fam)
                else:
                    accepted_fams.append(fam)
            if not accepted_fams:
                continue
            value = accepted_fams

        validated[key] = value

    # 5. Deduplicate unrecognised_terms.
    seen: set[str] = set()
    deduped_unrec: list[str] = []
    for t in unrecognised_terms:
        if t not in seen:
            seen.add(t)
            deduped_unrec.append(t)
    unrecognised_terms = deduped_unrec

    # 6. Build warnings list.
    warnings: list[str] = []
    for f in unknown_spec_fields:
        warnings.append(f"Unknown field '{f}' in model output ignored.")

    # 7. Create ResearchSpec.
    try:
        research_spec = ResearchSpec.from_dict(validated)
    except Exception as exc:
        return IntentParseResult(
            status="invalid_output",
            raw_model_output=raw,
            errors=[f"Could not construct ResearchSpec: {exc}"],
            summary="ResearchSpec construction failed.",
        )

    # 8. Determine status.
    confidence = _safe_float(data.get("confidence"))
    summary = str(data.get("summary") or "")

    if not validated:
        status = "needs_clarification"
    else:
        status = "parsed"

    # 9. Identify important fields still missing after parse.
    missing_after_parse = [
        f for f in ("preset", "model", "end_date")
        if f not in validated
    ]

    return IntentParseResult(
        status=status,
        research_spec=research_spec,
        raw_model_output=raw,
        extracted_fields=dict(validated),
        unrecognised_terms=unrecognised_terms,
        registry_warnings=registry_warnings,
        missing_after_parse=missing_after_parse,
        confidence=confidence,
        warnings=warnings,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_parse_json(text: str) -> tuple[Any | None, str]:
    """Try to parse text as JSON. Returns (data, error_message)."""
    # Direct parse first.
    try:
        return json.loads(text), ""
    except json.JSONDecodeError:
        pass

    # Strip common markdown code fences (```json ... ``` or ``` ... ```).
    stripped = _strip_code_fences(text)
    if stripped != text:
        try:
            return json.loads(stripped), ""
        except json.JSONDecodeError:
            pass

    # Last resort: find the first { ... } block.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), ""
        except json.JSONDecodeError:
            pass

    return None, f"Could not parse JSON from: {text[:120]!r}"


def _strip_code_fences(text: str) -> str:
    """Remove leading ```json or ``` and trailing ``` fences."""
    text = text.strip()
    # Remove opening fence.
    if text.startswith("```"):
        newline = text.find("\n")
        if newline != -1:
            text = text[newline + 1:]
    # Remove trailing fence.
    if text.endswith("```"):
        text = text[: text.rfind("```")].rstrip()
    return text


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
