"""Prompt builder for the registry-constrained intent parser.

Constructs the LLM prompt that instructs the model to extract a ResearchSpec
from a natural-language research request, constrained to registry values only.
"""

from __future__ import annotations

import json
from typing import Any

# The exact JSON shape the model must output.
_OUTPUT_SCHEMA = {
    "research_spec": {
        "preset": None,
        "objective": None,
        "asset_class": None,
        "universe": None,
        "tickers": None,
        "start_date": None,
        "end_date": None,
        "frequency": None,
        "data_source": None,
        "feature_families": None,
        "features": None,
        "label": None,
        "label_horizon": None,
        "model": None,
        "model_params": None,
        "signal": None,
        "signal_params": None,
        "validation_protocol": None,
        "train_window": None,
        "test_window": None,
        "transaction_cost_bps": None,
        "benchmark": None,
        "diagnostics": None,
        "risk_constraints": None,
        "notes": None,
    },
    "unrecognised_terms": [],
    "confidence": 0.0,
    "summary": "",
}


def build_intent_parser_prompt(
    user_request: str,
    registry_context: dict[str, Any],
) -> str:
    """Build the LLM prompt for registry-constrained intent extraction.

    Args:
        user_request:     The user's natural-language research request.
        registry_context: Output of build_registry_context().

    Returns:
        A complete prompt string suitable for passing to call_llm().
    """
    context_section = _format_registry_context(registry_context)
    schema_json = json.dumps(_OUTPUT_SCHEMA, indent=2)
    return _PROMPT_TEMPLATE.format(
        registry_context=context_section,
        output_schema=schema_json,
        user_request=user_request,
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are a structured research intent extractor for a quantitative research platform.

Your ONLY task: extract the user's research intent as a structured JSON object.

══════════════════════════════════════════════════════════════
STRICT RULES — READ CAREFULLY
══════════════════════════════════════════════════════════════

1. OUTPUT STRICT JSON ONLY.
   - Do not output prose, markdown, YAML, or code blocks.
   - Do not wrap the JSON in ```json ... ``` or any other wrapper.
   - The very first character of your response must be {{ and the very last must be }}.

2. USE ONLY REGISTRY-SUPPORTED VALUES.
   - Only use model names, preset names, feature names, label names,
     signal names, universe names, asset class names, and validation protocol
     names that appear in the SUPPORTED REGISTRY VALUES section below.
   - If a value the user mentions is NOT in the registry, do NOT include it
     in research_spec. Instead, add the unrecognised term to "unrecognised_terms".

3. SET UNKNOWN OR UNCONFIDENT FIELDS TO null.
   - If you cannot confidently extract a field, set it to null.
   - Do not invent plausible-sounding values.

4. FORBIDDEN ACTIONS — NEVER DO THESE:
   - Do NOT produce YAML.
   - Do NOT produce an experiment config.
   - Do NOT recommend executing, running, or approving anything.
   - Do NOT claim a strategy will work or outperform.
   - Do NOT add fields not listed in the output schema.

5. PRESERVE USER-SPECIFIED VALUES:
   - Preserve exact dates, model names, presets, transaction costs,
     and constraints if they appear in the user's request AND are in the registry.
   - If a date like "2015 to 2024" is mentioned, set start_date="2015-01-01"
     and end_date="2024-12-31" (or the exact date string if one is given).
   - If transaction costs like "5 bps" are mentioned, set transaction_cost_bps=5.0.

══════════════════════════════════════════════════════════════
SUPPORTED REGISTRY VALUES
══════════════════════════════════════════════════════════════

{registry_context}

══════════════════════════════════════════════════════════════
REQUIRED OUTPUT SHAPE
══════════════════════════════════════════════════════════════

Output exactly this JSON shape (fill in values; leave unsupported/unknown as null):

{output_schema}

Field guidance:
- preset: one of the supported preset names, or null
- model: one of the supported model names, or null
- asset_class: one of the supported asset class names, or null
- universe: one of the supported universe names, or null
- label: one of the supported label names, or null
- signal: one of the supported signal names, or null
- validation_protocol: one of the supported validation protocol names, or null
- features: list of supported feature names, or null
- feature_families: list of supported feature family names, or null
- diagnostics: list of supported diagnostic names, or null
- start_date / end_date: ISO date strings (YYYY-MM-DD), or null
- transaction_cost_bps: numeric value (e.g. 5.0), or null
- label_horizon: integer (number of trading days), or null
- train_window / test_window: string like "48m" or "12m", or null
- model_params / signal_params / risk_constraints: dict or null
- tickers: list of ticker strings, or null
- objective / benchmark / frequency / data_source / notes: string or null
- confidence: float 0.0–1.0 reflecting your certainty in the extraction
- summary: one-sentence summary of what was extracted
- unrecognised_terms: list of terms from the request you could not map to
  any registry value

══════════════════════════════════════════════════════════════
USER REQUEST
══════════════════════════════════════════════════════════════

{user_request}

══════════════════════════════════════════════════════════════
YOUR RESPONSE (strict JSON only, starting with {{):
══════════════════════════════════════════════════════════════\
"""


def _format_registry_context(ctx: dict[str, Any]) -> str:
    """Format the registry context for prompt embedding."""
    lines: list[str] = []

    def _names_line(label: str, names: list[str]) -> str:
        if not names:
            return f"{label}: (none)"
        return f"{label}: {', '.join(names)}"

    lines.append(_names_line("Presets", ctx.get("presets", [])))
    lines.append(_names_line("Asset classes", ctx.get("asset_classes", [])))
    lines.append(_names_line("Universes", ctx.get("universes", [])))
    lines.append(_names_line("Models", ctx.get("models", [])))
    lines.append(_names_line("Features", ctx.get("features", [])))
    lines.append(_names_line("Feature families", ctx.get("feature_families", [])))
    lines.append(_names_line("Labels", ctx.get("labels", [])))
    lines.append(_names_line("Signals", ctx.get("signals", [])))
    lines.append(_names_line("Validation protocols", ctx.get("validation_protocols", [])))
    lines.append(_names_line("Diagnostics", ctx.get("diagnostics", [])))

    default_mode = ctx.get("default_mode", "recommended")
    lines.append(f"Default resolution mode: {default_mode}")

    return "\n".join(lines)
