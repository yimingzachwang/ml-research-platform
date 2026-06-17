"""IntentParseResult — output envelope for the natural-language intent parser.

Phase 4A: returned by parse_research_intent() in intent_parser.py.
The LLM is an intent extractor only — the result contains a ResearchSpec (or
None), never a draft, never an approved config, never an execution instruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.intent.research_spec import ResearchSpec


@dataclass
class IntentParseResult:
    """Envelope returned by the natural-language intent parser.

    status:
        "parsed"            — ResearchSpec extracted; may have warnings.
        "needs_clarification" — Request was intelligible but too ambiguous to
                                produce a useful ResearchSpec.  Ask the user
                                for more detail before resolving.
        "invalid_output"    — Model returned non-JSON, a non-object, or an
                                otherwise uninterpretable response.
        "parser_error"      — Unexpected exception before or during the LLM
                                call (missing API key, network failure, etc.).

    Governance guarantees (always present):
        no_plan_created       True  — parser never calls planner.
        no_draft_created      True  — parser never calls drafter.
        no_execution_performed True  — parser never executes anything.
    """

    status: str  # "parsed" | "needs_clarification" | "invalid_output" | "parser_error"

    # Extracted ResearchSpec — populated on status=="parsed" only.
    research_spec: ResearchSpec | None = None

    # Raw inputs / outputs — always preserved for debugging
    raw_text: str = ""
    raw_model_output: str | None = None

    # Parse provenance
    extracted_fields: dict[str, Any] = field(default_factory=dict)
    unrecognised_terms: list[str] = field(default_factory=list)
    registry_warnings: list[str] = field(default_factory=list)
    missing_after_parse: list[str] = field(default_factory=list)
    confidence: float | None = None

    # Advisory
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: str = ""

    # Governance flags — always present, always these values.
    no_plan_created: bool = True
    no_draft_created: bool = True
    no_execution_performed: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (MCP-safe)."""
        return {
            "status": self.status,
            "research_spec": self.research_spec.to_dict() if self.research_spec else None,
            "raw_text": self.raw_text,
            "raw_model_output": self.raw_model_output,
            "extracted_fields": dict(self.extracted_fields),
            "unrecognised_terms": list(self.unrecognised_terms),
            "registry_warnings": list(self.registry_warnings),
            "missing_after_parse": list(self.missing_after_parse),
            "confidence": self.confidence,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "summary": self.summary,
            "no_plan_created": self.no_plan_created,
            "no_draft_created": self.no_draft_created,
            "no_execution_performed": self.no_execution_performed,
        }
