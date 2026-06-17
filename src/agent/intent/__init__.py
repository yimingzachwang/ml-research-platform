"""Agent intent layer — ResearchSpec and natural-language intent parsing (Phase 4A).

Public exports:
    ResearchSpec                 — partial research intent dataclass
    IntentParseResult            — parse result envelope
    parse_research_intent()      — NL request → IntentParseResult
    parse_and_resolve_research_intent() — NL request → WorkflowResult (resolve only)
    build_registry_context()     — registry → compact prompt context dict
    build_intent_parser_prompt() — user request + context → LLM prompt string
    parse_intent_response()      — raw model output → IntentParseResult
"""

from src.agent.intent.intent_parser import (
    parse_and_resolve_research_intent,
    parse_research_intent,
)
from src.agent.intent.intent_result import IntentParseResult
from src.agent.intent.prompt_builder import build_intent_parser_prompt
from src.agent.intent.registry_context import build_registry_context
from src.agent.intent.research_spec import ResearchSpec
from src.agent.intent.response_parser import parse_intent_response

__all__ = [
    "ResearchSpec",
    "IntentParseResult",
    "parse_research_intent",
    "parse_and_resolve_research_intent",
    "build_registry_context",
    "build_intent_parser_prompt",
    "parse_intent_response",
]
