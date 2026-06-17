"""LLM provider/model/base_url resolution helper.

Resolution order for each setting:
  1. Explicit argument (wins when not None or empty string)
  2. ZETO_AGENT_LLM_* environment variable
  3. Safe fallback: provider="stub", base_url=None, model=None

Environment variables:
  ZETO_AGENT_LLM_PROVIDER  — e.g. "openai"
  ZETO_AGENT_LLM_BASE_URL  — e.g. "http://127.0.0.1:1234/v1"
  ZETO_AGENT_LLM_MODEL     — e.g. "qwen2.5-7b-instruct"

This lets users configure a local LM Studio endpoint once in their MCP server
environment rather than passing provider/base_url/model on every tool call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_ENV_PROVIDER = "ZETO_AGENT_LLM_PROVIDER"
_ENV_BASE_URL = "ZETO_AGENT_LLM_BASE_URL"
_ENV_MODEL = "ZETO_AGENT_LLM_MODEL"

_FALLBACK_PROVIDER = "stub"


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    base_url: str | None
    model: str | None


def resolve_llm_settings(
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> LLMSettings:
    """Resolve LLM settings using explicit-arg → env-var → stub-fallback priority.

    An explicit argument (non-None, non-empty string) always wins.
    When omitted, the corresponding ZETO_AGENT_LLM_* env var is used.
    If neither is set, provider falls back to "stub" and base_url/model to None.
    """
    resolved_provider: str = (
        provider
        if provider
        else os.environ.get(_ENV_PROVIDER) or _FALLBACK_PROVIDER
    )
    resolved_base_url: str | None = (
        base_url
        if base_url
        else os.environ.get(_ENV_BASE_URL) or None
    )
    resolved_model: str | None = (
        model
        if model
        else os.environ.get(_ENV_MODEL) or None
    )
    return LLMSettings(
        provider=resolved_provider,
        base_url=resolved_base_url,
        model=resolved_model,
    )
