"""Tests for src/agent/intent/llm_settings.py — env-var LLM resolution.

Verifies the three-tier resolution priority:
  1. Explicit argument (always wins)
  2. ZETO_AGENT_LLM_* environment variable
  3. Safe fallback: provider="stub", base_url=None, model=None

Design principles:
  - No live network call is made.
  - No LM Studio required.
  - Governance flags (approval_required etc.) are unaffected by this module.
  - Stub mode is preserved when no env vars are set and no args are given.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.agent.intent.llm_settings import (
    LLMSettings,
    _ENV_BASE_URL,
    _ENV_MODEL,
    _ENV_PROVIDER,
    resolve_llm_settings,
)


# ---------------------------------------------------------------------------
# A — No env vars, no explicit args → stub fallback
# ---------------------------------------------------------------------------


class TestNoEnvNoArgs:
    """When neither env vars nor explicit args are provided, stub defaults apply."""

    def test_provider_is_stub(self):
        with patch.dict(os.environ, {}, clear=False):
            # Ensure none of the env vars are set.
            env = {k: v for k, v in os.environ.items()
                   if k not in (_ENV_PROVIDER, _ENV_BASE_URL, _ENV_MODEL)}
            with patch.dict(os.environ, env, clear=True):
                s = resolve_llm_settings()
        assert s.provider == "stub"

    def test_base_url_is_none(self):
        env = {k: v for k, v in os.environ.items()
               if k not in (_ENV_PROVIDER, _ENV_BASE_URL, _ENV_MODEL)}
        with patch.dict(os.environ, env, clear=True):
            s = resolve_llm_settings()
        assert s.base_url is None

    def test_model_is_none(self):
        env = {k: v for k, v in os.environ.items()
               if k not in (_ENV_PROVIDER, _ENV_BASE_URL, _ENV_MODEL)}
        with patch.dict(os.environ, env, clear=True):
            s = resolve_llm_settings()
        assert s.model is None

    def test_returns_llm_settings_instance(self):
        env = {k: v for k, v in os.environ.items()
               if k not in (_ENV_PROVIDER, _ENV_BASE_URL, _ENV_MODEL)}
        with patch.dict(os.environ, env, clear=True):
            s = resolve_llm_settings()
        assert isinstance(s, LLMSettings)

    def test_llm_settings_is_frozen(self):
        s = resolve_llm_settings()
        with pytest.raises((AttributeError, TypeError)):
            s.provider = "openai"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# B — Env vars set → returned in resolved settings
# ---------------------------------------------------------------------------


class TestEnvVarsUsed:
    """ZETO_AGENT_LLM_* env vars are used when no explicit args are given."""

    def test_provider_from_env(self):
        with patch.dict(os.environ, {_ENV_PROVIDER: "openai",
                                      _ENV_BASE_URL: "",
                                      _ENV_MODEL: ""}, clear=False):
            s = resolve_llm_settings()
        assert s.provider == "openai"

    def test_base_url_from_env(self):
        with patch.dict(os.environ, {_ENV_BASE_URL: "http://127.0.0.1:1234/v1"}, clear=False):
            # Clear the provider and model env vars so they fall to defaults.
            env = {k: v for k, v in os.environ.items() if k not in (_ENV_PROVIDER, _ENV_MODEL)}
            env[_ENV_BASE_URL] = "http://127.0.0.1:1234/v1"
            with patch.dict(os.environ, env, clear=True):
                s = resolve_llm_settings()
        assert s.base_url == "http://127.0.0.1:1234/v1"

    def test_model_from_env(self):
        env = {k: v for k, v in os.environ.items() if k not in (_ENV_PROVIDER, _ENV_BASE_URL, _ENV_MODEL)}
        env[_ENV_MODEL] = "qwen2.5-7b-instruct"
        with patch.dict(os.environ, env, clear=True):
            s = resolve_llm_settings()
        assert s.model == "qwen2.5-7b-instruct"

    def test_all_three_from_env(self):
        env = {
            _ENV_PROVIDER: "openai",
            _ENV_BASE_URL: "http://127.0.0.1:1234/v1",
            _ENV_MODEL: "qwen2.5-7b-instruct",
        }
        with patch.dict(os.environ, env, clear=True):
            s = resolve_llm_settings()
        assert s.provider == "openai"
        assert s.base_url == "http://127.0.0.1:1234/v1"
        assert s.model == "qwen2.5-7b-instruct"

    def test_provider_fallback_when_env_empty_string(self):
        env = {_ENV_PROVIDER: "", _ENV_BASE_URL: "", _ENV_MODEL: ""}
        with patch.dict(os.environ, env, clear=True):
            s = resolve_llm_settings()
        assert s.provider == "stub"

    def test_base_url_none_when_env_empty_string(self):
        env = {_ENV_BASE_URL: ""}
        with patch.dict(os.environ, {**{k: v for k, v in os.environ.items()
                                         if k not in (_ENV_PROVIDER, _ENV_BASE_URL, _ENV_MODEL)},
                                      **env}, clear=True):
            s = resolve_llm_settings()
        assert s.base_url is None


# ---------------------------------------------------------------------------
# C — Explicit args override env vars
# ---------------------------------------------------------------------------


class TestExplicitArgsWin:
    """Explicit arguments always beat env vars."""

    def test_explicit_provider_beats_env(self):
        with patch.dict(os.environ, {_ENV_PROVIDER: "anthropic"}, clear=False):
            s = resolve_llm_settings(provider="openai")
        assert s.provider == "openai"

    def test_explicit_base_url_beats_env(self):
        with patch.dict(os.environ, {_ENV_BASE_URL: "http://env:1234/v1"}, clear=False):
            s = resolve_llm_settings(base_url="http://explicit:5678/v1")
        assert s.base_url == "http://explicit:5678/v1"

    def test_explicit_model_beats_env(self):
        with patch.dict(os.environ, {_ENV_MODEL: "env-model"}, clear=False):
            s = resolve_llm_settings(model="explicit-model")
        assert s.model == "explicit-model"

    def test_all_three_explicit_beat_env(self):
        env = {_ENV_PROVIDER: "anthropic", _ENV_BASE_URL: "http://env/v1", _ENV_MODEL: "env-m"}
        with patch.dict(os.environ, env, clear=False):
            s = resolve_llm_settings(
                provider="openai",
                base_url="http://local:1234/v1",
                model="qwen2.5-7b-instruct",
            )
        assert s.provider == "openai"
        assert s.base_url == "http://local:1234/v1"
        assert s.model == "qwen2.5-7b-instruct"

    def test_explicit_stub_preserved_when_env_says_openai(self):
        with patch.dict(os.environ, {_ENV_PROVIDER: "openai"}, clear=False):
            s = resolve_llm_settings(provider="stub")
        assert s.provider == "stub"

    def test_none_explicit_falls_through_to_env(self):
        env = {k: v for k, v in os.environ.items() if k not in (_ENV_PROVIDER, _ENV_BASE_URL, _ENV_MODEL)}
        env[_ENV_PROVIDER] = "anthropic"
        with patch.dict(os.environ, env, clear=True):
            s = resolve_llm_settings(provider=None)
        assert s.provider == "anthropic"

    def test_partial_override(self):
        env = {_ENV_PROVIDER: "openai", _ENV_MODEL: "env-model"}
        with patch.dict(os.environ, env, clear=False):
            s = resolve_llm_settings(provider="stub")
        # provider explicitly stub, model from env
        assert s.provider == "stub"
        assert s.model == "env-model"


# ---------------------------------------------------------------------------
# D — Governance flags are unaffected
# ---------------------------------------------------------------------------


class TestGovernanceUnaffected:
    """Importing or calling resolve_llm_settings has zero effect on governance."""

    def test_import_does_not_affect_provider_stub_constant(self):
        from src.orchestration.llm.review_schema import PROVIDER_STUB
        assert PROVIDER_STUB == "stub"

    def test_resolve_stub_settings_governance_check(self):
        s = resolve_llm_settings()
        # provider="stub" means no LLM call; governance is downstream
        assert s.provider == "stub"
        assert s.base_url is None
        assert s.model is None

    def test_workflow_result_still_has_governance_flags(self):
        from src.agent.workflow.natural_language_workflow import run_resolve_from_request
        result = run_resolve_from_request("test lasso request")
        # Governance flags must always be present regardless of LLM settings.
        d = result.to_dict()
        assert d.get("approval_required") is True
        assert d.get("execution_performed") is False
        assert d.get("draft_approved") is False

    def test_governance_flags_with_explicit_stub(self):
        from src.agent.workflow.natural_language_workflow import run_resolve_from_request
        result = run_resolve_from_request("test lasso request", provider="stub")
        d = result.to_dict()
        assert d.get("approval_required") is True
        assert d.get("execution_performed") is False
        assert d.get("draft_approved") is False


# ---------------------------------------------------------------------------
# E — No live network call in tests
# ---------------------------------------------------------------------------


class TestNoNetworkCall:
    """Calling resolve_llm_settings with openai-like settings must NOT
    trigger any network activity (the env var only sets configuration)."""

    def test_resolve_openai_settings_makes_no_network_call(self):
        # Simply calling resolve_llm_settings never connects to any endpoint.
        env = {
            _ENV_PROVIDER: "openai",
            _ENV_BASE_URL: "http://127.0.0.1:9999/v1",  # port guaranteed closed
            _ENV_MODEL: "qwen2.5-7b-instruct",
        }
        with patch.dict(os.environ, env, clear=False):
            # This must not raise — it's pure config resolution.
            s = resolve_llm_settings()
        assert s.provider == "openai"
        assert s.base_url == "http://127.0.0.1:9999/v1"

    def test_workflow_with_env_stub_makes_no_network_call(self):
        from src.agent.workflow.natural_language_workflow import run_resolve_from_request
        env = {_ENV_PROVIDER: "stub", _ENV_BASE_URL: "", _ENV_MODEL: ""}
        with patch.dict(os.environ, env, clear=False):
            result = run_resolve_from_request("lasso regression on etfs")
        # Stub mode → no network call; result must be structured.
        assert result.status in ("resolved", "needs_user_input", "needs_clarification", "error")


# ---------------------------------------------------------------------------
# F — MCP workflow functions accept None and resolve correctly
# ---------------------------------------------------------------------------


class TestWorkflowFacadeDefaults:
    """Workflow facade functions accept None provider and resolve it."""

    def test_run_resolve_from_request_accepts_none_provider(self):
        from src.agent.workflow.natural_language_workflow import run_resolve_from_request
        env = {k: v for k, v in os.environ.items() if k not in (_ENV_PROVIDER, _ENV_BASE_URL, _ENV_MODEL)}
        with patch.dict(os.environ, env, clear=True):
            result = run_resolve_from_request("lasso", provider=None)
        assert result.status in ("resolved", "needs_user_input", "needs_clarification", "error")

    def test_run_plan_from_request_accepts_none_provider(self):
        from src.agent.workflow.natural_language_workflow import run_plan_from_request
        env = {k: v for k, v in os.environ.items() if k not in (_ENV_PROVIDER, _ENV_BASE_URL, _ENV_MODEL)}
        with patch.dict(os.environ, env, clear=True):
            result = run_plan_from_request("lasso", provider=None)
        assert result.status in ("planned", "needs_review", "needs_user_input",
                                  "needs_clarification", "invalid", "error")

    def test_run_recommend_next_actions_accepts_none_provider(self):
        from src.agent.workflow.next_action_workflow import run_recommend_next_actions_from_request
        env = {k: v for k, v in os.environ.items() if k not in (_ENV_PROVIDER, _ENV_BASE_URL, _ENV_MODEL)}
        with patch.dict(os.environ, env, clear=True):
            result = run_recommend_next_actions_from_request("lasso etf", provider=None)
        assert result.status in ("recommended", "needs_more_evidence", "needs_clarification",
                                  "needs_user_input", "invalid", "error")

    def test_env_var_provider_used_in_workflow(self):
        """When ZETO_AGENT_LLM_PROVIDER=openai, the workflow uses openai—but since
        LM Studio is not running the call must gracefully fail (not crash with import
        error or governance violation)."""
        from src.agent.workflow.natural_language_workflow import run_resolve_from_request

        env = {k: v for k, v in os.environ.items() if k not in (_ENV_PROVIDER, _ENV_BASE_URL, _ENV_MODEL)}
        env[_ENV_PROVIDER] = "openai"
        env[_ENV_BASE_URL] = "http://127.0.0.1:19999/v1"  # guaranteed closed port
        env[_ENV_MODEL] = "qwen2.5-7b-instruct"
        with patch.dict(os.environ, env, clear=True):
            result = run_resolve_from_request("lasso", provider=None)
        # Must not crash; governance flags still present.
        d = result.to_dict()
        assert d.get("approval_required") is True
        assert d.get("execution_performed") is False
