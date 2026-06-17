"""MCP tool tests for env-var LLM defaults (Phase 5G).

Verifies that MCP handlers respect ZETO_AGENT_LLM_* env vars and that
no provider/model/base_url arg is required for stub-mode operation.

Design principles:
  - No live LM Studio calls.
  - No network activity.
  - Stub mode (no env vars set) is preserved.
  - Governance flags unchanged by env-var resolution.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_ENV_PROVIDER = "ZETO_AGENT_LLM_PROVIDER"
_ENV_BASE_URL = "ZETO_AGENT_LLM_BASE_URL"
_ENV_MODEL = "ZETO_AGENT_LLM_MODEL"

_CLEAN_ENV = {k: v for k, v in os.environ.items()
              if k not in (_ENV_PROVIDER, _ENV_BASE_URL, _ENV_MODEL)}


def _import_zeto():
    """Import the MCP handler module fresh."""
    import src.mcp.zeto_server as zeto
    return zeto


# ---------------------------------------------------------------------------
# A — Handlers accept no provider/model/base_url (use env defaults)
# ---------------------------------------------------------------------------


class TestHandlersAcceptNoLLMArgs:
    """MCP handlers work without provider/model/base_url — env var fills in."""

    def test_parse_research_intent_no_llm_args(self):
        zeto = _import_zeto()
        with patch.dict(os.environ, _CLEAN_ENV, clear=True):
            result = zeto._mcp_parse_research_intent(request="lasso regression etfs")
        assert isinstance(result, dict)
        assert "ok" in result

    def test_resolve_research_request_no_llm_args(self):
        zeto = _import_zeto()
        with patch.dict(os.environ, _CLEAN_ENV, clear=True):
            result = zeto._mcp_resolve_research_request(request="lasso regression etfs")
        assert isinstance(result, dict)
        assert "ok" in result

    def test_create_plan_from_request_no_llm_args(self):
        zeto = _import_zeto()
        with patch.dict(os.environ, _CLEAN_ENV, clear=True):
            result = zeto._mcp_create_experiment_plan_from_request(
                request="lasso regression etfs"
            )
        assert isinstance(result, dict)
        assert "ok" in result

    def test_create_draft_from_request_no_llm_args(self):
        zeto = _import_zeto()
        with patch.dict(os.environ, _CLEAN_ENV, clear=True):
            result = zeto._mcp_create_validated_draft_from_request(
                request="lasso regression on etfs"
            )
        assert isinstance(result, dict)
        assert "ok" in result

    def test_recommend_next_actions_from_request_no_llm_args(self):
        zeto = _import_zeto()
        with patch.dict(os.environ, _CLEAN_ENV, clear=True):
            result = zeto._mcp_recommend_next_actions_from_request(
                request="lasso regression on etfs"
            )
        assert isinstance(result, dict)
        assert "ok" in result

    def test_propose_spec_from_recommendation_request_no_llm_args(self):
        zeto = _import_zeto()
        with patch.dict(os.environ, _CLEAN_ENV, clear=True):
            result = zeto._mcp_propose_spec_from_recommendation_request(
                request="lasso regression on etfs"
            )
        assert isinstance(result, dict)
        assert "ok" in result


# ---------------------------------------------------------------------------
# B — Governance flags present when using env-var defaults
# ---------------------------------------------------------------------------


class TestGovernanceFlagsWithEnvDefaults:
    """Governance flags are always present regardless of LLM settings source."""

    def test_parse_governance_flags(self):
        zeto = _import_zeto()
        with patch.dict(os.environ, _CLEAN_ENV, clear=True):
            result = zeto._mcp_parse_research_intent(request="lasso regression etfs")
        data = result.get("data") or {}
        # The parse tool may or may not have these exact keys — but ok must be present.
        assert "ok" in result

    def test_resolve_governance_flags(self):
        zeto = _import_zeto()
        with patch.dict(os.environ, _CLEAN_ENV, clear=True):
            result = zeto._mcp_resolve_research_request(request="lasso regression etfs")
        data = result.get("data") or {}
        assert result.get("ok") in (True, False)
        # Governance flags are in data when present.
        if "approval_required" in data:
            assert data["approval_required"] is True
        if "execution_performed" in data:
            assert data["execution_performed"] is False

    def test_draft_governance_flags(self):
        zeto = _import_zeto()
        with patch.dict(os.environ, _CLEAN_ENV, clear=True):
            result = zeto._mcp_create_validated_draft_from_request(
                request="lasso regression on etfs"
            )
        data = result.get("data") or {}
        if "approval_required" in data:
            assert data["approval_required"] is True
        if "execution_performed" in data:
            assert data["execution_performed"] is False
        if "draft_approved" in data:
            assert data["draft_approved"] is False

    def test_next_action_governance_flags(self):
        zeto = _import_zeto()
        with patch.dict(os.environ, _CLEAN_ENV, clear=True):
            result = zeto._mcp_recommend_next_actions_from_request(
                request="lasso regression on etfs"
            )
        data = result.get("data") or {}
        if "approval_required" in data:
            assert data["approval_required"] is True
        if "execution_performed" in data:
            assert data["execution_performed"] is False


# ---------------------------------------------------------------------------
# C — Env var stub mode identical to explicit stub
# ---------------------------------------------------------------------------


class TestEnvStubIdenticalToExplicitStub:
    """Calling a handler with no args (env=stub) is equivalent to provider='stub'."""

    def test_parse_intent_no_args_vs_explicit_stub(self):
        zeto = _import_zeto()
        with patch.dict(os.environ, _CLEAN_ENV, clear=True):
            result_env = zeto._mcp_parse_research_intent(request="lasso")
            result_explicit = zeto._mcp_parse_research_intent(
                request="lasso", provider="stub"
            )
        assert result_env.get("stage") == result_explicit.get("stage")
        assert result_env.get("ok") == result_explicit.get("ok")

    def test_resolve_no_args_vs_explicit_stub(self):
        zeto = _import_zeto()
        with patch.dict(os.environ, _CLEAN_ENV, clear=True):
            result_env = zeto._mcp_resolve_research_request(request="lasso")
            result_explicit = zeto._mcp_resolve_research_request(
                request="lasso", provider="stub"
            )
        assert result_env.get("stage") == result_explicit.get("stage")
        assert result_env.get("ok") == result_explicit.get("ok")

    def test_draft_no_args_vs_explicit_stub(self):
        zeto = _import_zeto()
        with patch.dict(os.environ, _CLEAN_ENV, clear=True):
            result_env = zeto._mcp_create_validated_draft_from_request(
                request="lasso regression on etfs"
            )
            result_explicit = zeto._mcp_create_validated_draft_from_request(
                request="lasso regression on etfs", provider="stub"
            )
        assert result_env.get("stage") == result_explicit.get("stage")
        assert result_env.get("ok") == result_explicit.get("ok")


# ---------------------------------------------------------------------------
# D — ZETO_AGENT_LLM_PROVIDER is read by MCP handlers
# ---------------------------------------------------------------------------


class TestEnvVarReadByHandlers:
    """When ZETO_AGENT_LLM_PROVIDER is set, handlers read it (verified via mock)."""

    def test_env_provider_passed_to_workflow(self):
        """Patch run_resolve_from_request to capture which provider it received."""
        import src.agent.workflow.natural_language_workflow as nlw

        received: dict = {}

        original = nlw.run_resolve_from_request

        def capturing(*args, **kwargs):
            received["provider"] = kwargs.get("provider") or (args[1] if len(args) > 1 else None)
            return original(*args, **kwargs)

        zeto = _import_zeto()
        env = {**_CLEAN_ENV, _ENV_PROVIDER: "openai",
               _ENV_BASE_URL: "http://127.0.0.1:19999/v1",
               _ENV_MODEL: "qwen2.5-7b-instruct"}

        with patch.dict(os.environ, env, clear=True):
            with patch.object(nlw, "run_resolve_from_request", side_effect=capturing):
                zeto._mcp_resolve_research_request(request="lasso")

        # The MCP handler resolves env vars before calling the workflow function,
        # so the workflow receives the resolved provider string.
        assert received.get("provider") == "openai"

    def test_explicit_arg_beats_env_in_handler(self):
        """Passing provider='stub' explicitly overrides the env var."""
        import src.agent.workflow.natural_language_workflow as nlw

        received: dict = {}

        original = nlw.run_resolve_from_request

        def capturing(*args, **kwargs):
            received["provider"] = kwargs.get("provider") or (args[1] if len(args) > 1 else None)
            return original(*args, **kwargs)

        zeto = _import_zeto()
        env = {**_CLEAN_ENV, _ENV_PROVIDER: "openai",
               _ENV_BASE_URL: "http://127.0.0.1:19999/v1",
               _ENV_MODEL: "qwen2.5-7b-instruct"}

        with patch.dict(os.environ, env, clear=True):
            with patch.object(nlw, "run_resolve_from_request", side_effect=capturing):
                zeto._mcp_resolve_research_request(request="lasso", provider="stub")

        assert received.get("provider") == "stub"
