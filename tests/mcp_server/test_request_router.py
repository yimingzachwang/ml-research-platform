"""Tests for Phase 5G.2: route_research_request and wrong-tool guards.

Covers:
  A. route_research_request classification (new_research_design, memory_next_action,
     existing_experiment_review, execution_request)
  B. Wrong-tool guards for build_context_summary, run_experiment_review,
     generate_iteration_proposal (non-existent experiment names)
  C. Valid existing-experiment paths still work
  D. execute_approved_config placeholder path guard
  E. Governance flags always false in guards/router
  F. Contract keys present in all guard responses
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import src.mcp.zeto_server as zeto

_MOD = "src.mcp.zeto_server._api"
_CONTRACT_KEYS = {"ok", "stage", "display", "data", "next_suggested_action"}


# ---------------------------------------------------------------------------
# A — route_research_request classification
# ---------------------------------------------------------------------------


class TestRouteNewResearchDesign:
    """Canonical ETF + Lasso request should classify as new_research_design."""

    _REQUEST = (
        "Use the canonical multi-asset ETF preset from 2015 to 2024 "
        "and test Lasso instead of Ridge."
    )

    def _out(self):
        return zeto.route_research_request(self._REQUEST)

    def test_ok_true(self):
        assert self._out()["ok"] is True

    def test_stage_request_routed(self):
        assert self._out()["stage"] == "request_routed"

    def test_request_type_new_research_design(self):
        assert self._out()["data"]["request_type"] == "new_research_design"

    def test_recommended_tool_create_validated_draft(self):
        assert self._out()["data"]["recommended_tool"] == "create_validated_draft_from_request"

    def test_next_suggested_action_is_recommended_tool(self):
        out = self._out()
        assert out["next_suggested_action"] == out["data"]["recommended_tool"]

    def test_discouraged_tools_include_session_tools(self):
        disc = self._out()["data"]["discouraged_tools"]
        assert "create_research_session" in disc
        assert "run_experiment_review" in disc
        assert "generate_experiment_draft" in disc

    def test_requires_confirmation_false(self):
        assert self._out()["data"]["requires_confirmation"] is False

    def test_governance_flags_all_false(self):
        gov = self._out()["data"]["governance"]
        assert gov["draft_created"] is False
        assert gov["draft_approved"] is False
        assert gov["execution_performed"] is False

    def test_contract_keys_present(self):
        assert _CONTRACT_KEYS.issubset(self._out())

    def test_lasso_text_triggers_new_design(self):
        out = zeto.route_research_request("Test Lasso on the ETF universe")
        assert out["data"]["request_type"] == "new_research_design"

    def test_preset_text_triggers_new_design(self):
        out = zeto.route_research_request("Use the canonical multi-asset preset")
        assert out["data"]["request_type"] == "new_research_design"

    def test_timeframe_triggers_new_design(self):
        out = zeto.route_research_request("from 2015 to 2024 with Ridge")
        assert out["data"]["request_type"] == "new_research_design"


class TestRouteMemoryNextAction:
    def test_what_should_we_try_next(self):
        out = zeto.route_research_request("What should we try next?")
        assert out["data"]["request_type"] == "memory_next_action"

    def test_recommended_tool_is_recommend_next_actions(self):
        out = zeto.route_research_request("What should we try next?")
        assert out["data"]["recommended_tool"] == "recommend_next_actions_from_request"

    def test_recommend_keyword(self):
        out = zeto.route_research_request("recommend the next experiment")
        assert out["data"]["request_type"] == "memory_next_action"

    def test_governance_flags_all_false(self):
        out = zeto.route_research_request("What should we try next?")
        gov = out["data"]["governance"]
        assert gov["draft_created"] is False
        assert gov["draft_approved"] is False
        assert gov["execution_performed"] is False

    def test_contract_keys_present(self):
        assert _CONTRACT_KEYS.issubset(zeto.route_research_request("What should we try?"))


class TestRouteExistingExperimentReview:
    def test_review_keyword(self):
        out = zeto.route_research_request("Review canonical_ml_showcase")
        assert out["data"]["request_type"] == "existing_experiment_review"

    def test_recommended_tool_is_run_experiment_review(self):
        out = zeto.route_research_request("Review canonical_ml_showcase")
        assert out["data"]["recommended_tool"] == "run_experiment_review"

    def test_compare_keyword(self):
        out = zeto.route_research_request("compare canonical_ml_showcase and v2")
        assert out["data"]["request_type"] == "existing_experiment_review"

    def test_contract_keys_present(self):
        assert _CONTRACT_KEYS.issubset(zeto.route_research_request("review my experiment"))


class TestRouteExecutionRequest:
    def test_run_it_keyword(self):
        out = zeto.route_research_request("Run it now")
        assert out["data"]["request_type"] == "execution_request"

    def test_recommended_tool_execute(self):
        out = zeto.route_research_request("run the experiment now")
        assert out["data"]["recommended_tool"] == "execute_approved_config"

    def test_requires_confirmation_true(self):
        out = zeto.route_research_request("execute the experiment now")
        assert out["data"]["requires_confirmation"] is True

    def test_contract_keys_present(self):
        assert _CONTRACT_KEYS.issubset(zeto.route_research_request("run it"))

    def test_governance_execution_not_performed(self):
        out = zeto.route_research_request("execute now")
        assert out["data"]["governance"]["execution_performed"] is False


# ---------------------------------------------------------------------------
# B — Wrong-tool guards: non-existent experiment names
# ---------------------------------------------------------------------------


_INVENTED = "CanonicalMultiAssetETF_2015_to_2024"
_INVENTED2 = "TestLasso2015_2024"


class TestBuildContextSummaryGuard:
    def test_invented_name_returns_wrong_tool_stage(self):
        out = zeto.build_context_summary(_INVENTED)
        assert out["stage"] == "wrong_tool_for_new_research_design_request"

    def test_ok_false(self):
        assert zeto.build_context_summary(_INVENTED)["ok"] is False

    def test_recommended_tool_in_data(self):
        out = zeto.build_context_summary(_INVENTED)
        assert out["data"]["recommended_tool"] == "create_validated_draft_from_request"

    def test_next_suggested_action_is_create_draft(self):
        out = zeto.build_context_summary(_INVENTED)
        assert out["next_suggested_action"] == "create_validated_draft_from_request"

    def test_governance_flags_all_false(self):
        gov = zeto.build_context_summary(_INVENTED)["data"]["governance"]
        assert gov["draft_created"] is False
        assert gov["draft_approved"] is False
        assert gov["execution_performed"] is False

    def test_contract_keys_present(self):
        assert _CONTRACT_KEYS.issubset(zeto.build_context_summary(_INVENTED))

    def test_does_not_call_api_when_guard_fires(self):
        with patch(f"{_MOD}.build_llm_context") as m:
            zeto.build_context_summary(_INVENTED)
        m.assert_not_called()


class TestRunExperimentReviewGuard:
    def test_invented_name_returns_wrong_tool_stage(self):
        out = zeto.run_experiment_review(_INVENTED2)
        assert out["stage"] == "wrong_tool_for_new_research_design_request"

    def test_ok_false(self):
        assert zeto.run_experiment_review(_INVENTED2)["ok"] is False

    def test_recommended_tool_in_data(self):
        out = zeto.run_experiment_review(_INVENTED2)
        assert out["data"]["recommended_tool"] == "create_validated_draft_from_request"

    def test_next_suggested_action_is_create_draft(self):
        out = zeto.run_experiment_review(_INVENTED2)
        assert out["next_suggested_action"] == "create_validated_draft_from_request"

    def test_governance_flags_all_false(self):
        gov = zeto.run_experiment_review(_INVENTED2)["data"]["governance"]
        assert gov["draft_created"] is False
        assert gov["draft_approved"] is False
        assert gov["execution_performed"] is False

    def test_contract_keys_present(self):
        assert _CONTRACT_KEYS.issubset(zeto.run_experiment_review(_INVENTED2))

    def test_does_not_call_api_when_guard_fires(self):
        with patch(f"{_MOD}.get_experiment_metrics") as m:
            zeto.run_experiment_review(_INVENTED2)
        m.assert_not_called()


class TestGenerateIterationProposalGuard:
    def test_invented_name_returns_wrong_tool_stage(self):
        out = zeto.generate_iteration_proposal(_INVENTED)
        assert out["stage"] == "wrong_tool_for_new_research_design_request"

    def test_ok_false(self):
        assert zeto.generate_iteration_proposal(_INVENTED)["ok"] is False

    def test_recommended_tool_in_data(self):
        out = zeto.generate_iteration_proposal(_INVENTED)
        assert out["data"]["recommended_tool"] == "create_validated_draft_from_request"

    def test_next_suggested_action_is_create_draft(self):
        out = zeto.generate_iteration_proposal(_INVENTED)
        assert out["next_suggested_action"] == "create_validated_draft_from_request"

    def test_governance_flags_all_false(self):
        gov = zeto.generate_iteration_proposal(_INVENTED)["data"]["governance"]
        assert gov["draft_created"] is False
        assert gov["draft_approved"] is False
        assert gov["execution_performed"] is False

    def test_contract_keys_present(self):
        assert _CONTRACT_KEYS.issubset(zeto.generate_iteration_proposal(_INVENTED))

    def test_does_not_call_api_when_guard_fires(self):
        with patch(f"{_MOD}.generate_iteration_proposal") as m:
            zeto.generate_iteration_proposal(_INVENTED)
        m.assert_not_called()


# ---------------------------------------------------------------------------
# C — Valid existing-experiment paths still work (guard bypassed)
# ---------------------------------------------------------------------------


class TestExistingExperimentPathsUnblocked:
    """When _experiment_config_exists returns True, guards do not fire."""

    def _stub_ctx(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            experiment_name="canonical_ml_showcase",
            failure_modes=[],
            performance={"sharpe_ratio": 0.8},
            validation={"mean_oos_sharpe": 0.5, "n_splits": 5},
        )

    def test_build_context_summary_passes_through(self):
        ctx = self._stub_ctx()
        with (
            patch(f"{_MOD}.build_llm_context", return_value=ctx),
            patch(f"{_MOD}.compute_context_hash", return_value="a" * 64),
            patch("src.mcp.zeto_server._experiment_config_exists", return_value=True),
        ):
            out = zeto.build_context_summary("canonical_ml_showcase")
        assert out["ok"] is True
        assert out["stage"] == "context_built"

    def test_run_experiment_review_passes_through(self):
        from types import SimpleNamespace
        review = SimpleNamespace(sections={"summary": "ok"}, flags=[])
        ctx = self._stub_ctx()
        with (
            patch(f"{_MOD}.get_experiment_metrics", return_value={
                "status": "found", "metrics": {"sharpe_ratio": 0.8},
                "missing_metrics": [], "experiment_name": "canonical_ml_showcase",
            }),
            patch(f"{_MOD}.run_llm_review", return_value=review),
            patch(f"{_MOD}.build_llm_context", return_value=ctx),
            patch(f"{_MOD}.compute_context_hash", return_value="a" * 64),
            patch("src.mcp.zeto_server._experiment_config_exists", return_value=True),
        ):
            out = zeto.run_experiment_review("canonical_ml_showcase")
        assert out["ok"] is True
        assert out["stage"] == "review_generated"

    def test_generate_iteration_proposal_passes_through(self):
        from types import SimpleNamespace
        proposal = SimpleNamespace(
            research_focus="focus", context_hash="a" * 64,
            confidence="high", validation_concerns=[],
        )
        with (
            patch(f"{_MOD}.generate_iteration_proposal", return_value=proposal),
            patch(f"{_MOD}.proposal_artifact_path", return_value="proposals/p.json"),
            patch("src.mcp.zeto_server._experiment_config_exists", return_value=True),
        ):
            out = zeto.generate_iteration_proposal("canonical_ml_showcase")
        assert out["ok"] is True
        assert out["stage"] == "proposal_generated"


# ---------------------------------------------------------------------------
# D — execute_approved_config placeholder guard
# ---------------------------------------------------------------------------


class TestExecutePlaceholderGuard:
    def _call(self, path: str, confirmation: str = "RUN"):
        return zeto.execute_approved_config(path, confirmation=confirmation)

    def test_placeholder_path_prefix_rejected(self):
        out = self._call("/path/to/experiment/config.yaml")
        assert out["ok"] is False
        assert out["stage"] == "invalid_execution_request"

    def test_relative_placeholder_path_rejected(self):
        out = self._call("path/to/config.yaml")
        assert out["ok"] is False
        assert out["stage"] == "invalid_execution_request"

    def test_angle_bracket_template_rejected(self):
        out = self._call("<config_path>")
        assert out["ok"] is False
        assert out["stage"] == "invalid_execution_request"

    def test_empty_path_rejected(self):
        out = self._call("")
        assert out["ok"] is False
        assert out["stage"] == "invalid_execution_request"

    def test_whitespace_only_path_rejected(self):
        out = self._call("   ")
        assert out["ok"] is False
        assert out["stage"] == "invalid_execution_request"

    def test_execution_not_performed_in_data(self):
        out = self._call("/path/to/config.yaml")
        assert out["data"]["execution_performed"] is False

    def test_draft_not_approved_in_data(self):
        out = self._call("/path/to/config.yaml")
        assert out["data"]["draft_approved"] is False

    def test_contract_keys_present(self):
        out = self._call("/path/to/config.yaml")
        assert _CONTRACT_KEYS.issubset(out)

    def test_real_path_not_rejected_by_placeholder_guard(self):
        # Real configs/ path must not be rejected by the placeholder guard.
        # Execution is refused by the RUN gate instead (no mock needed).
        out = self._call("configs/experiments/canonical_ml_showcase_v2.yaml",
                         confirmation="")
        # Placeholder guard should NOT fire; RUN gate fires.
        assert out["stage"] == "execution_refused"

    def test_placeholder_rejected_even_with_correct_confirmation(self):
        # Guard fires before RUN check.
        out = self._call("/path/to/config.yaml", confirmation="RUN")
        assert out["stage"] == "invalid_execution_request"

    def test_api_not_called_when_placeholder(self):
        with patch(f"{_MOD}.execute_approved_config") as m:
            zeto.execute_approved_config("/path/to/config.yaml", confirmation="RUN")
        m.assert_not_called()


# ---------------------------------------------------------------------------
# E — is_placeholder_path helper
# ---------------------------------------------------------------------------


class TestIsPlaceholderPath:
    def test_slash_path_to_is_placeholder(self):
        assert zeto._is_placeholder_path("/path/to/config.yaml") is True

    def test_relative_path_to_is_placeholder(self):
        assert zeto._is_placeholder_path("path/to/config.yaml") is True

    def test_angle_bracket_is_placeholder(self):
        assert zeto._is_placeholder_path("<config_path>") is True
        assert zeto._is_placeholder_path("<experiment/config.yaml>") is True

    def test_empty_is_placeholder(self):
        assert zeto._is_placeholder_path("") is True
        assert zeto._is_placeholder_path("   ") is True

    def test_real_configs_path_not_placeholder(self):
        assert zeto._is_placeholder_path("configs/experiments/exp_v2.yaml") is False

    def test_absolute_real_path_not_placeholder(self):
        assert zeto._is_placeholder_path(
            "/Users/zach/quant-research-platform/configs/experiments/exp.yaml"
        ) is False
