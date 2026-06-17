"""Tests for Phase 5G.3: create_research_session guard and operator manual display.

Covers:
  A. create_research_session with invented new-design names → redirect
  B. Valid existing-experiment session creation still works
  C. Neutral names (no design signals, no config) still pass through
  D. Operator manual display includes route-first rule prominently
  E. Governance flags always false in guard response
  F. No API call in redirect case
  G. _looks_like_new_design_session helper
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import src.mcp.zeto_server as zeto

_MOD = "src.mcp.zeto_server._api"
_CONTRACT_KEYS = {"ok", "stage", "display", "data", "next_suggested_action"}


def _stub_session():
    """Minimal stub for a created session."""
    return SimpleNamespace(
        session_id="s-test-001",
        research_goal="test goal",
        root_experiment="canonical_ml_showcase",
        active_experiment="canonical_ml_showcase",
        status="active",
    )


# ---------------------------------------------------------------------------
# A — Invented new-design names trigger the guard
# ---------------------------------------------------------------------------


class TestSessionCreationGuardFires:
    _INVENTED_CASES = [
        ("CanonicalMultiAssetETF_2015_to_2024", "test Lasso instead of Ridge"),
        ("canonical_multi_asset_ETF_preset_2015_to_2024", "baseline"),
        ("TestLasso2015_2024", "canonical preset"),
        ("lasso_experiment_2024", "use preset"),
        ("etf_model_ridge_2022", "run experiment"),
        ("test_canonical_preset", "2015 to 2024"),
    ]

    @pytest.mark.parametrize("name,goal", _INVENTED_CASES)
    def test_invented_name_returns_wrong_tool_stage(self, name, goal):
        out = zeto.create_research_session(name, goal)
        assert out["stage"] == "wrong_tool_for_new_research_design_request", (
            f"Expected guard to fire for name={name!r} goal={goal!r}, "
            f"got stage={out['stage']!r}"
        )

    @pytest.mark.parametrize("name,goal", _INVENTED_CASES)
    def test_ok_false(self, name, goal):
        assert zeto.create_research_session(name, goal)["ok"] is False

    def test_recommends_route_research_request(self):
        out = zeto.create_research_session(
            "CanonicalMultiAssetETF_2015_to_2024", "test Lasso"
        )
        assert out["data"]["recommended_tool"] == "route_research_request"
        assert out["next_suggested_action"] == "route_research_request"

    def test_recommends_create_validated_draft_as_next(self):
        out = zeto.create_research_session(
            "CanonicalMultiAssetETF_2015_to_2024", "test Lasso"
        )
        assert out["data"]["next_recommended_tool"] == "create_validated_draft_from_request"

    def test_discouraged_tools_list(self):
        out = zeto.create_research_session(
            "canonical_multi_asset_ETF_preset_2015_to_2024", "baseline"
        )
        disc = out["data"]["discouraged_tools"]
        assert "create_research_session" in disc
        assert "build_context_summary" in disc
        assert "run_experiment_review" in disc
        assert "generate_iteration_proposal" in disc
        assert "generate_experiment_draft" in disc
        assert "execute_approved_config" in disc

    def test_governance_flags_all_false(self):
        out = zeto.create_research_session(
            "CanonicalMultiAssetETF_2015_to_2024", "test Lasso"
        )
        gov = out["data"]["governance"]
        assert gov["draft_created"] is False
        assert gov["draft_approved"] is False
        assert gov["execution_performed"] is False

    def test_contract_keys_present(self):
        out = zeto.create_research_session(
            "CanonicalMultiAssetETF_2015_to_2024", "test Lasso"
        )
        assert _CONTRACT_KEYS.issubset(out)

    def test_display_mentions_route_research_request(self):
        out = zeto.create_research_session(
            "CanonicalMultiAssetETF_2015_to_2024", "test Lasso"
        )
        assert "route_research_request" in out["display"]

    def test_display_mentions_create_validated_draft(self):
        out = zeto.create_research_session(
            "CanonicalMultiAssetETF_2015_to_2024", "test Lasso"
        )
        assert "create_validated_draft_from_request" in out["display"]

    def test_no_session_created_when_guard_fires(self):
        with patch(f"{_MOD}.create_research_session") as m:
            zeto.create_research_session(
                "CanonicalMultiAssetETF_2015_to_2024", "test Lasso"
            )
        m.assert_not_called()


# ---------------------------------------------------------------------------
# B — Valid existing experiments pass through
# ---------------------------------------------------------------------------


class TestSessionCreationValidExperiments:
    """Known config-backed experiments bypass the guard."""

    def _call_with_existing(self, name="canonical_ml_showcase", goal="test goal"):
        with (
            patch(f"{_MOD}.create_research_session", return_value=_stub_session()),
            patch("src.mcp.zeto_server._experiment_config_exists", return_value=True),
        ):
            return zeto.create_research_session(name, goal)

    def test_stage_session_created(self):
        assert self._call_with_existing()["stage"] == "session_created"

    def test_ok_true(self):
        assert self._call_with_existing()["ok"] is True

    def test_session_id_in_data(self):
        out = self._call_with_existing()
        assert "session_id" in out["data"]

    def test_canonical_ml_showcase_passes(self):
        out = self._call_with_existing("canonical_ml_showcase", "improve Sharpe")
        assert out["ok"] is True

    def test_canonical_ml_multi_asset_passes(self):
        out = self._call_with_existing("canonical_ml_multi_asset", "explore features")
        assert out["ok"] is True

    def test_api_called_when_guard_does_not_fire(self):
        with (
            patch(f"{_MOD}.create_research_session", return_value=_stub_session()) as m,
            patch("src.mcp.zeto_server._experiment_config_exists", return_value=True),
        ):
            zeto.create_research_session("canonical_ml_showcase", "test goal")
        m.assert_called_once()


# ---------------------------------------------------------------------------
# C — Neutral names without design signals pass through
# ---------------------------------------------------------------------------


class TestSessionCreationNeutralNames:
    """Names without design signals and no config YAML still create a session."""

    def _call_neutral(self, name, goal):
        with patch(f"{_MOD}.create_research_session", return_value=_stub_session()):
            return zeto.create_research_session(name, goal)

    def test_exp_a_goal_passes(self):
        out = self._call_neutral("exp_a", "goal")
        # Guard won't fire: no design signals in "exp_a" + "goal"
        assert out["stage"] == "session_created"

    def test_plain_name_passes(self):
        out = self._call_neutral("my_baseline", "improve performance")
        assert out["stage"] == "session_created"

    def test_generic_name_with_version_passes(self):
        out = self._call_neutral("experiment_v3", "testing methodology")
        assert out["stage"] == "session_created"


# ---------------------------------------------------------------------------
# D — Operator manual display includes route-first rule
# ---------------------------------------------------------------------------


class TestOperatorManualDisplayRouteFirst:
    def _display(self):
        return zeto.get_zeto_operator_manual()["display"]

    def _blob(self):
        return json.dumps(zeto.get_zeto_operator_manual())

    def test_display_includes_route_research_request(self):
        assert "route_research_request" in self._display()

    def test_display_includes_create_validated_draft(self):
        assert "create_validated_draft_from_request" in self._display()

    def test_display_says_do_not_create_session_first(self):
        display = self._display().lower()
        assert "do not create a research session" in display

    def test_display_says_do_not_invent_experiment_names(self):
        display = self._display().lower()
        assert "do not invent experiment names" in display

    def test_manual_blob_contains_routing_first(self):
        blob = self._blob().lower()
        assert "routing first" in blob

    def test_manual_blob_contains_session_guard(self):
        blob = self._blob().lower()
        assert "session guard" in blob

    def test_payload_still_under_limit(self):
        assert len(json.dumps(zeto.get_zeto_operator_manual())) < 9000


# ---------------------------------------------------------------------------
# E — _looks_like_new_design_session helper
# ---------------------------------------------------------------------------


class TestLooksLikeNewDesignSession:
    def test_canonical_etf_name(self):
        assert zeto._looks_like_new_design_session(
            "CanonicalMultiAssetETF_2015_to_2024", "baseline"
        ) is True

    def test_snake_case_canonical_preset(self):
        assert zeto._looks_like_new_design_session(
            "canonical_multi_asset_ETF_preset_2015_to_2024", ""
        ) is True

    def test_lasso_year_name(self):
        assert zeto._looks_like_new_design_session("TestLasso2015_2024", "") is True

    def test_year_in_goal(self):
        assert zeto._looks_like_new_design_session(
            "new_experiment", "use data from 2015 to 2024"
        ) is True

    def test_lasso_in_goal(self):
        assert zeto._looks_like_new_design_session(
            "exp_v3", "test lasso instead of ridge"
        ) is True

    def test_neutral_name_and_goal(self):
        assert zeto._looks_like_new_design_session("exp_a", "goal") is False

    def test_plain_improvement_goal(self):
        assert zeto._looks_like_new_design_session(
            "my_experiment", "improve performance"
        ) is False

    def test_generic_versioned_name(self):
        assert zeto._looks_like_new_design_session(
            "experiment_v3", "testing methodology"
        ) is False

    def test_returns_bool(self):
        assert isinstance(
            zeto._looks_like_new_design_session("exp_a", "goal"), bool
        )
