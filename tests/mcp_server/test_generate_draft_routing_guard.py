"""Tests for the generate_experiment_draft routing guard (Phase 5G.1).

The guard fires when generate_experiment_draft raises an exception AND the
experiment config YAML does not exist on disk (i.e. brand-new design request).

When the experiment config DOES exist (existing-experiment, missing-proposal
case), the guard must NOT fire — the original draft_generation_failed envelope
is returned instead.
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
# A — Guard fires for non-existent experiment (brand-new design request)
# ---------------------------------------------------------------------------


class TestRoutingGuardNonexistentExperiment:
    """Guard redirects to create_validated_draft_from_request when config is missing."""

    def _call_with_nonexistent(self, exc=None):
        if exc is None:
            exc = FileNotFoundError("No IterationProposal found for 'TestLasso2015_2024'")
        with (
            patch(f"{_MOD}.generate_experiment_draft", side_effect=exc),
            patch("src.mcp.zeto_server._experiment_config_exists", return_value=False),
        ):
            return zeto.generate_experiment_draft("TestLasso2015_2024", provider="stub")

    def test_guard_returns_ok_false(self):
        out = self._call_with_nonexistent()
        assert out["ok"] is False

    def test_guard_stage_is_wrong_tool(self):
        out = self._call_with_nonexistent()
        assert out["stage"] == "wrong_tool_for_research_design_request"

    def test_guard_next_action_is_create_draft(self):
        out = self._call_with_nonexistent()
        assert out["next_suggested_action"] == "create_validated_draft_from_request"

    def test_guard_data_has_recommended_tool(self):
        out = self._call_with_nonexistent()
        assert out["data"]["recommended_tool"] == "create_validated_draft_from_request"

    def test_guard_data_has_experiment_name(self):
        out = self._call_with_nonexistent()
        assert out["data"]["experiment_name"] == "TestLasso2015_2024"

    def test_guard_governance_flags_all_false(self):
        out = self._call_with_nonexistent()
        gov = out["data"]["governance"]
        assert gov["draft_created"] is False
        assert gov["draft_approved"] is False
        assert gov["execution_performed"] is False

    def test_guard_contract_keys_present(self):
        out = self._call_with_nonexistent()
        assert _CONTRACT_KEYS.issubset(out)

    def test_guard_display_mentions_create_validated_draft(self):
        out = self._call_with_nonexistent()
        assert "create_validated_draft_from_request" in out["display"]

    def test_guard_fires_for_value_error_too(self):
        out = self._call_with_nonexistent(exc=ValueError("no config"))
        assert out["stage"] == "wrong_tool_for_research_design_request"

    def test_guard_fires_for_runtime_error_too(self):
        out = self._call_with_nonexistent(exc=RuntimeError("unexpected"))
        assert out["stage"] == "wrong_tool_for_research_design_request"


# ---------------------------------------------------------------------------
# B — Guard does NOT fire for existing-experiment missing-proposal case
# ---------------------------------------------------------------------------


class TestRoutingGuardExistingExperiment:
    """When config exists but proposal is missing, return draft_generation_failed."""

    def _call_with_existing(self):
        with (
            patch(f"{_MOD}.generate_experiment_draft",
                  side_effect=FileNotFoundError("No IterationProposal found for 'canonical_ml_showcase'")),
            patch("src.mcp.zeto_server._experiment_config_exists", return_value=True),
        ):
            return zeto.generate_experiment_draft("canonical_ml_showcase", provider="stub")

    def test_guard_does_not_fire(self):
        out = self._call_with_existing()
        assert out["stage"] == "draft_generation_failed"

    def test_ok_is_false(self):
        out = self._call_with_existing()
        assert out["ok"] is False

    def test_next_action_is_generate_iteration_proposal(self):
        out = self._call_with_existing()
        assert out["next_suggested_action"] == "generate_iteration_proposal"

    def test_error_field_present(self):
        out = self._call_with_existing()
        assert "error" in out["data"]

    def test_contract_keys_present(self):
        out = self._call_with_existing()
        assert _CONTRACT_KEYS.issubset(out)


# ---------------------------------------------------------------------------
# C — _experiment_config_exists helper
# ---------------------------------------------------------------------------


class TestExperimentConfigExists:
    def test_returns_false_for_nonexistent_name(self):
        result = zeto._experiment_config_exists("__totally_nonexistent_experiment_xyz__")
        assert result is False

    def test_returns_bool(self):
        result = zeto._experiment_config_exists("canonical_ml_showcase")
        assert isinstance(result, bool)

    def test_patchable(self):
        with patch("src.mcp.zeto_server._experiment_config_exists", return_value=True):
            assert zeto._experiment_config_exists("anything") is True
        with patch("src.mcp.zeto_server._experiment_config_exists", return_value=False):
            assert zeto._experiment_config_exists("anything") is False
