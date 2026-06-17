"""Tests for Phase 3D MCP persistence behaviour in create_validated_draft_from_spec.

Covers save_draft=True paths in the MCP tool:
  - Happy path: save_draft=True + draft_created → saved_draft key in data
  - Governance: saved_draft carries approval_required=True/execution_performed=False/draft_approved=False
  - Not-saved path: persistence returns not_saved (status mismatch, hard-unsupported)
  - Error path: persistence returns error
  - save_draft=False: no saved_draft key, no save_draft_note key
  - Display updates when draft is saved or not saved
  - MCP envelope contract always satisfied

All calls to run_draft_from_spec and _save_agent_draft are mocked.
No disk I/O. No real resolver. No quant engine. No LLM.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.mcp import zeto_server as zeto

_CONTRACT_KEYS = {"ok", "stage", "display", "data", "next_suggested_action"}
_MOD_WORKFLOW = "src.mcp.zeto_server._agent_workflow"
_MOD_PERSIST = "src.mcp.zeto_server._save_agent_draft"

_CANONICAL_SPEC = json.dumps({"preset": "canonical_multi_asset_etf", "end_date": "2024-12-31"})
_HARD_UNSUPPORTED_SPEC = json.dumps({
    "preset": "canonical_multi_asset_etf",
    "end_date": "2024-12-31",
    "universe": "canonical_multi_asset_etfs",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wf_result(status: str = "draft_created", **overrides) -> MagicMock:
    d = {
        "status": status,
        "summary": f"stub {status}",
        "resolution": {"status": status, "resolved_spec": {}},
        "plan": {"plan_id": "p1", "name": "stub"},
        "draft": {"draft_id": "d1", "proposed_name": "exp_v2", "changes": []},
        "warnings": [],
        "errors": [],
        "missing_fields": [],
        "defaulted_fields": {},
        "assumptions": [],
        "requires_confirmation": False,
        "unsupported_fields": [],
        "hard_unsupported_fields": [],
        "approval_required": True,
        "execution_performed": False,
        "draft_approved": False,
        "mapped_changes": [],
        "validation_errors": [],
        "proposed_name": "exp_v2",
        "base_experiment": "canonical_ml_multi_asset",
    }
    d.update(overrides)
    m = MagicMock()
    m.status = status
    for k, v in d.items():
        setattr(m, k, v)
    m.to_dict.return_value = d
    return m


def _make_persist_result(
    status: str = "saved",
    draft_id: str = "d001",
    draft_path: str = "/tmp/results/llm_reviews/canonical/draft_d001.json",
    summary: str = "Draft 'exp_v2' (id=d001) saved.",
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> MagicMock:
    m = MagicMock()
    m.status = status
    m.draft_id = draft_id
    m.draft_path = draft_path
    m.summary = summary
    d = {
        "status": status,
        "draft_id": draft_id,
        "draft_path": draft_path,
        "base_experiment": "canonical_ml_multi_asset",
        "proposed_name": "exp_v2",
        "approval_required": True,
        "execution_performed": False,
        "draft_approved": False,
        "warnings": warnings or [],
        "errors": errors or [],
        "summary": summary,
    }
    m.to_dict.return_value = d
    return m


def _assert_contract(out: dict) -> None:
    assert _CONTRACT_KEYS.issubset(out.keys()), (
        f"Missing contract keys: {_CONTRACT_KEYS - set(out.keys())}"
    )
    assert isinstance(out["ok"], bool)
    assert isinstance(out["stage"], str) and out["stage"]
    assert isinstance(out["display"], str)
    assert isinstance(out["data"], dict)
    assert isinstance(out["next_suggested_action"], str)


# ---------------------------------------------------------------------------
# save_draft=False: no persistence key
# ---------------------------------------------------------------------------


class TestSaveDraftFalse:
    def test_no_saved_draft_key(self):
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=False)
        assert "saved_draft" not in out["data"]

    def test_no_save_draft_note_key(self):
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=False)
        assert "save_draft_note" not in out["data"]

    def test_persistence_not_called_when_false(self):
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")), \
             patch(_MOD_PERSIST) as mock_persist:
            zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=False)
        mock_persist.assert_not_called()

    def test_contract_satisfied_when_false(self):
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=False)
        _assert_contract(out)


# ---------------------------------------------------------------------------
# save_draft=True + draft_created → saved
# ---------------------------------------------------------------------------


class TestSaveDraftTrueHappyPath:
    def test_saved_draft_key_present_in_data(self):
        persist = _make_persist_result("saved")
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")), \
             patch(_MOD_PERSIST, return_value=persist):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        assert "saved_draft" in out["data"]

    def test_saved_draft_status_is_saved(self):
        persist = _make_persist_result("saved")
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")), \
             patch(_MOD_PERSIST, return_value=persist):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        assert out["data"]["saved_draft"]["status"] == "saved"

    def test_saved_draft_has_draft_id(self):
        persist = _make_persist_result("saved", draft_id="d001")
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")), \
             patch(_MOD_PERSIST, return_value=persist):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        assert out["data"]["saved_draft"]["draft_id"] == "d001"

    def test_saved_draft_has_draft_path(self):
        persist = _make_persist_result("saved", draft_path="/tmp/draft.json")
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")), \
             patch(_MOD_PERSIST, return_value=persist):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        assert out["data"]["saved_draft"]["draft_path"] == "/tmp/draft.json"

    def test_saved_draft_governance_flags(self):
        persist = _make_persist_result("saved")
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")), \
             patch(_MOD_PERSIST, return_value=persist):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        sd = out["data"]["saved_draft"]
        assert sd["approval_required"] is True
        assert sd["execution_performed"] is False
        assert sd["draft_approved"] is False

    def test_ok_still_true_when_saved(self):
        persist = _make_persist_result("saved")
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")), \
             patch(_MOD_PERSIST, return_value=persist):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        assert out["ok"] is True

    def test_display_mentions_saved(self):
        persist = _make_persist_result("saved", draft_id="d001", draft_path="/tmp/d.json")
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")), \
             patch(_MOD_PERSIST, return_value=persist):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        assert "saved" in out["display"].lower() or "d001" in out["display"]

    def test_contract_satisfied_when_saved(self):
        persist = _make_persist_result("saved")
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")), \
             patch(_MOD_PERSIST, return_value=persist):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        _assert_contract(out)

    def test_persistence_called_with_workflow_result(self):
        persist = _make_persist_result("saved")
        wf = _make_wf_result("draft_created")
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=wf), \
             patch(_MOD_PERSIST, return_value=persist) as mock_p:
            zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        mock_p.assert_called_once_with(wf)


# ---------------------------------------------------------------------------
# save_draft=True + draft_created → not_saved
# ---------------------------------------------------------------------------


class TestSaveDraftTrueNotSaved:
    def test_not_saved_adds_saved_draft_with_not_saved_status(self):
        persist = _make_persist_result(
            "not_saved",
            summary="Draft not saved: hard-unsupported fields.",
        )
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")), \
             patch(_MOD_PERSIST, return_value=persist):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        assert out["data"]["saved_draft"]["status"] == "not_saved"

    def test_ok_still_true_when_not_saved_but_draft_created(self):
        persist = _make_persist_result("not_saved", summary="not saved")
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")), \
             patch(_MOD_PERSIST, return_value=persist):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        # draft_created → ok=True regardless of save outcome
        assert out["ok"] is True

    def test_display_mentions_not_saved(self):
        persist = _make_persist_result("not_saved", summary="wrong status")
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")), \
             patch(_MOD_PERSIST, return_value=persist):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        assert "not saved" in out["display"].lower() or "wrong status" in out["display"].lower()

    def test_contract_satisfied_when_not_saved(self):
        persist = _make_persist_result("not_saved", summary="not saved")
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")), \
             patch(_MOD_PERSIST, return_value=persist):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        _assert_contract(out)


# ---------------------------------------------------------------------------
# save_draft=True + persistence error
# ---------------------------------------------------------------------------


class TestSaveDraftTrueError:
    def test_error_status_in_saved_draft(self):
        persist = _make_persist_result("error", summary="disk full", errors=["OSError: disk full"])
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")), \
             patch(_MOD_PERSIST, return_value=persist):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        assert out["data"]["saved_draft"]["status"] == "error"

    def test_draft_still_ok_true_on_persistence_error(self):
        persist = _make_persist_result("error", summary="disk full")
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")), \
             patch(_MOD_PERSIST, return_value=persist):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        # draft_created is still the workflow result → ok=True
        assert out["ok"] is True

    def test_contract_satisfied_on_error(self):
        persist = _make_persist_result("error", summary="error")
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")), \
             patch(_MOD_PERSIST, return_value=persist):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        _assert_contract(out)


# ---------------------------------------------------------------------------
# save_draft=True + non-draft_created workflow status
# ---------------------------------------------------------------------------


class TestSaveDraftWithNonDraftCreatedStatus:
    @pytest.mark.parametrize("bad_status", [
        "planned", "needs_review", "unsupported_plan", "validation_failed",
        "needs_user_input", "invalid",
    ])
    def test_persistence_still_called_for_non_draft_status(self, bad_status: str):
        # Persistence bridge itself decides not to save; the MCP tool just calls it
        persist = _make_persist_result("not_saved", summary=f"Cannot persist {bad_status}")
        wf = _make_wf_result(bad_status)
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=wf), \
             patch(_MOD_PERSIST, return_value=persist) as mock_p:
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        mock_p.assert_called_once()
        assert out["data"]["saved_draft"]["status"] == "not_saved"

    @pytest.mark.parametrize("bad_status", [
        "unsupported_plan", "validation_failed", "needs_user_input", "invalid",
    ])
    def test_ok_false_on_non_draft_created(self, bad_status: str):
        persist = _make_persist_result("not_saved", summary="not saved")
        wf = _make_wf_result(bad_status)
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=wf), \
             patch(_MOD_PERSIST, return_value=persist):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        assert out["ok"] is False
