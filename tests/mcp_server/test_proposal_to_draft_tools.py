"""MCP tool tests for Phase 5F proposal-to-draft tool.

All workflow calls are mocked — no disk I/O, no real resolver, no LLM.
These tests verify:

  B1.  Tool registered, total count=49
  B2.  Tool has non-empty description mentioning governance
  B3.  confirm_proposal=False → confirmation_required, ok=False
  B4.  confirm_proposal=True, valid proposal → draft preview, ok=True
  B5.  save_draft=True → saved note in display / governance
  B6.  Unsupported proposal does not create draft, ok=False
  B7.  Invalid JSON proposal → structured error, ok=False
  B8.  Governance flags always present in data.governance
  B9.  proposal_to_draft_workflow in approved import surface
  B10. Invalid governance proposal → invalid_proposal, ok=False
  B11. confirmation_required stage name
  B12. draft_created stage name
  B13. next_action after draft_created is approve_experiment_draft
  B14. next_action after confirmation_required is propose_spec_from_next_action
  B15. requires_user_confirmation=True in governance always
  B16. draft_approved=False always
  B17. execution_performed=False always
  B18. Hard-unsupported blocked draft → ok=False, unsupported_plan stage
  B19. Malformed but valid JSON (empty dict) is handled gracefully
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import src.mcp.zeto_server as zeto

_CONTRACT_KEYS = {"ok", "stage", "display", "data", "next_suggested_action"}
_PTDW_MOD = "src.mcp.zeto_server._ptdw"

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


def _proposal_json(
    model: str = "LassoRegression",
    unsupported_changes: list[str] | None = None,
) -> str:
    d = {
        "proposal_id": "p5f_test",
        "source_candidate_id": "cand_001",
        "source_action_type": "model_comparison",
        "base_spec": {"preset": "canonical_multi_asset_etf", "end_date": "2024-12-31"},
        "proposed_spec": {
            "preset": "canonical_multi_asset_etf",
            "end_date": "2024-12-31",
            "model": model,
        },
        "proposed_changes": {"model": model},
        "unchanged_fields": {"preset": "canonical_multi_asset_etf", "end_date": "2024-12-31"},
        "rationale": "test rationale",
        "supporting_evidence_ids": ["ev1"],
        "warnings": [],
        "unsupported_changes": unsupported_changes or [],
        "resolution_status": "resolved",
        "resolution_summary": "resolved",
        "requires_user_confirmation": True,
        "creates_draft": False,
        "executes_experiment": False,
        "approval_required": True,
    }
    return json.dumps(d)


def _stub_workflow_result(status: str = "draft_created", save_draft: bool = False) -> SimpleNamespace:
    prop_name = "canonical_ml_multi_asset_v2" if status == "draft_created" else None
    sd = (
        {
            "status": "saved",
            "draft_id": "d_stub",
            "draft_path": "results/drafts/d_stub.json",
            "base_experiment": "canonical_ml_multi_asset",
            "proposed_name": prop_name,
            "approval_required": True,
            "execution_performed": False,
            "draft_approved": False,
            "warnings": [],
            "errors": [],
            "summary": "Draft saved.",
        }
        if save_draft and status == "draft_created"
        else None
    )
    draft_saved = save_draft and status == "draft_created"
    gov = {
        "approval_required": True,
        "requires_user_confirmation": True,
        "execution_performed": False,
        "draft_approved": False,
        "draft_created": status == "draft_created",
        "draft_saved": draft_saved,
    }
    d = {
        "status": status,
        "summary": f"stub [{status}].",
        "resolution": {"status": "resolved", "resolved_spec": {}, "missing_fields": []} if status == "draft_created" else None,
        "plan": {"plan_id": "plan_x"} if status == "draft_created" else None,
        "draft": {"draft_id": "d_stub", "proposed_name": prop_name, "changes": []} if status == "draft_created" else None,
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
        "proposed_name": prop_name,
        "base_experiment": "canonical_ml_multi_asset",
        "parse": None,
        "saved_draft": sd,
        "next_actions": None,
        "memory_context": None,
        "spec_proposal": {},
        "selected_candidate": None,
        "governance": gov,
    }
    ns = SimpleNamespace(
        status=status,
        summary=f"stub [{status}].",
        missing_fields=[],
        warnings=[],
        errors=[],
        spec_proposal={},
        selected_candidate=None,
        next_actions=None,
        memory_context_summary=None,
        approval_required=True,
        execution_performed=False,
        draft_approved=False,
        draft_saved=draft_saved,
        draft=d["draft"],
        saved_draft=sd,
        proposed_name=prop_name,
        unsupported_fields=[],
        hard_unsupported_fields=[],
        parse=None,
    )
    ns.to_dict = lambda: d
    return ns


def _stub_confirmation_required() -> SimpleNamespace:
    gov = {
        "approval_required": True,
        "requires_user_confirmation": True,
        "execution_performed": False,
        "draft_approved": False,
        "draft_created": False,
        "draft_saved": False,
    }
    d = {
        "status": "confirmation_required",
        "summary": "Explicit confirmation is required before creating a draft.",
        "resolution": None,
        "plan": None,
        "draft": None,
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
        "proposed_name": None,
        "base_experiment": None,
        "parse": None,
        "saved_draft": None,
        "next_actions": None,
        "memory_context": None,
        "spec_proposal": {},
        "selected_candidate": None,
        "governance": gov,
    }
    ns = SimpleNamespace(
        status="confirmation_required",
        summary=d["summary"],
        missing_fields=[], warnings=[], errors=[],
        spec_proposal={}, selected_candidate=None,
        next_actions=None, memory_context_summary=None,
        approval_required=True, execution_performed=False,
        draft_approved=False, draft_saved=False, draft=None,
        saved_draft=None, proposed_name=None,
        unsupported_fields=[], hard_unsupported_fields=[],
        parse=None,
    )
    ns.to_dict = lambda: d
    return ns


def _stub_unsupported_proposal() -> SimpleNamespace:
    gov = {
        "approval_required": True,
        "requires_user_confirmation": True,
        "execution_performed": False,
        "draft_approved": False,
        "draft_created": False,
        "draft_saved": False,
    }
    d = {
        "status": "unsupported_proposal",
        "summary": "1 unsupported change(s).",
        "resolution": None,
        "plan": None,
        "draft": None,
        "warnings": ["Unsupported change: bad_model"],
        "errors": ["bad_model"],
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
        "proposed_name": None,
        "base_experiment": None,
        "parse": None,
        "saved_draft": None,
        "next_actions": None,
        "memory_context": None,
        "spec_proposal": {},
        "selected_candidate": None,
        "governance": gov,
    }
    ns = SimpleNamespace(
        status="unsupported_proposal",
        summary=d["summary"],
        missing_fields=[], warnings=d["warnings"], errors=d["errors"],
        spec_proposal={}, selected_candidate=None,
        next_actions=None, memory_context_summary=None,
        approval_required=True, execution_performed=False,
        draft_approved=False, draft_saved=False, draft=None,
        saved_draft=None, proposed_name=None,
        unsupported_fields=[], hard_unsupported_fields=[],
        parse=None,
    )
    ns.to_dict = lambda: d
    return ns


def _stub_unsupported_plan() -> SimpleNamespace:
    gov = {
        "approval_required": True,
        "requires_user_confirmation": True,
        "execution_performed": False,
        "draft_approved": False,
        "draft_created": False,
        "draft_saved": False,
    }
    d = {
        "status": "unsupported_plan",
        "summary": "Hard-unsupported fields blocked draft.",
        "resolution": None,
        "plan": None,
        "draft": None,
        "warnings": [],
        "errors": ["universe not supported"],
        "missing_fields": [],
        "defaulted_fields": {},
        "assumptions": [],
        "requires_confirmation": False,
        "unsupported_fields": [],
        "hard_unsupported_fields": ["universe"],
        "approval_required": True,
        "execution_performed": False,
        "draft_approved": False,
        "mapped_changes": [],
        "validation_errors": [],
        "proposed_name": None,
        "base_experiment": None,
        "parse": None,
        "saved_draft": None,
        "next_actions": None,
        "memory_context": None,
        "spec_proposal": {},
        "selected_candidate": None,
        "governance": gov,
    }
    ns = SimpleNamespace(
        status="unsupported_plan",
        summary=d["summary"],
        missing_fields=[], warnings=[], errors=d["errors"],
        spec_proposal={}, selected_candidate=None,
        next_actions=None, memory_context_summary=None,
        approval_required=True, execution_performed=False,
        draft_approved=False, draft_saved=False, draft=None,
        saved_draft=None, proposed_name=None,
        unsupported_fields=[], hard_unsupported_fields=["universe"],
        parse=None,
    )
    ns.to_dict = lambda: d
    return ns


# ---------------------------------------------------------------------------
# B1-B2. Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_create_validated_draft_from_spec_proposal_registered(self):
        names = {t.name for t in zeto.mcp._tool_manager.list_tools()}
        assert "create_validated_draft_from_spec_proposal" in names

    def test_total_tool_count_is_50(self):
        names = {t.name for t in zeto.mcp._tool_manager.list_tools()}
        assert len(names) == 50  # +1 Phase-5G.2 route_research_request

    def test_tool_has_description(self):
        by_name = {t.name: t.description for t in zeto.mcp._tool_manager.list_tools()}
        desc = by_name.get("create_validated_draft_from_spec_proposal", "")
        assert desc and desc.strip()

    def test_description_mentions_governance(self):
        by_name = {t.name: t.description for t in zeto.mcp._tool_manager.list_tools()}
        desc = by_name.get("create_validated_draft_from_spec_proposal", "").lower()
        assert (
            "governance" in desc
            or "no draft is approved" in desc
            or "approval" in desc
            or "confirm" in desc
        )


# ---------------------------------------------------------------------------
# B3. confirm_proposal=False → confirmation_required
# ---------------------------------------------------------------------------


class TestConfirmationGate:
    def test_no_confirm_returns_not_ok(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_confirmation_required()):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=False
            )
        assert out["ok"] is False

    def test_no_confirm_stage(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_confirmation_required()):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=False
            )
        assert "confirmation_required" in out["stage"]

    def test_no_confirm_has_contract_keys(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_confirmation_required()):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=False
            )
        assert _CONTRACT_KEYS.issubset(out)

    def test_no_confirm_next_action(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_confirmation_required()):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=False
            )
        assert out["next_suggested_action"] == "propose_spec_from_next_action"

    def test_no_confirm_display_mentions_confirmation(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_confirmation_required()):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=False
            )
        assert "confirm" in out["display"].lower()


# ---------------------------------------------------------------------------
# B4. confirm_proposal=True → draft_created
# ---------------------------------------------------------------------------


class TestConfirmedDraft:
    def test_confirmed_valid_proposal_is_ok(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_workflow_result("draft_created")):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=True
            )
        assert out["ok"] is True

    def test_confirmed_valid_proposal_stage(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_workflow_result("draft_created")):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=True
            )
        assert "draft_created" in out["stage"]

    def test_confirmed_next_action_is_approve(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_workflow_result("draft_created")):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=True
            )
        assert out["next_suggested_action"] == "approve_experiment_draft"

    def test_confirmed_has_contract_keys(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_workflow_result("draft_created")):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=True
            )
        assert _CONTRACT_KEYS.issubset(out)

    def test_confirmed_data_is_dict(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_workflow_result("draft_created")):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=True
            )
        assert isinstance(out["data"], dict)


# ---------------------------------------------------------------------------
# B5. save_draft=True → saved note in display / governance
# ---------------------------------------------------------------------------


class TestSaveDraft:
    def test_save_draft_true_reflects_in_governance(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal",
                   return_value=_stub_workflow_result("draft_created", save_draft=True)):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=True, save_draft=True
            )
        gov = out["data"].get("governance") or {}
        assert gov.get("draft_saved") is True

    def test_save_draft_false_not_saved_in_governance(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal",
                   return_value=_stub_workflow_result("draft_created", save_draft=False)):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=True, save_draft=False
            )
        gov = out["data"].get("governance") or {}
        assert gov.get("draft_saved") is False


# ---------------------------------------------------------------------------
# B6. Unsupported proposal does not create draft
# ---------------------------------------------------------------------------


class TestUnsupportedProposal:
    def test_unsupported_proposal_is_not_ok(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_unsupported_proposal()):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(unsupported_changes=["bad_model"]),
                confirm_proposal=True,
            )
        assert out["ok"] is False

    def test_unsupported_proposal_stage(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_unsupported_proposal()):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(unsupported_changes=["bad_model"]),
                confirm_proposal=True,
            )
        assert "unsupported_proposal" in out["stage"]

    def test_unsupported_proposal_no_draft_in_data(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_unsupported_proposal()):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(unsupported_changes=["bad_model"]),
                confirm_proposal=True,
            )
        assert out["data"].get("draft") is None


# ---------------------------------------------------------------------------
# B7. Invalid JSON proposal → structured error
# ---------------------------------------------------------------------------


class TestInvalidJSON:
    def test_invalid_json_is_not_ok(self):
        out = zeto._mcp_create_validated_draft_from_spec_proposal(
            proposal="{not valid json", confirm_proposal=True
        )
        assert out["ok"] is False

    def test_invalid_json_has_contract_keys(self):
        out = zeto._mcp_create_validated_draft_from_spec_proposal(
            proposal="{bad json}", confirm_proposal=True
        )
        assert _CONTRACT_KEYS.issubset(out)

    def test_invalid_json_stage_indicates_invalid_input(self):
        out = zeto._mcp_create_validated_draft_from_spec_proposal(
            proposal="{bad}", confirm_proposal=True
        )
        assert "invalid" in out["stage"].lower() or "error" in out["stage"].lower()


# ---------------------------------------------------------------------------
# B8. Governance flags always present
# ---------------------------------------------------------------------------


class TestGovernanceFlags:
    def test_governance_dict_present_in_data(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_confirmation_required()):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=False
            )
        assert "governance" in out["data"]

    def test_approval_required_true(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_confirmation_required()):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=False
            )
        assert out["data"]["governance"]["approval_required"] is True

    def test_requires_user_confirmation_true(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_confirmation_required()):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=False
            )
        assert out["data"]["governance"]["requires_user_confirmation"] is True

    def test_draft_approved_false(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_workflow_result("draft_created")):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=True
            )
        assert out["data"]["governance"]["draft_approved"] is False

    def test_execution_performed_false(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_workflow_result("draft_created")):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=True
            )
        assert out["data"]["governance"]["execution_performed"] is False


# ---------------------------------------------------------------------------
# B9. Import surface
# ---------------------------------------------------------------------------


class TestImportSurface:
    def test_proposal_to_draft_workflow_in_approved_imports(self):
        import inspect
        import_lines = [
            line
            for line in inspect.getsource(zeto).splitlines()
            if line.startswith("from src.") or line.startswith("import src.")
        ]
        assert any("proposal_to_draft_workflow" in line for line in import_lines), (
            "proposal_to_draft_workflow not imported in zeto_server.py"
        )


# ---------------------------------------------------------------------------
# B10. Invalid governance proposal → invalid_proposal
# ---------------------------------------------------------------------------


class TestInvalidGovernanceProposal:
    def test_invalid_governance_is_not_ok(self):
        stub = SimpleNamespace(
            status="invalid_proposal",
            summary="Proposal has invalid governance: creates_draft must be False. No draft created.",
            missing_fields=[], warnings=[], errors=["Invalid proposal governance: creates_draft must be False"],
            spec_proposal={}, selected_candidate=None,
            next_actions=None, memory_context_summary=None,
            approval_required=True, execution_performed=False,
            draft_approved=False, draft_saved=False, draft=None,
            saved_draft=None, proposed_name=None,
            unsupported_fields=[], hard_unsupported_fields=[],
            parse=None,
        )
        stub.to_dict = lambda: {
            "status": "invalid_proposal",
            "summary": stub.summary,
            "resolution": None, "plan": None, "draft": None,
            "warnings": [], "errors": stub.errors,
            "missing_fields": [], "defaulted_fields": {}, "assumptions": [],
            "requires_confirmation": False, "unsupported_fields": [], "hard_unsupported_fields": [],
            "approval_required": True, "execution_performed": False, "draft_approved": False,
            "mapped_changes": [], "validation_errors": [], "proposed_name": None,
            "base_experiment": None, "parse": None, "saved_draft": None,
            "next_actions": None, "memory_context": None, "spec_proposal": {}, "selected_candidate": None,
            "governance": {
                "approval_required": True, "requires_user_confirmation": True,
                "execution_performed": False, "draft_approved": False,
                "draft_created": False, "draft_saved": False,
            },
        }
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=stub):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=True
            )
        assert out["ok"] is False
        assert "invalid_proposal" in out["stage"]


# ---------------------------------------------------------------------------
# B18. Hard-unsupported blocked draft
# ---------------------------------------------------------------------------


class TestHardUnsupportedPlan:
    def test_unsupported_plan_is_not_ok(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_unsupported_plan()):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=True
            )
        assert out["ok"] is False

    def test_unsupported_plan_stage(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_unsupported_plan()):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=True
            )
        assert "unsupported_plan" in out["stage"]

    def test_unsupported_plan_next_action(self):
        with patch(f"{_PTDW_MOD}.run_draft_from_spec_proposal", return_value=_stub_unsupported_plan()):
            out = zeto._mcp_create_validated_draft_from_spec_proposal(
                proposal=_proposal_json(), confirm_proposal=True
            )
        assert out["next_suggested_action"] == "propose_spec_from_next_action"
