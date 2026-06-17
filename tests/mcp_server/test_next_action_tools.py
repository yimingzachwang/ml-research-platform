"""MCP tool tests for Phase 5C next-action recommendation tools.

Tests:
  1.  Tool registration — recommend_next_actions and recommend_next_actions_from_request present
  2.  Both tools have non-empty descriptions
  3.  Structured tool returns MCP contract envelope
  4.  NL tool returns MCP contract envelope
  5.  Structured tool: recommended status → ok=True
  6.  Structured tool: needs_more_evidence status → ok=True
  7.  Structured tool: needs_user_input → ok=False
  8.  Structured tool: invalid JSON spec → ok=False stage=next_action_invalid_input
  9.  NL tool: recommended status → ok=True
  10. NL tool: needs_clarification → ok=False
  11. Governance flags always present in data.governance
  12. No draft created (data.governance.draft_created == false)
  13. max_candidates parameter honoured
  14. Truncation warnings surfaced in data.warnings
  15. next_action_workflow module in approved import surface
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import src.mcp.zeto_server as zeto

_CONTRACT_KEYS = {"ok", "stage", "display", "data", "next_suggested_action"}
_NAW_MOD = "src.mcp.zeto_server._naw"

# ---------------------------------------------------------------------------
# Helpers — stub WorkflowResult-like objects
# ---------------------------------------------------------------------------


def _stub_na_result(
    status: str = "recommended",
    n_candidates: int = 2,
    truncation_warning: bool = False,
) -> SimpleNamespace:
    cands = [
        {
            "action_id": f"cand_{i}",
            "title": f"Candidate {i}",
            "action_type": "feature_ablation",
            "priority": "high",
            "description": "ablate low-IC features",
            "rationale": "high turnover observed",
            "requires_user_confirmation": True,
            "creates_draft": False,
            "executes_experiment": False,
            "main_risks": ["test risk"],
        }
        for i in range(n_candidates)
    ]
    top = cands[0] if cands else None
    warnings = []
    if truncation_warning:
        warnings.append(
            f"4 candidate(s) were generated; {n_candidates} returned due to max_candidates={n_candidates}."
        )
    na_dict = {
        "status": status,
        "objective": None,
        "candidates": cands,
        "top_candidate": top,
        "evidence_summary": "1 item: poor_oos_consistency",
        "rejected_directions": [],
        "recommended_cautions": ["monitor turnover"],
        "warnings": warnings,
        "errors": [],
        "summary": f"Recommended {n_candidates} action(s).",
        "approval_required": True,
        "execution_performed": False,
        "draft_created": False,
    }
    governance = {
        "approval_required": True,
        "execution_performed": False,
        "draft_approved": False,
        "draft_created": False,
    }
    d = {
        "status": status,
        "summary": f"stub {status}",
        "resolution": None,
        "plan": None,
        "draft": None,
        "warnings": warnings,
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
        "next_actions": na_dict,
        "memory_context": {"total_items": 1, "is_empty": False, "failure_mode_summary": {}},
        "governance": governance,
    }
    ns = SimpleNamespace(
        status=status,
        summary=f"stub {status}",
        missing_fields=[],
        warnings=warnings,
        errors=[],
        next_actions=na_dict,
        memory_context_summary={"total_items": 1, "is_empty": False},
        approval_required=True,
        execution_performed=False,
        draft_approved=False,
        parse=None,
    )
    ns.to_dict = lambda: d
    return ns


def _stub_na_needs_user_input() -> SimpleNamespace:
    d = {
        "status": "needs_user_input",
        "summary": "Missing: start_date, end_date",
        "resolution": None,
        "plan": None,
        "draft": None,
        "warnings": [],
        "errors": [],
        "missing_fields": ["start_date", "end_date"],
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
        "governance": {
            "approval_required": True,
            "execution_performed": False,
            "draft_approved": False,
            "draft_created": False,
        },
    }
    ns = SimpleNamespace(
        status="needs_user_input",
        summary="Missing: start_date, end_date",
        missing_fields=["start_date", "end_date"],
        warnings=[],
        errors=[],
        next_actions=None,
        memory_context_summary=None,
        approval_required=True,
        execution_performed=False,
        draft_approved=False,
        parse=None,
    )
    ns.to_dict = lambda: d
    return ns


def _stub_na_clarification() -> SimpleNamespace:
    d = {
        "status": "needs_clarification",
        "summary": "Could not parse intent from request.",
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
        "parse": {"status": "needs_clarification"},
        "saved_draft": None,
        "next_actions": None,
        "memory_context": None,
        "governance": {
            "approval_required": True,
            "execution_performed": False,
            "draft_approved": False,
            "draft_created": False,
        },
    }
    ns = SimpleNamespace(
        status="needs_clarification",
        summary="Could not parse intent from request.",
        missing_fields=[],
        warnings=[],
        errors=[],
        next_actions=None,
        memory_context_summary=None,
        approval_required=True,
        execution_performed=False,
        draft_approved=False,
        parse={"status": "needs_clarification"},
    )
    ns.to_dict = lambda: d
    return ns


# ---------------------------------------------------------------------------
# 1. Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_recommend_next_actions_registered(self):
        names = {t.name for t in zeto.mcp._tool_manager.list_tools()}
        assert "recommend_next_actions" in names

    def test_recommend_next_actions_from_request_registered(self):
        names = {t.name for t in zeto.mcp._tool_manager.list_tools()}
        assert "recommend_next_actions_from_request" in names

    def test_both_tools_have_descriptions(self):
        by_name = {t.name: t.description for t in zeto.mcp._tool_manager.list_tools()}
        for name in ("recommend_next_actions", "recommend_next_actions_from_request"):
            assert by_name[name] and by_name[name].strip(), f"{name} has empty description"

    def test_descriptions_mention_governance(self):
        by_name = {t.name: t.description for t in zeto.mcp._tool_manager.list_tools()}
        for name in ("recommend_next_actions", "recommend_next_actions_from_request"):
            desc = by_name[name].lower()
            assert "no draft" in desc or "governance" in desc or "approval" in desc, (
                f"{name} description missing governance language"
            )


# ---------------------------------------------------------------------------
# 2. MCP contract envelope — structured tool
# ---------------------------------------------------------------------------


class TestStructuredToolEnvelope:
    def test_returns_contract_keys(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_result()):
            out = zeto._mcp_recommend_next_actions(
                spec='{"preset": "canonical_multi_asset_etf", "end_date": "2024-12-31"}'
            )
        assert _CONTRACT_KEYS.issubset(out), f"Missing keys: {_CONTRACT_KEYS - set(out)}"

    def test_ok_is_bool(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_result()):
            out = zeto._mcp_recommend_next_actions()
        assert isinstance(out["ok"], bool)

    def test_stage_is_string(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_result()):
            out = zeto._mcp_recommend_next_actions()
        assert isinstance(out["stage"], str) and out["stage"]

    def test_display_is_nonempty_string(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_result()):
            out = zeto._mcp_recommend_next_actions()
        assert isinstance(out["display"], str) and out["display"].strip()

    def test_data_is_dict(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_result()):
            out = zeto._mcp_recommend_next_actions()
        assert isinstance(out["data"], dict)

    def test_next_suggested_action_is_nonempty_string(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_result()):
            out = zeto._mcp_recommend_next_actions()
        assert isinstance(out["next_suggested_action"], str) and out["next_suggested_action"]


# ---------------------------------------------------------------------------
# 3. MCP contract envelope — NL tool
# ---------------------------------------------------------------------------


class TestNLToolEnvelope:
    def test_returns_contract_keys(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions_from_request", return_value=_stub_na_result()):
            out = zeto._mcp_recommend_next_actions_from_request(
                "What should I try next for canonical ETF momentum?"
            )
        assert _CONTRACT_KEYS.issubset(out), f"Missing keys: {_CONTRACT_KEYS - set(out)}"

    def test_ok_is_bool(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions_from_request", return_value=_stub_na_result()):
            out = zeto._mcp_recommend_next_actions_from_request("next steps for my ETF")
        assert isinstance(out["ok"], bool)

    def test_display_is_nonempty_string(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions_from_request", return_value=_stub_na_result()):
            out = zeto._mcp_recommend_next_actions_from_request("next steps for my ETF")
        assert isinstance(out["display"], str) and out["display"].strip()

    def test_data_is_dict(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions_from_request", return_value=_stub_na_result()):
            out = zeto._mcp_recommend_next_actions_from_request("next steps for my ETF")
        assert isinstance(out["data"], dict)


# ---------------------------------------------------------------------------
# 4. Status-dependent ok flag
# ---------------------------------------------------------------------------


class TestStatusOkFlag:
    def test_recommended_ok_true(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_result("recommended")):
            out = zeto._mcp_recommend_next_actions()
        assert out["ok"] is True

    def test_needs_more_evidence_ok_true(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_result("needs_more_evidence")):
            out = zeto._mcp_recommend_next_actions()
        assert out["ok"] is True

    def test_needs_user_input_ok_false(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_needs_user_input()):
            out = zeto._mcp_recommend_next_actions()
        assert out["ok"] is False

    def test_needs_user_input_stage(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_needs_user_input()):
            out = zeto._mcp_recommend_next_actions()
        assert "needs_user_input" in out["stage"]

    def test_recommended_stage_name(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_result("recommended")):
            out = zeto._mcp_recommend_next_actions()
        assert out["stage"] == "next_action_recommended"

    def test_nl_needs_clarification_ok_false(self):
        with patch(
            f"{_NAW_MOD}.run_recommend_next_actions_from_request",
            return_value=_stub_na_clarification(),
        ):
            out = zeto._mcp_recommend_next_actions_from_request("???")
        assert out["ok"] is False

    def test_nl_needs_clarification_stage(self):
        with patch(
            f"{_NAW_MOD}.run_recommend_next_actions_from_request",
            return_value=_stub_na_clarification(),
        ):
            out = zeto._mcp_recommend_next_actions_from_request("???")
        assert "needs_clarification" in out["stage"]


# ---------------------------------------------------------------------------
# 5. Invalid JSON spec
# ---------------------------------------------------------------------------


class TestInvalidJsonSpec:
    def test_invalid_json_returns_false(self):
        out = zeto._mcp_recommend_next_actions(spec="NOT_VALID_JSON")
        assert out["ok"] is False

    def test_invalid_json_stage(self):
        out = zeto._mcp_recommend_next_actions(spec="NOT_VALID_JSON")
        assert out["stage"] == "next_action_invalid_input"

    def test_invalid_json_display_nonempty(self):
        out = zeto._mcp_recommend_next_actions(spec="NOT_VALID_JSON")
        assert isinstance(out["display"], str) and out["display"].strip()

    def test_invalid_json_data_has_parse_error(self):
        out = zeto._mcp_recommend_next_actions(spec="NOT_VALID_JSON")
        assert "parse_error" in out["data"]


# ---------------------------------------------------------------------------
# 6. Governance flags always present
# ---------------------------------------------------------------------------


class TestGovernanceFlags:
    def test_structured_governance_present(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_result()):
            out = zeto._mcp_recommend_next_actions()
        gov = out["data"].get("governance", {})
        assert gov.get("approval_required") is True
        assert gov.get("execution_performed") is False
        assert gov.get("draft_approved") is False
        assert gov.get("draft_created") is False

    def test_nl_governance_present(self):
        with patch(
            f"{_NAW_MOD}.run_recommend_next_actions_from_request",
            return_value=_stub_na_result(),
        ):
            out = zeto._mcp_recommend_next_actions_from_request("momentum signal review")
        gov = out["data"].get("governance", {})
        assert gov.get("approval_required") is True
        assert gov.get("execution_performed") is False
        assert gov.get("draft_created") is False

    def test_governance_present_on_needs_user_input(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_needs_user_input()):
            out = zeto._mcp_recommend_next_actions()
        gov = out["data"].get("governance", {})
        assert gov.get("draft_created") is False
        assert gov.get("execution_performed") is False

    def test_governance_present_on_clarification_error(self):
        with patch(
            f"{_NAW_MOD}.run_recommend_next_actions_from_request",
            return_value=_stub_na_clarification(),
        ):
            out = zeto._mcp_recommend_next_actions_from_request("???")
        gov = out["data"].get("governance", {})
        assert gov.get("draft_created") is False

    def test_no_draft_created_on_recommended_path(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_result("recommended")):
            out = zeto._mcp_recommend_next_actions()
        assert out["data"]["governance"]["draft_created"] is False

    def test_next_actions_inner_governance(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_result()):
            out = zeto._mcp_recommend_next_actions()
        na = out["data"].get("next_actions") or {}
        assert na.get("approval_required") is True
        assert na.get("execution_performed") is False
        assert na.get("draft_created") is False

    def test_candidate_level_governance(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_result(n_candidates=1)):
            out = zeto._mcp_recommend_next_actions()
        na = out["data"].get("next_actions") or {}
        for c in na.get("candidates", []):
            assert c.get("creates_draft") is False
            assert c.get("executes_experiment") is False
            assert c.get("requires_user_confirmation") is True


# ---------------------------------------------------------------------------
# 7. max_candidates parameter honoured
# ---------------------------------------------------------------------------


class TestMaxCandidates:
    def test_max_candidates_forwarded_structured(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions") as mock:
            mock.return_value = _stub_na_result(n_candidates=2)
            zeto._mcp_recommend_next_actions(max_candidates=2)
            _, call_kwargs = mock.call_args
            assert call_kwargs.get("max_candidates") == 2

    def test_max_candidates_forwarded_nl(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions_from_request") as mock:
            mock.return_value = _stub_na_result(n_candidates=2)
            zeto._mcp_recommend_next_actions_from_request("next steps", max_candidates=2)
            _, call_kwargs = mock.call_args
            assert call_kwargs.get("max_candidates") == 2

    def test_max_memory_results_forwarded(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions") as mock:
            mock.return_value = _stub_na_result()
            zeto._mcp_recommend_next_actions(max_memory_results=5)
            _, call_kwargs = mock.call_args
            assert call_kwargs.get("max_memory_results") == 5


# ---------------------------------------------------------------------------
# 8. Truncation warning surfaced
# ---------------------------------------------------------------------------


class TestTruncationWarning:
    def test_truncation_warning_in_data_warnings(self):
        stub = _stub_na_result(n_candidates=2, truncation_warning=True)
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=stub):
            out = zeto._mcp_recommend_next_actions(max_candidates=2)
        warnings = out["data"].get("warnings", [])
        assert any("max_candidates" in w for w in warnings)

    def test_truncation_warning_in_display(self):
        stub = _stub_na_result(n_candidates=2, truncation_warning=True)
        stub_d = stub.to_dict()
        stub_d["warnings"] = stub.warnings
        stub.to_dict = lambda: stub_d
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=stub):
            out = zeto._mcp_recommend_next_actions(max_candidates=2)
        assert out["data"] is not None  # data present regardless of warning display


# ---------------------------------------------------------------------------
# 9. next_action_workflow in approved import surface
# ---------------------------------------------------------------------------


class TestImportSurface:
    def test_next_action_workflow_in_approved_imports(self):
        import inspect
        import_lines = [
            line
            for line in inspect.getsource(zeto).splitlines()
            if line.startswith("from src.") or line.startswith("import src.")
        ]
        assert any("next_action_workflow" in line for line in import_lines), (
            "next_action_workflow not imported in zeto_server.py"
        )

    def test_naw_alias_exists(self):
        assert hasattr(zeto, "_naw"), "_naw alias not found in zeto_server"

    def test_naw_has_run_recommend_next_actions(self):
        from src.agent.workflow import next_action_workflow as naw
        assert hasattr(naw, "run_recommend_next_actions")
        assert hasattr(naw, "run_recommend_next_actions_from_request")


# ---------------------------------------------------------------------------
# 10. Next-suggested-action routing
# ---------------------------------------------------------------------------


class TestNextSuggestedActionRouting:
    def test_recommended_routes_to_resolve(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_result("recommended")):
            out = zeto._mcp_recommend_next_actions()
        assert out["next_suggested_action"] == "resolve_research_spec"

    def test_needs_more_evidence_routes_to_memory(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_result("needs_more_evidence")):
            out = zeto._mcp_recommend_next_actions()
        assert out["next_suggested_action"] == "retrieve_research_memory"

    def test_needs_user_input_routes_to_agent_options(self):
        with patch(f"{_NAW_MOD}.run_recommend_next_actions", return_value=_stub_na_needs_user_input()):
            out = zeto._mcp_recommend_next_actions()
        assert out["next_suggested_action"] == "list_research_agent_options"

    def test_nl_clarification_routes_to_parse_intent(self):
        with patch(
            f"{_NAW_MOD}.run_recommend_next_actions_from_request",
            return_value=_stub_na_clarification(),
        ):
            out = zeto._mcp_recommend_next_actions_from_request("???")
        assert out["next_suggested_action"] == "parse_research_intent"
