"""MCP tool tests for Phase 5E spec proposal tools.

All workflow calls are mocked — no disk I/O, no real resolver, no LLM.
These tests verify:

  1.  Tool registration — propose_spec_from_next_action and
      propose_spec_from_recommendation_request are registered
  2.  Both tools have non-empty descriptions mentioning governance
  3.  Tool count updated to 48
  4.  Direct candidate tool returns MCP contract envelope
  5.  Direct candidate tool: proposal_created → ok=True
  6.  Direct candidate tool: unsupported_action → ok=False
  7.  Direct candidate tool: invalid JSON candidate → ok=False
  8.  Direct candidate tool: missing candidate fields → ok=False
  9.  Direct candidate tool: governance flags always present in data.governance
  10. Direct candidate tool: draft_created always False in governance
  11. Direct candidate tool: execution_performed always False in governance
  12. Direct candidate tool: requires_user_confirmation always True
  13. NL request tool returns MCP contract envelope
  14. NL request tool: proposal_created → ok=True
  15. NL request tool: invalid_candidate_index → ok=False
  16. NL request tool: governance flags always present
  17. NL request tool: no draft created
  18. spec_proposal_workflow in approved import surface
  19. Unsupported feature family surfaced in response
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import src.mcp.zeto_server as zeto

_CONTRACT_KEYS = {"ok", "stage", "display", "data", "next_suggested_action"}
_SPW_MOD = "src.mcp.zeto_server._spw"

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


def _make_spec_proposal_data(
    status: str = "proposal_created",
    model: str = "LassoRegression",
    unsupported_changes: list[str] | None = None,
) -> dict:
    proposal = {
        "proposal_id": "p5e_test",
        "source_candidate_id": "act_001",
        "source_action_type": "model_comparison",
        "proposed_spec": {"preset": "canonical_multi_asset_etf", "model": model},
        "proposed_changes": {"model": model},
        "unchanged_fields": {"preset": "canonical_multi_asset_etf"},
        "rationale": "Test rationale",
        "supporting_evidence_ids": ["ev1"],
        "warnings": [],
        "unsupported_changes": unsupported_changes or [],
        "resolution_status": "resolved",
        "resolution_summary": "resolved",
        "requires_user_confirmation": True,
        "creates_draft": False,
        "executes_experiment": False,
        "approval_required": True,
        "base_spec": None,
    }
    return {
        "status": status,
        "summary": f"Spec proposal [{status}]. No draft created. No execution. Approval required.",
        "proposal": proposal if status == "proposal_created" else None,
        "proposed_spec": proposal.get("proposed_spec") if status == "proposal_created" else None,
        "proposed_changes": {"model": model} if status == "proposal_created" else None,
        "unchanged_fields": {"preset": "canonical_multi_asset_etf"} if status == "proposal_created" else None,
        "unsupported_changes": unsupported_changes or [],
        "supporting_evidence_ids": ["ev1"] if status == "proposal_created" else [],
        "resolution_status": "resolved" if status == "proposal_created" else None,
        "rationale": "Test rationale" if status == "proposal_created" else None,
        "warnings": [],
        "errors": [] if status == "proposal_created" else ["unsupported"],
        "governance": {
            "approval_required": True,
            "requires_user_confirmation": True,
            "draft_created": False,
            "execution_performed": False,
            "draft_approved": False,
        },
    }


def _stub_proposal_result(
    status: str = "proposal_created",
    model: str = "LassoRegression",
    unsupported_changes: list[str] | None = None,
) -> SimpleNamespace:
    sp = _make_spec_proposal_data(status, model, unsupported_changes)
    cand = {
        "action_id": "act_001",
        "action_type": "model_comparison",
        "title": "Test Lasso",
        "description": "Compare Lasso",
        "rationale": "test",
        "suggested_spec_changes": {"model": model},
        "supporting_evidence_ids": ["ev1"],
        "main_risks": [],
    }
    gov = {
        "approval_required": True,
        "requires_user_confirmation": True,
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
        "warnings": [],
        "errors": [] if status == "proposal_created" else ["All specified models are not in the registry."],
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
        "spec_proposal": sp,
        "selected_candidate": cand,
        "governance": gov,
    }
    ns = SimpleNamespace(
        status=status,
        summary=f"stub {status}",
        missing_fields=[],
        warnings=[],
        errors=d["errors"],
        spec_proposal=sp,
        selected_candidate=cand,
        next_actions=None,
        memory_context_summary=None,
        approval_required=True,
        execution_performed=False,
        draft_approved=False,
        parse=None,
    )
    ns.to_dict = lambda: d
    return ns


def _stub_nl_proposal_result(
    status: str = "proposal_created",
    n_candidates: int = 2,
) -> SimpleNamespace:
    cands = [
        {
            "action_id": f"cand_{i}",
            "action_type": "feature_ablation",
            "title": f"Candidate {i}",
            "description": "ablate features",
            "rationale": "test",
            "suggested_spec_changes": {"features": ["momentum"]},
            "supporting_evidence_ids": [],
            "main_risks": [],
            "priority": "medium",
            "confidence": "medium",
        }
        for i in range(n_candidates)
    ]
    na_dict = {
        "status": "recommended",
        "candidates": cands,
        "top_candidate": cands[0] if cands else None,
        "evidence_summary": "",
        "rejected_directions": [],
        "recommended_cautions": [],
        "warnings": [],
        "errors": [],
        "summary": "Recommended.",
        "approval_required": True,
        "execution_performed": False,
        "draft_created": False,
    }
    sp = _make_spec_proposal_data(status) if status == "proposal_created" else None
    sel_cand = cands[0] if cands else None
    gov = {
        "approval_required": True,
        "requires_user_confirmation": True,
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
        "next_actions": na_dict,
        "memory_context": None,
        "spec_proposal": sp,
        "selected_candidate": sel_cand,
        "governance": gov,
    }
    ns = SimpleNamespace(
        status=status,
        summary=f"stub {status}",
        missing_fields=[],
        warnings=[],
        errors=[],
        spec_proposal=sp,
        selected_candidate=sel_cand,
        next_actions=na_dict,
        memory_context_summary=None,
        approval_required=True,
        execution_performed=False,
        draft_approved=False,
        parse=None,
    )
    ns.to_dict = lambda: d
    return ns


def _stub_invalid_index_result() -> SimpleNamespace:
    d = {
        "status": "invalid_candidate_index",
        "summary": "candidate_index=5 is out of range. 2 candidate(s) available (indices 0–1).",
        "resolution": None,
        "plan": None,
        "draft": None,
        "warnings": [],
        "errors": ["candidate_index=5 out of range (0–1)."],
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
        "spec_proposal": None,
        "selected_candidate": None,
        "governance": {
            "approval_required": True,
            "requires_user_confirmation": True,
            "execution_performed": False,
            "draft_approved": False,
            "draft_created": False,
        },
    }
    ns = SimpleNamespace(
        status="invalid_candidate_index",
        summary=d["summary"],
        missing_fields=[],
        warnings=[],
        errors=d["errors"],
        spec_proposal=None,
        selected_candidate=None,
        next_actions=None,
        memory_context_summary=None,
        approval_required=True,
        execution_performed=False,
        draft_approved=False,
        parse=None,
    )
    ns.to_dict = lambda: d
    return ns


# ---------------------------------------------------------------------------
# Candidate JSON fixture
# ---------------------------------------------------------------------------

_LASSO_CANDIDATE = json.dumps({
    "action_id": "act_model_lasso",
    "action_type": "model_comparison",
    "title": "Test Lasso regularisation",
    "description": "Compare LassoRegression against the baseline.",
    "rationale": "Prior evidence suggests regularisation sensitivity.",
    "supporting_evidence_ids": ["ev1"],
    "suggested_spec_changes": {"model": "LassoRegression"},
    "main_risks": ["overfitting"],
    "priority": "high",
    "confidence": "medium",
})

_BASE_SPEC = json.dumps({
    "preset": "canonical_multi_asset_etf",
    "start_date": "2015-01-01",
    "end_date": "2024-12-31",
})


# ---------------------------------------------------------------------------
# 1. Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_propose_spec_from_next_action_registered(self):
        names = {t.name for t in zeto.mcp._tool_manager.list_tools()}
        assert "propose_spec_from_next_action" in names

    def test_propose_spec_from_recommendation_request_registered(self):
        names = {t.name for t in zeto.mcp._tool_manager.list_tools()}
        assert "propose_spec_from_recommendation_request" in names

    def test_both_tools_have_descriptions(self):
        by_name = {t.name: t.description for t in zeto.mcp._tool_manager.list_tools()}
        for name in ("propose_spec_from_next_action", "propose_spec_from_recommendation_request"):
            assert by_name[name] and by_name[name].strip(), f"{name} has empty description"

    def test_descriptions_mention_governance(self):
        by_name = {t.name: t.description for t in zeto.mcp._tool_manager.list_tools()}
        for name in ("propose_spec_from_next_action", "propose_spec_from_recommendation_request"):
            desc = by_name[name].lower()
            assert "no draft" in desc or "governance" in desc or "approval" in desc, (
                f"{name} description missing governance language"
            )

    def test_total_tool_count_is_50(self):
        names = {t.name for t in zeto.mcp._tool_manager.list_tools()}
        assert len(names) == 50  # 46 Phase-5C + 2 Phase-5E + 1 Phase-5F + 1 Phase-5G.2


# ---------------------------------------------------------------------------
# 2. Direct candidate tool — MCP contract
# ---------------------------------------------------------------------------


class TestDirectCandidateToolEnvelope:
    def test_returns_contract_keys(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_candidate", return_value=_stub_proposal_result()):
            out = zeto._mcp_propose_spec_from_next_action(candidate=_LASSO_CANDIDATE)
        assert _CONTRACT_KEYS.issubset(out), f"Missing keys: {_CONTRACT_KEYS - set(out)}"

    def test_ok_is_bool(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_candidate", return_value=_stub_proposal_result()):
            out = zeto._mcp_propose_spec_from_next_action(candidate=_LASSO_CANDIDATE)
        assert isinstance(out["ok"], bool)

    def test_stage_is_string(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_candidate", return_value=_stub_proposal_result()):
            out = zeto._mcp_propose_spec_from_next_action(candidate=_LASSO_CANDIDATE)
        assert isinstance(out["stage"], str) and out["stage"]

    def test_display_is_nonempty(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_candidate", return_value=_stub_proposal_result()):
            out = zeto._mcp_propose_spec_from_next_action(candidate=_LASSO_CANDIDATE)
        assert isinstance(out["display"], str) and out["display"].strip()

    def test_data_is_dict(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_candidate", return_value=_stub_proposal_result()):
            out = zeto._mcp_propose_spec_from_next_action(candidate=_LASSO_CANDIDATE)
        assert isinstance(out["data"], dict)

    def test_next_suggested_action_nonempty(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_candidate", return_value=_stub_proposal_result()):
            out = zeto._mcp_propose_spec_from_next_action(candidate=_LASSO_CANDIDATE)
        assert isinstance(out["next_suggested_action"], str) and out["next_suggested_action"]


# ---------------------------------------------------------------------------
# 3. Direct candidate tool — status → ok mapping
# ---------------------------------------------------------------------------


class TestDirectCandidateToolStatus:
    def test_proposal_created_is_ok(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_candidate", return_value=_stub_proposal_result("proposal_created")):
            out = zeto._mcp_propose_spec_from_next_action(candidate=_LASSO_CANDIDATE)
        assert out["ok"] is True

    def test_unsupported_action_is_not_ok(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_candidate", return_value=_stub_proposal_result("unsupported_action")):
            out = zeto._mcp_propose_spec_from_next_action(candidate=_LASSO_CANDIDATE)
        assert out["ok"] is False

    def test_needs_user_input_is_not_ok(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_candidate", return_value=_stub_proposal_result("needs_user_input")):
            out = zeto._mcp_propose_spec_from_next_action(candidate=_LASSO_CANDIDATE)
        assert out["ok"] is False

    def test_invalid_json_candidate_is_not_ok(self):
        out = zeto._mcp_propose_spec_from_next_action(candidate="{not valid json")
        assert out["ok"] is False

    def test_invalid_json_stage_is_error_stage(self):
        out = zeto._mcp_propose_spec_from_next_action(candidate="{bad}")
        assert "invalid" in out["stage"].lower() or "error" in out["stage"].lower()

    def test_missing_candidate_fields_is_not_ok(self):
        incomplete = json.dumps({"action_type": "model_comparison"})
        out = zeto._mcp_propose_spec_from_next_action(candidate=incomplete)
        assert out["ok"] is False


# ---------------------------------------------------------------------------
# 4. Direct candidate tool — governance in data
# ---------------------------------------------------------------------------


class TestDirectCandidateToolGovernance:
    def test_governance_dict_present_in_data(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_candidate", return_value=_stub_proposal_result()):
            out = zeto._mcp_propose_spec_from_next_action(candidate=_LASSO_CANDIDATE)
        assert "governance" in out["data"]

    def test_approval_required_true(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_candidate", return_value=_stub_proposal_result()):
            out = zeto._mcp_propose_spec_from_next_action(candidate=_LASSO_CANDIDATE)
        gov = out["data"]["governance"]
        assert gov["approval_required"] is True

    def test_draft_created_false(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_candidate", return_value=_stub_proposal_result()):
            out = zeto._mcp_propose_spec_from_next_action(candidate=_LASSO_CANDIDATE)
        gov = out["data"]["governance"]
        assert gov["draft_created"] is False

    def test_execution_performed_false(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_candidate", return_value=_stub_proposal_result()):
            out = zeto._mcp_propose_spec_from_next_action(candidate=_LASSO_CANDIDATE)
        gov = out["data"]["governance"]
        assert gov["execution_performed"] is False

    def test_requires_user_confirmation_true(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_candidate", return_value=_stub_proposal_result()):
            out = zeto._mcp_propose_spec_from_next_action(candidate=_LASSO_CANDIDATE)
        gov = out["data"]["governance"]
        assert gov.get("requires_user_confirmation") is True


# ---------------------------------------------------------------------------
# 5. NL request tool — MCP contract
# ---------------------------------------------------------------------------


class TestNLRequestToolEnvelope:
    def test_returns_contract_keys(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_request", return_value=_stub_nl_proposal_result()):
            out = zeto._mcp_propose_spec_from_recommendation_request(request="test")
        assert _CONTRACT_KEYS.issubset(out), f"Missing keys: {_CONTRACT_KEYS - set(out)}"

    def test_ok_is_bool(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_request", return_value=_stub_nl_proposal_result()):
            out = zeto._mcp_propose_spec_from_recommendation_request(request="test")
        assert isinstance(out["ok"], bool)

    def test_display_is_nonempty(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_request", return_value=_stub_nl_proposal_result()):
            out = zeto._mcp_propose_spec_from_recommendation_request(request="test")
        assert isinstance(out["display"], str) and out["display"].strip()

    def test_data_is_dict(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_request", return_value=_stub_nl_proposal_result()):
            out = zeto._mcp_propose_spec_from_recommendation_request(request="test")
        assert isinstance(out["data"], dict)


# ---------------------------------------------------------------------------
# 6. NL request tool — status mapping + governance
# ---------------------------------------------------------------------------


class TestNLRequestToolStatus:
    def test_proposal_created_is_ok(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_request", return_value=_stub_nl_proposal_result("proposal_created")):
            out = zeto._mcp_propose_spec_from_recommendation_request(request="test")
        assert out["ok"] is True

    def test_invalid_candidate_index_is_not_ok(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_request", return_value=_stub_invalid_index_result()):
            out = zeto._mcp_propose_spec_from_recommendation_request(request="test", candidate_index=5)
        assert out["ok"] is False

    def test_governance_present_in_nl_result(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_request", return_value=_stub_nl_proposal_result()):
            out = zeto._mcp_propose_spec_from_recommendation_request(request="test")
        assert "governance" in out["data"]

    def test_no_draft_in_nl_governance(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_request", return_value=_stub_nl_proposal_result()):
            out = zeto._mcp_propose_spec_from_recommendation_request(request="test")
        gov = out["data"]["governance"]
        assert gov["draft_created"] is False

    def test_no_execution_in_nl_governance(self):
        with patch(f"{_SPW_MOD}.run_propose_spec_from_request", return_value=_stub_nl_proposal_result()):
            out = zeto._mcp_propose_spec_from_recommendation_request(request="test")
        gov = out["data"]["governance"]
        assert gov["execution_performed"] is False


# ---------------------------------------------------------------------------
# 7. Import surface
# ---------------------------------------------------------------------------


class TestImportSurface:
    def test_spec_proposal_workflow_in_approved_imports(self):
        import inspect
        import_lines = [
            line
            for line in inspect.getsource(zeto).splitlines()
            if line.startswith("from src.") or line.startswith("import src.")
        ]
        assert any("spec_proposal_workflow" in line for line in import_lines), (
            "spec_proposal_workflow not imported in zeto_server.py"
        )
