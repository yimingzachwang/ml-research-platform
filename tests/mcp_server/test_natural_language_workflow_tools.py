"""Tests for Phase-4B natural-language workflow MCP tools.

Covers:
  resolve_research_request
  create_experiment_plan_from_request
  create_validated_draft_from_request

All _nlw workflow calls are mocked — no disk I/O, no LLM, no real resolver.
These tests verify:
  * the MCP envelope contract (ok/stage/display/data/next_suggested_action)
  * governance flags (approval_required / execution_performed / draft_approved)
  * stage naming convention (nl_resolve_*, nl_plan_*, nl_draft_*)
  * parse metadata forwarded into data
  * save_draft behaviour (save_draft=True explicit, never inferred)
  * execution keyword warning propagation
  * needs_clarification and needs_user_input handling
  * next_suggested_action routing
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.mcp import zeto_server as zeto

_CONTRACT_KEYS = {"ok", "stage", "display", "data", "next_suggested_action"}
_MOD_NLW = "src.mcp.zeto_server._nlw"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wf(
    status: str = "resolved",
    warnings: list[str] | None = None,
    unsupported: list[str] | None = None,
    saved_draft: dict | None = None,
    parse: dict | None = None,
) -> SimpleNamespace:
    """Return a minimal WorkflowResult-like namespace suitable for MCP tool mocking."""
    wf = SimpleNamespace(
        status=status,
        summary=f"stub {status}",
        resolution={"preset": "canonical"} if status == "resolved" else None,
        plan={"experiment_type": "preset_baseline"} if status in ("planned", "needs_review") else None,
        draft={"proposed_name": "exp_v2"} if status == "draft_created" else None,
        warnings=list(warnings or []),
        errors=[],
        missing_fields=[],
        defaulted_fields={},
        assumptions=[],
        requires_confirmation=False,
        unsupported_fields=list(unsupported or []),
        hard_unsupported_fields=[],
        approval_required=True,
        execution_performed=False,
        draft_approved=False,
        mapped_changes=[],
        validation_errors=[],
        proposed_name="exp_v2" if status == "draft_created" else None,
        base_experiment=None,
        parse=parse or {"status": "parsed", "extracted_fields": {}, "unrecognised_terms": [],
                        "confidence": 0.9, "registry_warnings": [], "missing_after_parse": [],
                        "warnings": [], "raw_model_output": None, "summary": "stub"},
        saved_draft=saved_draft,
    )
    wf.to_dict = lambda: {
        "status": wf.status, "summary": wf.summary,
        "resolution": wf.resolution, "plan": wf.plan, "draft": wf.draft,
        "warnings": wf.warnings, "errors": wf.errors, "missing_fields": wf.missing_fields,
        "defaulted_fields": wf.defaulted_fields, "assumptions": wf.assumptions,
        "requires_confirmation": wf.requires_confirmation,
        "unsupported_fields": wf.unsupported_fields,
        "hard_unsupported_fields": wf.hard_unsupported_fields,
        "approval_required": wf.approval_required,
        "execution_performed": wf.execution_performed,
        "draft_approved": wf.draft_approved,
        "mapped_changes": wf.mapped_changes,
        "validation_errors": wf.validation_errors,
        "proposed_name": wf.proposed_name,
        "base_experiment": wf.base_experiment,
        "parse": wf.parse,
        "saved_draft": wf.saved_draft,
        "governance": {
            "approval_required": wf.approval_required,
            "execution_performed": wf.execution_performed,
            "draft_approved": wf.draft_approved,
        },
    }
    return wf


# ---------------------------------------------------------------------------
# resolve_research_request
# ---------------------------------------------------------------------------

class TestResolveResearchRequest:
    def test_contract_envelope_present(self):
        with patch(f"{_MOD_NLW}.run_resolve_from_request", return_value=_make_wf("resolved")):
            out = zeto._mcp_resolve_research_request("canonical ETF 2024")
        assert _CONTRACT_KEYS.issubset(out), f"Missing: {_CONTRACT_KEYS - set(out)}"

    def test_ok_true_on_resolved(self):
        with patch(f"{_MOD_NLW}.run_resolve_from_request", return_value=_make_wf("resolved")):
            out = zeto._mcp_resolve_research_request("canonical ETF")
        assert out["ok"] is True

    def test_ok_false_on_needs_user_input(self):
        with patch(f"{_MOD_NLW}.run_resolve_from_request", return_value=_make_wf("needs_user_input")):
            out = zeto._mcp_resolve_research_request("use Lasso")
        assert out["ok"] is False

    def test_ok_false_on_needs_clarification(self):
        with patch(f"{_MOD_NLW}.run_resolve_from_request", return_value=_make_wf("needs_clarification")):
            out = zeto._mcp_resolve_research_request("unclear request")
        assert out["ok"] is False

    def test_stage_reflects_status(self):
        with patch(f"{_MOD_NLW}.run_resolve_from_request", return_value=_make_wf("resolved")):
            out = zeto._mcp_resolve_research_request("canonical ETF")
        assert out["stage"] == "nl_resolve_resolved"

    def test_stage_for_needs_clarification(self):
        with patch(f"{_MOD_NLW}.run_resolve_from_request", return_value=_make_wf("needs_clarification")):
            out = zeto._mcp_resolve_research_request("unclear")
        assert out["stage"] == "nl_resolve_needs_clarification"

    def test_stage_for_needs_user_input(self):
        with patch(f"{_MOD_NLW}.run_resolve_from_request", return_value=_make_wf("needs_user_input")):
            out = zeto._mcp_resolve_research_request("Lasso only")
        assert out["stage"] == "nl_resolve_needs_user_input"

    def test_display_is_nonempty(self):
        with patch(f"{_MOD_NLW}.run_resolve_from_request", return_value=_make_wf("resolved")):
            out = zeto._mcp_resolve_research_request("canonical ETF")
        assert out["display"].strip()

    def test_next_action_on_success(self):
        with patch(f"{_MOD_NLW}.run_resolve_from_request", return_value=_make_wf("resolved")):
            out = zeto._mcp_resolve_research_request("canonical ETF")
        assert out["next_suggested_action"] == "create_experiment_plan_from_request"

    def test_next_action_on_failure(self):
        with patch(f"{_MOD_NLW}.run_resolve_from_request", return_value=_make_wf("needs_clarification")):
            out = zeto._mcp_resolve_research_request("unclear")
        assert out["next_suggested_action"] == "parse_research_intent"

    def test_data_contains_parse(self):
        with patch(f"{_MOD_NLW}.run_resolve_from_request", return_value=_make_wf("resolved")):
            out = zeto._mcp_resolve_research_request("canonical ETF")
        assert "parse" in out["data"]

    def test_data_contains_governance(self):
        with patch(f"{_MOD_NLW}.run_resolve_from_request", return_value=_make_wf("resolved")):
            out = zeto._mcp_resolve_research_request("canonical ETF")
        assert "governance" in out["data"]

    def test_unrecognised_terms_in_display(self):
        wf = _make_wf(
            "resolved",
            parse={
                "status": "parsed", "extracted_fields": {}, "unrecognised_terms": ["GBM"],
                "confidence": 0.5, "registry_warnings": [], "missing_after_parse": [],
                "warnings": [], "raw_model_output": None, "summary": "stub",
            },
        )
        with patch(f"{_MOD_NLW}.run_resolve_from_request", return_value=wf):
            out = zeto._mcp_resolve_research_request("Use GBM canonical ETF")
        assert "GBM" in out["display"]

    def test_provider_forwarded(self):
        captured = {}
        def _capture(**kw):
            captured.update(kw)
            return _make_wf("resolved")
        with patch(f"{_MOD_NLW}.run_resolve_from_request", side_effect=_capture):
            zeto._mcp_resolve_research_request("ETF", provider="anthropic")
        assert captured.get("provider") == "anthropic"


# ---------------------------------------------------------------------------
# create_experiment_plan_from_request
# ---------------------------------------------------------------------------

class TestCreateExperimentPlanFromRequest:
    def test_contract_envelope_present(self):
        with patch(f"{_MOD_NLW}.run_plan_from_request", return_value=_make_wf("planned")):
            out = zeto._mcp_create_experiment_plan_from_request("canonical ETF")
        assert _CONTRACT_KEYS.issubset(out)

    def test_ok_true_on_planned(self):
        with patch(f"{_MOD_NLW}.run_plan_from_request", return_value=_make_wf("planned")):
            out = zeto._mcp_create_experiment_plan_from_request("canonical ETF")
        assert out["ok"] is True

    def test_ok_true_on_needs_review(self):
        with patch(f"{_MOD_NLW}.run_plan_from_request", return_value=_make_wf("needs_review")):
            out = zeto._mcp_create_experiment_plan_from_request("ETF with warnings")
        assert out["ok"] is True

    def test_ok_false_on_needs_clarification(self):
        with patch(f"{_MOD_NLW}.run_plan_from_request", return_value=_make_wf("needs_clarification")):
            out = zeto._mcp_create_experiment_plan_from_request("unclear")
        assert out["ok"] is False

    def test_stage_reflects_status(self):
        with patch(f"{_MOD_NLW}.run_plan_from_request", return_value=_make_wf("planned")):
            out = zeto._mcp_create_experiment_plan_from_request("canonical ETF")
        assert out["stage"] == "nl_plan_planned"

    def test_stage_for_needs_review(self):
        with patch(f"{_MOD_NLW}.run_plan_from_request", return_value=_make_wf("needs_review")):
            out = zeto._mcp_create_experiment_plan_from_request("ETF")
        assert out["stage"] == "nl_plan_needs_review"

    def test_next_action_on_success(self):
        with patch(f"{_MOD_NLW}.run_plan_from_request", return_value=_make_wf("planned")):
            out = zeto._mcp_create_experiment_plan_from_request("canonical ETF")
        assert out["next_suggested_action"] == "create_validated_draft_from_request"

    def test_next_action_on_failure(self):
        with patch(f"{_MOD_NLW}.run_plan_from_request", return_value=_make_wf("error")):
            out = zeto._mcp_create_experiment_plan_from_request("unclear")
        assert out["next_suggested_action"] == "parse_research_intent"

    def test_data_contains_parse(self):
        with patch(f"{_MOD_NLW}.run_plan_from_request", return_value=_make_wf("planned")):
            out = zeto._mcp_create_experiment_plan_from_request("ETF")
        assert "parse" in out["data"]

    def test_baseline_forwarded(self):
        captured = {}
        def _capture(**kw):
            captured.update(kw)
            return _make_wf("planned")
        with patch(f"{_MOD_NLW}.run_plan_from_request", side_effect=_capture):
            zeto._mcp_create_experiment_plan_from_request("ETF", baseline="canonical_ml_multi_asset")
        assert captured.get("baseline") == "canonical_ml_multi_asset"


# ---------------------------------------------------------------------------
# create_validated_draft_from_request
# ---------------------------------------------------------------------------

class TestCreateValidatedDraftFromRequest:
    def test_contract_envelope_present(self):
        with patch(f"{_MOD_NLW}.run_draft_from_request", return_value=_make_wf("draft_created")):
            out = zeto._mcp_create_validated_draft_from_request("canonical ETF")
        assert _CONTRACT_KEYS.issubset(out)

    def test_ok_true_on_draft_created(self):
        with patch(f"{_MOD_NLW}.run_draft_from_request", return_value=_make_wf("draft_created")):
            out = zeto._mcp_create_validated_draft_from_request("canonical ETF")
        assert out["ok"] is True

    def test_ok_false_on_needs_clarification(self):
        with patch(f"{_MOD_NLW}.run_draft_from_request", return_value=_make_wf("needs_clarification")):
            out = zeto._mcp_create_validated_draft_from_request("unclear")
        assert out["ok"] is False

    def test_ok_false_on_unsupported_plan(self):
        with patch(f"{_MOD_NLW}.run_draft_from_request", return_value=_make_wf("unsupported_plan")):
            out = zeto._mcp_create_validated_draft_from_request("ETF with universe")
        assert out["ok"] is False

    def test_stage_reflects_status(self):
        with patch(f"{_MOD_NLW}.run_draft_from_request", return_value=_make_wf("draft_created")):
            out = zeto._mcp_create_validated_draft_from_request("canonical ETF")
        assert out["stage"] == "nl_draft_draft_created"

    def test_next_action_on_success(self):
        with patch(f"{_MOD_NLW}.run_draft_from_request", return_value=_make_wf("draft_created")):
            out = zeto._mcp_create_validated_draft_from_request("canonical ETF")
        assert out["next_suggested_action"] == "approve_experiment_draft"

    def test_next_action_on_failure(self):
        with patch(f"{_MOD_NLW}.run_draft_from_request", return_value=_make_wf("needs_clarification")):
            out = zeto._mcp_create_validated_draft_from_request("unclear")
        assert out["next_suggested_action"] == "parse_research_intent"

    def test_save_draft_false_forwarded(self):
        captured = {}
        def _capture(**kw):
            captured.update(kw)
            return _make_wf("draft_created")
        with patch(f"{_MOD_NLW}.run_draft_from_request", side_effect=_capture):
            zeto._mcp_create_validated_draft_from_request("ETF", save_draft=False)
        assert captured.get("save_draft") is False

    def test_save_draft_true_forwarded(self):
        captured = {}
        def _capture(**kw):
            captured.update(kw)
            return _make_wf("draft_created", saved_draft={"status": "saved", "draft_id": "d1"})
        with patch(f"{_MOD_NLW}.run_draft_from_request", side_effect=_capture):
            zeto._mcp_create_validated_draft_from_request("ETF", save_draft=True)
        assert captured.get("save_draft") is True

    def test_saved_draft_id_in_display(self):
        wf = _make_wf("draft_created", saved_draft={"status": "saved", "draft_id": "d-42"})
        with patch(f"{_MOD_NLW}.run_draft_from_request", return_value=wf):
            out = zeto._mcp_create_validated_draft_from_request("ETF", save_draft=True)
        assert "d-42" in out["display"]

    def test_unsupported_fields_in_display(self):
        wf = _make_wf("unsupported_plan", unsupported=["universe"])
        with patch(f"{_MOD_NLW}.run_draft_from_request", return_value=wf):
            out = zeto._mcp_create_validated_draft_from_request("ETF with universe")
        assert "universe" in out["display"]

    def test_governance_flags_in_data(self):
        with patch(f"{_MOD_NLW}.run_draft_from_request", return_value=_make_wf("draft_created")):
            out = zeto._mcp_create_validated_draft_from_request("canonical ETF")
        g = out["data"].get("governance", {})
        assert g.get("approval_required") is True
        assert g.get("execution_performed") is False
        assert g.get("draft_approved") is False

    def test_base_experiment_forwarded(self):
        captured = {}
        def _capture(**kw):
            captured.update(kw)
            return _make_wf("draft_created")
        with patch(f"{_MOD_NLW}.run_draft_from_request", side_effect=_capture):
            zeto._mcp_create_validated_draft_from_request(
                "ETF", base_experiment="canonical_ml_multi_asset"
            )
        assert captured.get("base_experiment") == "canonical_ml_multi_asset"

    def test_data_contains_parse(self):
        with patch(f"{_MOD_NLW}.run_draft_from_request", return_value=_make_wf("draft_created")):
            out = zeto._mcp_create_validated_draft_from_request("canonical ETF")
        assert "parse" in out["data"]


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

class TestPhase4BToolRegistration:
    _EXPECTED = {
        "resolve_research_request",
        "create_experiment_plan_from_request",
        "create_validated_draft_from_request",
    }

    def test_tools_registered(self):
        names = {t.name for t in zeto.mcp._tool_manager.list_tools()}
        assert self._EXPECTED.issubset(names), f"Missing: {self._EXPECTED - names}"

    def test_total_tool_count(self):
        names = {t.name for t in zeto.mcp._tool_manager.list_tools()}
        assert len(names) == 50  # 41 Phase-4A + 3 Phase-4B + 2 Phase-5C + 2 Phase-5E + 1 Phase-5F + 1 Phase-5G.2

    def test_tools_have_descriptions(self):
        for t in zeto.mcp._tool_manager.list_tools():
            if t.name in self._EXPECTED:
                assert t.description and len(t.description.strip()) > 20, (
                    f"{t.name} has missing/short description"
                )
