"""Tests for the Phase 5E spec proposal workflow façade.

A. Direct candidate → spec proposal (run_propose_spec_from_candidate)

A1.  Valid model_comparison candidate → proposal_created
A2.  Proposed spec has the expected model
A3.  Supporting evidence IDs preserved in spec_proposal
A4.  Governance flags: no draft, no execution, approval required
A5.  Unsupported model → status=unsupported_action
A6.  Unsupported model not in proposed spec
A7.  Feature-family candidate (Trend) → proposal_created
A8.  Unsupported feature family (MagicAlpha) → unsupported_action / surfaced
A9.  Missing candidate fields → structured error result
A10. Governance flags cannot be overridden by candidate input
A11. base_spec applied when provided
A12. spec_proposal key present in WorkflowResult.to_dict()
A13. selected_candidate key present in WorkflowResult.to_dict()
A14. Result can be serialised to dict (no Python objects)

B. Natural-language request → spec proposal (run_propose_spec_from_request)

B1.  Valid request → returns WorkflowResult (any terminal status)
B2.  candidate_index=0 selects top candidate
B3.  candidate_index out of range → invalid_candidate_index
B4.  Negative candidate_index → invalid_candidate_index
B5.  next_actions preserved from recommendation step
B6.  parse metadata preserved from recommendation step
B7.  Governance flags always present
B8.  Status propagated cleanly (not a raw exception)

C. WorkflowResult serialisation

C1.  to_dict() contains spec_proposal key
C2.  to_dict() contains selected_candidate key
C3.  to_dict() governance dict contains requires_user_confirmation=True
C4.  to_dict() governance dict contains draft_created=False
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.agent.intent.research_spec import ResearchSpec
from src.agent.next_actions.next_action import NextActionCandidate
from src.agent.workflow.spec_proposal_workflow import (
    _parse_candidate_from_dict,
    run_propose_spec_from_candidate,
    run_propose_spec_from_request,
)
from src.agent.workflow.workflow_result import WorkflowResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NAW_MOD = "src.agent.workflow.spec_proposal_workflow.run_recommend_next_actions_from_request"


def _candidate(
    action_type: str = "model_comparison",
    suggested_spec_changes: dict | None = None,
    action_id: str = "act_001",
    title: str = "Test candidate",
    rationale: str = "Test rationale",
    supporting_evidence_ids: list[str] | None = None,
) -> NextActionCandidate:
    return NextActionCandidate(
        action_id=action_id,
        title=title,
        action_type=action_type,
        description=f"Test candidate for {action_type}",
        rationale=rationale,
        suggested_spec_changes=suggested_spec_changes or {},
        supporting_evidence_ids=supporting_evidence_ids or [],
        main_risks=["test risk"],
    )


def _base_spec(**kwargs) -> ResearchSpec:
    defaults = {"preset": "canonical_multi_asset_etf", "end_date": "2024-12-31"}
    return ResearchSpec(**{**defaults, **kwargs})


def _model_lasso_cand_dict() -> dict:
    return {
        "action_id": "act_model_lasso",
        "action_type": "model_comparison",
        "title": "Test Lasso regularisation",
        "description": "Compare LassoRegression against the baseline.",
        "rationale": "Prior evidence suggests regularisation sensitivity.",
        "supporting_evidence_ids": ["ev1"],
        "suggested_spec_changes": {"model": "LassoRegression"},
        "main_risks": ["overfitting"],
        "failure_modes_to_monitor": ["high_split_variance"],
        "priority": "high",
        "confidence": "medium",
    }


def _stub_na_result(
    status: str = "recommended",
    n_candidates: int = 2,
    action_type: str = "feature_ablation",
) -> SimpleNamespace:
    cands = [
        {
            "action_id": f"cand_{i}",
            "title": f"Candidate {i}",
            "action_type": action_type,
            "description": "ablate low-IC features",
            "rationale": "high turnover observed",
            "priority": "high",
            "confidence": "medium",
            "requires_user_confirmation": True,
            "creates_draft": False,
            "executes_experiment": False,
            "supporting_evidence_ids": [f"ev_{i}"],
            "suggested_spec_changes": {"features": ["momentum"]},
            "main_risks": [],
        }
        for i in range(n_candidates)
    ]
    top = cands[0] if cands else None
    na_dict = {
        "status": status,
        "candidates": cands,
        "top_candidate": top,
        "evidence_summary": "1 item",
        "rejected_directions": [],
        "recommended_cautions": [],
        "warnings": [],
        "errors": [],
        "summary": f"Recommended {n_candidates} action(s).",
        "approval_required": True,
        "execution_performed": False,
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
        status=status,
        summary=f"stub {status}",
        missing_fields=[],
        warnings=[],
        errors=[],
        next_actions=na_dict,
        memory_context_summary=None,
        resolution=None,
        parse=None,
        approval_required=True,
        execution_performed=False,
        draft_approved=False,
        spec_proposal=None,
        selected_candidate=None,
    )
    ns.to_dict = lambda: d
    return ns


# ---------------------------------------------------------------------------
# A. Direct candidate → spec proposal
# ---------------------------------------------------------------------------


class TestParseCandidateFromDict:
    def test_valid_dict_returns_candidate(self):
        d = _model_lasso_cand_dict()
        cand, err = _parse_candidate_from_dict(d)
        assert cand is not None
        assert err is None

    def test_action_type_preserved(self):
        d = _model_lasso_cand_dict()
        cand, _ = _parse_candidate_from_dict(d)
        assert cand.action_type == "model_comparison"

    def test_suggested_spec_changes_preserved(self):
        d = _model_lasso_cand_dict()
        cand, _ = _parse_candidate_from_dict(d)
        assert cand.suggested_spec_changes == {"model": "LassoRegression"}

    def test_evidence_ids_preserved(self):
        d = _model_lasso_cand_dict()
        cand, _ = _parse_candidate_from_dict(d)
        assert cand.supporting_evidence_ids == ["ev1"]

    def test_governance_always_safe(self):
        d = {**_model_lasso_cand_dict(), "creates_draft": True, "executes_experiment": True}
        cand, _ = _parse_candidate_from_dict(d)
        assert cand is not None
        assert cand.creates_draft is False
        assert cand.executes_experiment is False
        assert cand.requires_user_confirmation is True

    def test_missing_action_id_returns_error(self):
        d = {k: v for k, v in _model_lasso_cand_dict().items() if k != "action_id"}
        cand, err = _parse_candidate_from_dict(d)
        assert cand is None
        assert err is not None
        assert "action_id" in err

    def test_missing_action_type_returns_error(self):
        d = {k: v for k, v in _model_lasso_cand_dict().items() if k != "action_type"}
        cand, err = _parse_candidate_from_dict(d)
        assert cand is None
        assert err is not None

    def test_missing_title_returns_error(self):
        d = {k: v for k, v in _model_lasso_cand_dict().items() if k != "title"}
        cand, err = _parse_candidate_from_dict(d)
        assert cand is None
        assert err is not None

    def test_missing_description_returns_error(self):
        d = {k: v for k, v in _model_lasso_cand_dict().items() if k != "description"}
        cand, err = _parse_candidate_from_dict(d)
        assert cand is None
        assert err is not None

    def test_rationale_falls_back_to_description(self):
        d = {k: v for k, v in _model_lasso_cand_dict().items() if k != "rationale"}
        cand, err = _parse_candidate_from_dict(d)
        assert cand is not None
        assert cand.rationale == d["description"]


class TestRunProposeSpecFromCandidate:
    def test_proposal_created_lasso(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        assert result.status == "proposal_created"

    def test_proposed_spec_model_set(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        assert result.spec_proposal is not None
        proposed = result.spec_proposal.get("proposed_spec") or {}
        assert proposed.get("model") == "LassoRegression"

    def test_evidence_ids_preserved(self):
        c = _candidate(
            "model_comparison",
            {"model": "LassoRegression"},
            supporting_evidence_ids=["ev1", "ev2"],
        )
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        sp = result.spec_proposal or {}
        assert "ev1" in (sp.get("supporting_evidence_ids") or [])

    def test_governance_no_draft(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        assert result.approval_required is True
        assert result.execution_performed is False
        assert result.draft_approved is False
        sp = result.spec_proposal or {}
        gov = sp.get("governance") or {}
        assert gov.get("draft_created") is False
        assert gov.get("execution_performed") is False

    def test_governance_requires_user_confirmation(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        sp = result.spec_proposal or {}
        gov = sp.get("governance") or {}
        assert gov.get("requires_user_confirmation") is True
        assert gov.get("approval_required") is True

    def test_unsupported_model_status(self):
        c = _candidate("model_comparison", {"model": "RandomForest"})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        assert result.status == "unsupported_action"

    def test_unsupported_model_not_in_spec(self):
        c = _candidate("model_comparison", {"model": "RandomForest"})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        sp = result.spec_proposal or {}
        proposed = sp.get("proposed_spec") or {}
        assert proposed.get("model") != "RandomForest"

    def test_unsupported_model_surfaced_in_errors(self):
        c = _candidate("model_comparison", {"model": "RandomForest"})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        combined = " ".join(result.errors + result.warnings)
        assert "RandomForest" in combined or "unsupported" in combined.lower()

    def test_supported_feature_family_accepted(self):
        c = _candidate("feature_ablation", {"feature_families": ["Trend"]})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        assert result.status == "proposal_created"
        sp = result.spec_proposal or {}
        proposed = sp.get("proposed_spec") or {}
        assert "Trend" in (proposed.get("feature_families") or [])

    def test_unsupported_feature_family_rejected(self):
        c = _candidate("feature_ablation", {"feature_families": ["MagicAlpha"]})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        assert result.status == "unsupported_action"

    def test_unsupported_family_not_in_spec(self):
        c = _candidate("feature_ablation", {"feature_families": ["MagicAlpha"]})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        sp = result.spec_proposal or {}
        proposed = sp.get("proposed_spec") or {}
        assert "MagicAlpha" not in (proposed.get("feature_families") or [])

    def test_unsupported_family_surfaced(self):
        c = _candidate("feature_ablation", {"feature_families": ["MagicAlpha"]})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        combined = " ".join(result.errors + result.warnings)
        assert "MagicAlpha" in combined or "unsupported" in combined.lower()

    def test_spec_proposal_key_in_to_dict(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        d = result.to_dict()
        assert "spec_proposal" in d

    def test_selected_candidate_key_in_to_dict(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        d = result.to_dict()
        assert "selected_candidate" in d
        assert d["selected_candidate"] is not None

    def test_to_dict_no_python_objects(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        d = result.to_dict()
        for key, val in d.items():
            assert not hasattr(val, "to_dict"), (
                f"to_dict() still contains a Python object at key '{key}'"
            )

    def test_base_spec_applied_to_diff(self):
        base = _base_spec(model="RidgeRegression")
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = run_propose_spec_from_candidate(c, base_spec=base)
        assert result.status == "proposal_created"
        sp = result.spec_proposal or {}
        proposed = sp.get("proposed_spec") or {}
        assert proposed.get("model") == "LassoRegression"

    def test_no_base_spec_still_works(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = run_propose_spec_from_candidate(c)
        assert result.status in ("proposal_created", "needs_user_input", "error")

    def test_result_is_workflow_result(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        assert isinstance(result, WorkflowResult)


# ---------------------------------------------------------------------------
# B. Natural-language request → spec proposal
# ---------------------------------------------------------------------------


class TestRunProposeSpecFromRequest:
    def test_propagates_non_recommendation_status(self):
        stub = _stub_na_result("needs_user_input", n_candidates=0)
        stub.status = "needs_user_input"
        stub.next_actions = None
        with patch(_NAW_MOD, return_value=stub):
            result = run_propose_spec_from_request("some request")
        assert result.status == "needs_user_input"

    def test_propagates_needs_clarification(self):
        stub = _stub_na_result("needs_clarification", n_candidates=0)
        stub.status = "needs_clarification"
        stub.next_actions = None
        with patch(_NAW_MOD, return_value=stub):
            result = run_propose_spec_from_request("some request")
        assert result.status == "needs_clarification"

    def test_valid_request_returns_workflow_result(self):
        stub = _stub_na_result("recommended", n_candidates=2)
        with patch(_NAW_MOD, return_value=stub):
            result = run_propose_spec_from_request("test request for model comparison")
        assert isinstance(result, WorkflowResult)

    def test_candidate_index_0_selects_first(self):
        stub = _stub_na_result("recommended", n_candidates=2)
        with patch(_NAW_MOD, return_value=stub):
            result = run_propose_spec_from_request("test", candidate_index=0)
        assert result.selected_candidate is not None
        assert result.selected_candidate.get("action_id") == "cand_0"

    def test_candidate_index_1_selects_second(self):
        stub = _stub_na_result("recommended", n_candidates=2)
        with patch(_NAW_MOD, return_value=stub):
            result = run_propose_spec_from_request("test", candidate_index=1)
        assert result.selected_candidate is not None
        assert result.selected_candidate.get("action_id") == "cand_1"

    def test_out_of_range_index_returns_error_status(self):
        stub = _stub_na_result("recommended", n_candidates=2)
        with patch(_NAW_MOD, return_value=stub):
            result = run_propose_spec_from_request("test", candidate_index=5)
        assert result.status == "invalid_candidate_index"

    def test_negative_index_returns_error_status(self):
        stub = _stub_na_result("recommended", n_candidates=2)
        with patch(_NAW_MOD, return_value=stub):
            result = run_propose_spec_from_request("test", candidate_index=-1)
        assert result.status == "invalid_candidate_index"

    def test_index_error_contains_range_info(self):
        stub = _stub_na_result("recommended", n_candidates=2)
        with patch(_NAW_MOD, return_value=stub):
            result = run_propose_spec_from_request("test", candidate_index=99)
        combined = " ".join(result.errors + [result.summary])
        assert "99" in combined or "range" in combined.lower()

    def test_next_actions_preserved(self):
        stub = _stub_na_result("recommended", n_candidates=2)
        with patch(_NAW_MOD, return_value=stub):
            result = run_propose_spec_from_request("test", candidate_index=0)
        assert result.next_actions is not None

    def test_governance_always_present(self):
        stub = _stub_na_result("recommended", n_candidates=2)
        with patch(_NAW_MOD, return_value=stub):
            result = run_propose_spec_from_request("test")
        assert result.approval_required is True
        assert result.execution_performed is False
        assert result.draft_approved is False

    def test_no_candidates_returns_needs_more_evidence(self):
        stub = _stub_na_result("needs_more_evidence", n_candidates=0)
        stub.next_actions = {
            "status": "needs_more_evidence",
            "candidates": [],
            "top_candidate": None,
            "evidence_summary": "",
            "rejected_directions": [],
            "recommended_cautions": [],
            "warnings": [],
            "errors": [],
            "summary": "no evidence",
            "approval_required": True,
            "execution_performed": False,
            "draft_created": False,
        }
        with patch(_NAW_MOD, return_value=stub):
            result = run_propose_spec_from_request("test")
        assert result.status == "needs_more_evidence"

    def test_result_status_is_string(self):
        stub = _stub_na_result("recommended", n_candidates=2)
        with patch(_NAW_MOD, return_value=stub):
            result = run_propose_spec_from_request("test")
        assert isinstance(result.status, str)


# ---------------------------------------------------------------------------
# C. WorkflowResult serialisation
# ---------------------------------------------------------------------------


class TestWorkflowResultSerialisation:
    def test_to_dict_contains_spec_proposal(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        d = result.to_dict()
        assert "spec_proposal" in d

    def test_to_dict_contains_selected_candidate(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        d = result.to_dict()
        assert "selected_candidate" in d

    def test_governance_requires_user_confirmation(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        d = result.to_dict()
        gov = d.get("governance") or {}
        assert gov.get("requires_user_confirmation") is True

    def test_governance_draft_created_false(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        d = result.to_dict()
        gov = d.get("governance") or {}
        assert gov.get("draft_created") is False

    def test_governance_execution_performed_false(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = run_propose_spec_from_candidate(c, base_spec=_base_spec())
        d = result.to_dict()
        gov = d.get("governance") or {}
        assert gov.get("execution_performed") is False

    def test_empty_workflow_result_has_both_keys(self):
        wr = WorkflowResult(
            status="test",
            summary="test",
            approval_required=True,
            execution_performed=False,
            draft_approved=False,
        )
        d = wr.to_dict()
        assert "spec_proposal" in d
        assert "selected_candidate" in d
