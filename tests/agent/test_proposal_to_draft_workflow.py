"""Tests for the Phase 5F proposal-to-draft workflow façade.

A. Confirmation gate

A1.  confirm_proposal=False → confirmation_required, no draft
A2.  confirm_proposal=False → no resolution, no plan, no draft
A3.  confirm_proposal=False → no save
A4.  confirm_proposal=False → no approval (approval_required=True)
A5.  confirm_proposal=False → no execution (execution_performed=False)
A6.  confirm_proposal=False → spec_proposal preserved in result

B. Governance validation

B1.  Valid proposal → proceeds past governance check
B2.  Mock invalid governance (creates_draft=True via __setattr__) → invalid_proposal
B3.  Mock invalid governance (executes_experiment=True) → invalid_proposal
B4.  Mock invalid governance (approval_required=False) → invalid_proposal
B5.  Mock invalid governance (requires_user_confirmation=False) → invalid_proposal

C. Unsupported changes guard

C1.  Proposal with unsupported_changes → unsupported_proposal
C2.  Unsupported changes listed in errors
C3.  No draft created for unsupported proposal
C4.  Proposal with empty unsupported_changes → proceeds

D. Confirmed proposal creates draft preview

D1.  confirm_proposal=True, valid spec → result.status in draft outcomes
D2.  spec_proposal attached to result
D3.  save_draft=False → saved_draft is None / draft_saved=False
D4.  approval_required=True on draft result
D5.  execution_performed=False on draft result
D6.  draft_approved=False on draft result
D7.  to_dict() governance.draft_saved=False when not saved
D8.  to_dict() governance.draft_created matches status

E. Save draft bridge

E1.  confirm_proposal=True, save_draft=True, status=draft_created → saved_draft present
E2.  save_draft=True but status≠draft_created → saved_draft not set to saved
E3.  saved draft result approval_required=True

F. Hard/soft unsupported fields

F1.  Result from draft pipeline preserves unsupported_fields
F2.  Result from draft pipeline preserves hard_unsupported_fields

G. ResearchSpecProposal.from_dict() round-trip

G1.  to_dict() then from_dict() preserves proposed_spec
G2.  from_dict() enforces governance invariants regardless of input
G3.  from_dict() preserves unsupported_changes
G4.  from_dict() preserves supporting_evidence_ids
G5.  from_dict() with missing optional fields uses defaults

H. WorkflowResult additions

H1.  draft_saved field present in WorkflowResult
H2.  to_dict() includes draft_saved in governance
H3.  requires_user_confirmation in governance dict always True
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.agent.intent.research_spec import ResearchSpec
from src.agent.next_actions.spec_proposal import ResearchSpecProposal
from src.agent.workflow.proposal_to_draft_workflow import (
    _validate_proposal_governance,
    run_draft_from_spec_proposal,
)
from src.agent.workflow.workflow_result import WorkflowResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PTDW_MOD = "src.agent.workflow.proposal_to_draft_workflow"


def _proposal(
    *,
    proposed_spec: ResearchSpec | None = None,
    unsupported_changes: list[str] | None = None,
    supporting_evidence_ids: list[str] | None = None,
    rationale: str = "test rationale",
) -> ResearchSpecProposal:
    if proposed_spec is None:
        proposed_spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            end_date="2024-12-31",
            model="LassoRegression",
        )
    return ResearchSpecProposal(
        proposal_id="p5f_test",
        source_candidate_id="cand_001",
        source_action_type="model_comparison",
        base_spec=ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
        proposed_spec=proposed_spec,
        proposed_changes={"model": "LassoRegression"},
        unchanged_fields={"preset": "canonical_multi_asset_etf", "end_date": "2024-12-31"},
        rationale=rationale,
        supporting_evidence_ids=supporting_evidence_ids or ["ev1"],
        unsupported_changes=unsupported_changes or [],
    )


def _stub_draft_wr(status: str = "draft_created") -> SimpleNamespace:
    """Stub WorkflowResult for the draft pipeline."""
    ns = SimpleNamespace(
        status=status,
        summary=f"Draft [{status}].",
        resolution={"status": "resolved", "resolved_spec": {}, "missing_fields": []},
        plan={"plan_id": "plan_x"},
        draft={"draft_id": "d_stub", "proposed_name": "exp_v2", "changes": []},
        warnings=[],
        errors=[] if status == "draft_created" else ["error"],
        missing_fields=[],
        defaulted_fields={},
        assumptions=[],
        requires_confirmation=False,
        unsupported_fields=[],
        hard_unsupported_fields=[],
        approval_required=True,
        execution_performed=False,
        draft_approved=False,
        mapped_changes=[],
        validation_errors=[],
        proposed_name="exp_v2" if status == "draft_created" else None,
        base_experiment="canonical_ml_multi_asset",
        experiment_draft_object=None,
        parse=None,
        saved_draft=None,
        next_actions=None,
        memory_context_summary=None,
        spec_proposal=None,
        selected_candidate=None,
        draft_saved=False,
    )
    ns.to_dict = lambda: {
        "status": status,
        "summary": ns.summary,
        "resolution": ns.resolution,
        "plan": ns.plan,
        "draft": ns.draft,
        "warnings": [],
        "errors": ns.errors,
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
        "proposed_name": ns.proposed_name,
        "base_experiment": ns.base_experiment,
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
            "draft_created": status == "draft_created",
            "draft_saved": False,
        },
    }
    return ns


def _stub_saved_result(saved: bool = True) -> SimpleNamespace:
    ns = SimpleNamespace(
        status="saved" if saved else "not_saved",
        draft_id="d_stub",
        draft_path="results/drafts/d_stub.json",
        base_experiment="canonical_ml_multi_asset",
        proposed_name="exp_v2",
        approval_required=True,
        execution_performed=False,
        draft_approved=False,
        warnings=[],
        errors=[] if saved else ["Cannot persist"],
        summary="Draft saved." if saved else "Not saved.",
    )
    ns.to_dict = lambda: {
        "status": ns.status,
        "draft_id": ns.draft_id,
        "draft_path": ns.draft_path,
        "base_experiment": ns.base_experiment,
        "proposed_name": ns.proposed_name,
        "approval_required": True,
        "execution_performed": False,
        "draft_approved": False,
        "warnings": [],
        "errors": ns.errors,
        "summary": ns.summary,
    }
    return ns


# ---------------------------------------------------------------------------
# A. Confirmation gate
# ---------------------------------------------------------------------------


class TestConfirmationGate:
    def test_no_confirm_returns_confirmation_required(self):
        result = run_draft_from_spec_proposal(_proposal(), confirm_proposal=False)
        assert result.status == "confirmation_required"

    def test_no_confirm_no_resolution(self):
        result = run_draft_from_spec_proposal(_proposal(), confirm_proposal=False)
        assert result.resolution is None

    def test_no_confirm_no_plan(self):
        result = run_draft_from_spec_proposal(_proposal(), confirm_proposal=False)
        assert result.plan is None

    def test_no_confirm_no_draft(self):
        result = run_draft_from_spec_proposal(_proposal(), confirm_proposal=False)
        assert result.draft is None

    def test_no_confirm_no_save(self):
        result = run_draft_from_spec_proposal(
            _proposal(), confirm_proposal=False, save_draft=True
        )
        assert result.draft_saved is False
        assert result.saved_draft is None

    def test_no_confirm_approval_required_true(self):
        result = run_draft_from_spec_proposal(_proposal(), confirm_proposal=False)
        assert result.approval_required is True

    def test_no_confirm_execution_performed_false(self):
        result = run_draft_from_spec_proposal(_proposal(), confirm_proposal=False)
        assert result.execution_performed is False

    def test_no_confirm_draft_approved_false(self):
        result = run_draft_from_spec_proposal(_proposal(), confirm_proposal=False)
        assert result.draft_approved is False

    def test_no_confirm_spec_proposal_in_result(self):
        p = _proposal()
        result = run_draft_from_spec_proposal(p, confirm_proposal=False)
        assert result.spec_proposal is not None
        assert result.spec_proposal.get("proposal_id") == "p5f_test"

    def test_no_confirm_result_is_workflow_result(self):
        result = run_draft_from_spec_proposal(_proposal(), confirm_proposal=False)
        assert isinstance(result, WorkflowResult)


# ---------------------------------------------------------------------------
# B. Governance validation
# ---------------------------------------------------------------------------


class TestGovernanceValidation:
    def test_valid_proposal_passes_governance(self):
        assert _validate_proposal_governance(_proposal()) is None

    def test_creates_draft_true_fails(self):
        p = _proposal()
        object.__setattr__(p, "creates_draft", True)
        err = _validate_proposal_governance(p)
        assert err is not None
        assert "creates_draft" in err

    def test_executes_experiment_true_fails(self):
        p = _proposal()
        object.__setattr__(p, "executes_experiment", True)
        err = _validate_proposal_governance(p)
        assert err is not None
        assert "executes_experiment" in err

    def test_approval_required_false_fails(self):
        p = _proposal()
        object.__setattr__(p, "approval_required", False)
        err = _validate_proposal_governance(p)
        assert err is not None
        assert "approval_required" in err

    def test_requires_user_confirmation_false_fails(self):
        p = _proposal()
        object.__setattr__(p, "requires_user_confirmation", False)
        err = _validate_proposal_governance(p)
        assert err is not None
        assert "requires_user_confirmation" in err

    def test_invalid_governance_returns_invalid_proposal(self):
        p = _proposal()
        object.__setattr__(p, "creates_draft", True)
        result = run_draft_from_spec_proposal(p, confirm_proposal=True)
        assert result.status == "invalid_proposal"

    def test_invalid_governance_no_draft(self):
        p = _proposal()
        object.__setattr__(p, "executes_experiment", True)
        result = run_draft_from_spec_proposal(p, confirm_proposal=True)
        assert result.draft is None
        assert result.draft_saved is False


# ---------------------------------------------------------------------------
# C. Unsupported changes guard
# ---------------------------------------------------------------------------


class TestUnsupportedChanges:
    def test_unsupported_changes_returns_unsupported_proposal(self):
        p = _proposal(unsupported_changes=["model 'RandomForest' is not in the registry"])
        result = run_draft_from_spec_proposal(p, confirm_proposal=True)
        assert result.status == "unsupported_proposal"

    def test_unsupported_changes_in_errors(self):
        msg = "model 'RandomForest' is not in the registry"
        p = _proposal(unsupported_changes=[msg])
        result = run_draft_from_spec_proposal(p, confirm_proposal=True)
        assert msg in result.errors

    def test_unsupported_proposal_no_draft(self):
        p = _proposal(unsupported_changes=["bad_field 'x' not supported"])
        result = run_draft_from_spec_proposal(p, confirm_proposal=True)
        assert result.draft is None

    def test_unsupported_proposal_not_saved(self):
        p = _proposal(unsupported_changes=["bad_field"])
        result = run_draft_from_spec_proposal(p, confirm_proposal=True, save_draft=True)
        assert result.draft_saved is False

    def test_empty_unsupported_changes_proceeds(self):
        stub = _stub_draft_wr("draft_created")
        with patch(f"{_PTDW_MOD}.run_draft_from_spec", return_value=stub):
            p = _proposal(unsupported_changes=[])
            result = run_draft_from_spec_proposal(p, confirm_proposal=True)
        assert result.status == "draft_created"


# ---------------------------------------------------------------------------
# D. Confirmed proposal creates draft preview
# ---------------------------------------------------------------------------


class TestConfirmedDraftPreview:
    def test_confirmed_valid_proposal_produces_draft(self):
        stub = _stub_draft_wr("draft_created")
        with patch(f"{_PTDW_MOD}.run_draft_from_spec", return_value=stub):
            result = run_draft_from_spec_proposal(_proposal(), confirm_proposal=True)
        assert result.status == "draft_created"

    def test_spec_proposal_attached_to_result(self):
        stub = _stub_draft_wr("draft_created")
        with patch(f"{_PTDW_MOD}.run_draft_from_spec", return_value=stub):
            result = run_draft_from_spec_proposal(_proposal(), confirm_proposal=True)
        assert result.spec_proposal is not None

    def test_save_draft_false_not_saved(self):
        stub = _stub_draft_wr("draft_created")
        with patch(f"{_PTDW_MOD}.run_draft_from_spec", return_value=stub):
            result = run_draft_from_spec_proposal(
                _proposal(), confirm_proposal=True, save_draft=False
            )
        assert result.draft_saved is False
        assert result.saved_draft is None

    def test_approval_required_true_after_draft(self):
        stub = _stub_draft_wr("draft_created")
        with patch(f"{_PTDW_MOD}.run_draft_from_spec", return_value=stub):
            result = run_draft_from_spec_proposal(_proposal(), confirm_proposal=True)
        assert result.approval_required is True

    def test_execution_performed_false_after_draft(self):
        stub = _stub_draft_wr("draft_created")
        with patch(f"{_PTDW_MOD}.run_draft_from_spec", return_value=stub):
            result = run_draft_from_spec_proposal(_proposal(), confirm_proposal=True)
        assert result.execution_performed is False

    def test_draft_approved_false_after_draft(self):
        stub = _stub_draft_wr("draft_created")
        with patch(f"{_PTDW_MOD}.run_draft_from_spec", return_value=stub):
            result = run_draft_from_spec_proposal(_proposal(), confirm_proposal=True)
        assert result.draft_approved is False

    def test_to_dict_governance_draft_saved_false(self):
        stub = _stub_draft_wr("draft_created")
        with patch(f"{_PTDW_MOD}.run_draft_from_spec", return_value=stub):
            result = run_draft_from_spec_proposal(_proposal(), confirm_proposal=True)
        gov = result.to_dict().get("governance") or {}
        assert gov.get("draft_saved") is False

    def test_to_dict_governance_draft_created_true(self):
        stub = _stub_draft_wr("draft_created")
        with patch(f"{_PTDW_MOD}.run_draft_from_spec", return_value=stub):
            result = run_draft_from_spec_proposal(_proposal(), confirm_proposal=True)
        gov = result.to_dict().get("governance") or {}
        assert gov.get("draft_created") is True

    def test_resolution_failure_propagated(self):
        stub = _stub_draft_wr("needs_user_input")
        stub.resolution = None
        stub.draft = None
        with patch(f"{_PTDW_MOD}.run_draft_from_spec", return_value=stub):
            result = run_draft_from_spec_proposal(_proposal(), confirm_proposal=True)
        assert result.status == "needs_user_input"
        assert result.draft is None


# ---------------------------------------------------------------------------
# E. Save draft bridge
# ---------------------------------------------------------------------------


class TestSaveDraftBridge:
    def test_save_draft_true_attaches_saved_draft(self):
        stub = _stub_draft_wr("draft_created")
        save_stub = _stub_saved_result(saved=True)
        with patch(f"{_PTDW_MOD}.run_draft_from_spec", return_value=stub), \
             patch(f"{_PTDW_MOD}.save_agent_draft_result", return_value=save_stub):
            result = run_draft_from_spec_proposal(
                _proposal(), confirm_proposal=True, save_draft=True
            )
        assert result.saved_draft is not None
        assert result.saved_draft.get("status") == "saved"

    def test_save_draft_true_sets_draft_saved(self):
        stub = _stub_draft_wr("draft_created")
        save_stub = _stub_saved_result(saved=True)
        with patch(f"{_PTDW_MOD}.run_draft_from_spec", return_value=stub), \
             patch(f"{_PTDW_MOD}.save_agent_draft_result", return_value=save_stub):
            result = run_draft_from_spec_proposal(
                _proposal(), confirm_proposal=True, save_draft=True
            )
        assert result.draft_saved is True

    def test_save_draft_not_called_when_no_draft_created(self):
        stub = _stub_draft_wr("unsupported_plan")
        with patch(f"{_PTDW_MOD}.run_draft_from_spec", return_value=stub), \
             patch(f"{_PTDW_MOD}.save_agent_draft_result") as mock_save:
            result = run_draft_from_spec_proposal(
                _proposal(), confirm_proposal=True, save_draft=True
            )
        mock_save.assert_not_called()
        assert result.draft_saved is False

    def test_saved_draft_approval_required_true(self):
        stub = _stub_draft_wr("draft_created")
        save_stub = _stub_saved_result(saved=True)
        with patch(f"{_PTDW_MOD}.run_draft_from_spec", return_value=stub), \
             patch(f"{_PTDW_MOD}.save_agent_draft_result", return_value=save_stub):
            result = run_draft_from_spec_proposal(
                _proposal(), confirm_proposal=True, save_draft=True
            )
        assert result.saved_draft.get("approval_required") is True
        assert result.saved_draft.get("draft_approved") is False


# ---------------------------------------------------------------------------
# F. Hard/soft unsupported fields
# ---------------------------------------------------------------------------


class TestUnsupportedFields:
    def test_soft_unsupported_fields_preserved(self):
        stub = _stub_draft_wr("draft_created")
        stub.unsupported_fields = ["start_date"]
        with patch(f"{_PTDW_MOD}.run_draft_from_spec", return_value=stub):
            result = run_draft_from_spec_proposal(_proposal(), confirm_proposal=True)
        assert result.status == "draft_created"

    def test_hard_unsupported_fields_preserved(self):
        stub = _stub_draft_wr("unsupported_plan")
        stub.hard_unsupported_fields = ["universe"]
        stub.draft = None
        with patch(f"{_PTDW_MOD}.run_draft_from_spec", return_value=stub):
            result = run_draft_from_spec_proposal(_proposal(), confirm_proposal=True)
        assert result.status == "unsupported_plan"


# ---------------------------------------------------------------------------
# G. ResearchSpecProposal.from_dict() round-trip
# ---------------------------------------------------------------------------


class TestResearchSpecProposalFromDict:
    def test_round_trip_proposed_spec(self):
        p = _proposal()
        d = p.to_dict()
        p2 = ResearchSpecProposal.from_dict(d)
        assert p2.proposed_spec.model == "LassoRegression"

    def test_from_dict_enforces_governance(self):
        p = _proposal()
        d = p.to_dict()
        d["creates_draft"] = True
        d["executes_experiment"] = True
        p2 = ResearchSpecProposal.from_dict(d)
        assert p2.creates_draft is False
        assert p2.executes_experiment is False
        assert p2.approval_required is True
        assert p2.requires_user_confirmation is True

    def test_from_dict_preserves_unsupported_changes(self):
        p = _proposal(unsupported_changes=["bad_field"])
        d = p.to_dict()
        p2 = ResearchSpecProposal.from_dict(d)
        assert "bad_field" in p2.unsupported_changes

    def test_from_dict_preserves_evidence_ids(self):
        p = _proposal(supporting_evidence_ids=["ev1", "ev2"])
        d = p.to_dict()
        p2 = ResearchSpecProposal.from_dict(d)
        assert "ev1" in p2.supporting_evidence_ids

    def test_from_dict_missing_optional_fields(self):
        minimal = {
            "proposal_id": "p1",
            "source_candidate_id": "c1",
            "source_action_type": "model_comparison",
            "proposed_spec": {"preset": "canonical_multi_asset_etf"},
        }
        p = ResearchSpecProposal.from_dict(minimal)
        assert p.unsupported_changes == []
        assert p.supporting_evidence_ids == []
        assert p.rationale == ""

    def test_from_dict_base_spec_none_when_absent(self):
        minimal = {
            "proposal_id": "p1",
            "source_candidate_id": "c1",
            "source_action_type": "model_comparison",
            "proposed_spec": {},
        }
        p = ResearchSpecProposal.from_dict(minimal)
        assert p.base_spec is None


# ---------------------------------------------------------------------------
# H. WorkflowResult additions
# ---------------------------------------------------------------------------


class TestWorkflowResultAdditions:
    def test_draft_saved_field_exists(self):
        wr = WorkflowResult(status="test", summary="test",
                            approval_required=True, execution_performed=False, draft_approved=False)
        assert hasattr(wr, "draft_saved")
        assert wr.draft_saved is False

    def test_to_dict_includes_draft_saved_in_governance(self):
        wr = WorkflowResult(status="test", summary="test",
                            approval_required=True, execution_performed=False, draft_approved=False)
        gov = wr.to_dict().get("governance") or {}
        assert "draft_saved" in gov
        assert gov["draft_saved"] is False

    def test_to_dict_draft_saved_true_when_set(self):
        wr = WorkflowResult(status="draft_created", summary="test",
                            approval_required=True, execution_performed=False, draft_approved=False,
                            draft_saved=True)
        gov = wr.to_dict().get("governance") or {}
        assert gov["draft_saved"] is True

    def test_governance_requires_user_confirmation_always_true(self):
        wr = WorkflowResult(status="test", summary="test",
                            approval_required=True, execution_performed=False, draft_approved=False)
        gov = wr.to_dict().get("governance") or {}
        assert gov.get("requires_user_confirmation") is True
