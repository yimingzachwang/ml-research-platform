"""Tests for ResearchSpecProposal and ResearchSpecProposalResult — Phase 5D.

A. Spec proposal dataclass
   A1.  Serialisation — to_dict() returns plain dict
   A2.  Governance defaults — requires_user_confirmation=True always
   A3.  Governance defaults — creates_draft=False always
   A4.  Governance defaults — executes_experiment=False always
   A5.  Governance defaults — approval_required=True always
   A6.  Governance cannot be overridden via constructor
   A7.  Stores source_candidate_id correctly
   A8.  Stores proposed_changes correctly
   A9.  Stores supporting_evidence_ids correctly
   A10. Stores source_action_type correctly
   A11. to_dict() includes base_spec dict when present
   A12. to_dict() base_spec is None when not set
   A13. to_dict() proposed_spec is always present
   A14. proposed_changes, unchanged_fields, unsupported_changes in to_dict()
   A15. resolution_status / resolution_summary in to_dict()
   A16. Stores warnings and unsupported_changes separately

B. ResearchSpecProposalResult
   B1.  Serialisation — to_dict() returns plain dict
   B2.  Status field present and valid
   B3.  proposal field present (None for failure statuses)
   B4.  resolution_result serialised compactly
   B5.  warnings and errors are lists
   B6.  summary is a string

C. summarise_spec_proposal
   C1.  Returns a string
   C2.  Includes action type
   C3.  Includes proposed changes
   C4.  Includes governance statement (no draft / no execution)
   C5.  Mentions rationale
   C6.  Mentions supporting evidence when present
   C7.  Mentions unsupported changes when present
   C8.  Mentions resolution status when present
   C9.  Mentions errors when present
   C10. Works for proposal=None (failure results)
"""

from __future__ import annotations

import pytest

from src.agent.intent.research_spec import ResearchSpec
from src.agent.next_actions.spec_proposal import (
    ResearchSpecProposal,
    ResearchSpecProposalResult,
    summarise_spec_proposal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_base_spec(**kwargs) -> ResearchSpec:
    defaults = {
        "preset": "canonical_multi_asset_etf",
        "end_date": "2024-12-31",
        "model": "RidgeRegression",
    }
    return ResearchSpec(**{**defaults, **kwargs})


def _make_proposed_spec(**kwargs) -> ResearchSpec:
    defaults = {
        "preset": "canonical_multi_asset_etf",
        "end_date": "2024-12-31",
        "model": "LassoRegression",
    }
    return ResearchSpec(**{**defaults, **kwargs})


def _make_proposal(
    *,
    source_candidate_id: str = "cand_001",
    source_action_type: str = "model_comparison",
    base_spec: ResearchSpec | None = None,
    proposed_changes: dict | None = None,
    supporting_evidence_ids: list[str] | None = None,
    unsupported_changes: list[str] | None = None,
    resolution_status: str | None = "resolved",
    resolution_summary: str | None = "resolved",
    requires_user_confirmation: bool = True,
    creates_draft: bool = False,
    executes_experiment: bool = False,
    approval_required: bool = True,
    rationale: str = "Model comparison to test regularisation",
) -> ResearchSpecProposal:
    return ResearchSpecProposal(
        proposal_id="p5d_testid",
        source_candidate_id=source_candidate_id,
        source_action_type=source_action_type,
        base_spec=base_spec or _make_base_spec(),
        proposed_spec=_make_proposed_spec(),
        proposed_changes=proposed_changes if proposed_changes is not None else {"model": "LassoRegression"},
        unchanged_fields={"preset": "canonical_multi_asset_etf", "end_date": "2024-12-31"},
        rationale=rationale,
        supporting_evidence_ids=supporting_evidence_ids or ["mem_001"],
        warnings=[],
        unsupported_changes=unsupported_changes or [],
        resolution_status=resolution_status,
        resolution_summary=resolution_summary,
        requires_user_confirmation=requires_user_confirmation,
        creates_draft=creates_draft,
        executes_experiment=executes_experiment,
        approval_required=approval_required,
    )


def _make_result(
    status: str = "proposal_created",
    proposal: ResearchSpecProposal | None = None,
) -> ResearchSpecProposalResult:
    return ResearchSpecProposalResult(
        status=status,  # type: ignore[arg-type]
        proposal=proposal or _make_proposal(),
        resolution_result=None,
        warnings=["warn_a"],
        errors=[],
        summary="stub summary",
    )


# ---------------------------------------------------------------------------
# A. ResearchSpecProposal dataclass
# ---------------------------------------------------------------------------


class TestResearchSpecProposal:
    def test_to_dict_returns_plain_dict(self):
        p = _make_proposal()
        d = p.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_no_nested_objects(self):
        p = _make_proposal()
        d = p.to_dict()
        # Every value must be a plain Python type (no ResearchSpec objects)
        for key, val in d.items():
            assert not hasattr(val, "to_dict"), (
                f"to_dict() still contains an object at key '{key}'"
            )

    def test_governance_requires_user_confirmation_true(self):
        p = _make_proposal()
        assert p.requires_user_confirmation is True

    def test_governance_creates_draft_false(self):
        p = _make_proposal()
        assert p.creates_draft is False

    def test_governance_executes_experiment_false(self):
        p = _make_proposal()
        assert p.executes_experiment is False

    def test_governance_approval_required_true(self):
        p = _make_proposal()
        assert p.approval_required is True

    def test_governance_creates_draft_cannot_be_overridden(self):
        p = _make_proposal(creates_draft=True)
        assert p.creates_draft is False

    def test_governance_executes_experiment_cannot_be_overridden(self):
        p = _make_proposal(executes_experiment=True)
        assert p.executes_experiment is False

    def test_governance_requires_user_confirmation_cannot_be_overridden(self):
        p = _make_proposal(requires_user_confirmation=False)
        assert p.requires_user_confirmation is True

    def test_governance_approval_required_cannot_be_overridden(self):
        p = _make_proposal(approval_required=False)
        assert p.approval_required is True

    def test_stores_source_candidate_id(self):
        p = _make_proposal(source_candidate_id="my_candidate_xyz")
        assert p.source_candidate_id == "my_candidate_xyz"

    def test_stores_source_action_type(self):
        p = _make_proposal(source_action_type="feature_ablation")
        assert p.source_action_type == "feature_ablation"

    def test_stores_proposed_changes(self):
        changes = {"model": "LassoRegression", "model_params": {"alpha": 2.0}}
        p = _make_proposal(proposed_changes=changes)
        assert p.proposed_changes == changes

    def test_stores_supporting_evidence_ids(self):
        ids = ["mem_001", "mem_002"]
        p = _make_proposal(supporting_evidence_ids=ids)
        assert p.supporting_evidence_ids == ids

    def test_to_dict_includes_base_spec(self):
        base = _make_base_spec()
        p = _make_proposal(base_spec=base)
        d = p.to_dict()
        assert d["base_spec"] is not None
        assert isinstance(d["base_spec"], dict)

    def test_to_dict_base_spec_none_when_not_set(self):
        p = ResearchSpecProposal(
            proposal_id="p5d_x",
            source_candidate_id="c1",
            source_action_type="repeat_baseline",
            base_spec=None,
            proposed_spec=_make_proposed_spec(),
            proposed_changes={},
            unchanged_fields={},
            rationale="repeat",
        )
        d = p.to_dict()
        assert d["base_spec"] is None

    def test_to_dict_proposed_spec_always_present(self):
        p = _make_proposal()
        d = p.to_dict()
        assert d["proposed_spec"] is not None
        assert isinstance(d["proposed_spec"], dict)

    def test_to_dict_includes_provenance_diffs(self):
        p = _make_proposal(
            proposed_changes={"model": "LassoRegression"},
            unsupported_changes=["some_field"],
        )
        d = p.to_dict()
        assert "proposed_changes" in d
        assert "unchanged_fields" in d
        assert "unsupported_changes" in d

    def test_to_dict_includes_resolution_fields(self):
        p = _make_proposal(resolution_status="resolved", resolution_summary="resolved")
        d = p.to_dict()
        assert d["resolution_status"] == "resolved"
        assert d["resolution_summary"] == "resolved"

    def test_to_dict_includes_governance_keys(self):
        p = _make_proposal()
        d = p.to_dict()
        assert d["requires_user_confirmation"] is True
        assert d["creates_draft"] is False
        assert d["executes_experiment"] is False
        assert d["approval_required"] is True

    def test_warnings_and_unsupported_stored_separately(self):
        p = ResearchSpecProposal(
            proposal_id="p5d_y",
            source_candidate_id="c1",
            source_action_type="model_comparison",
            base_spec=_make_base_spec(),
            proposed_spec=_make_proposed_spec(),
            proposed_changes={"model": "LassoRegression"},
            unchanged_fields={},
            rationale="test",
            warnings=["advisory warning"],
            unsupported_changes=["field_x"],
        )
        assert p.warnings == ["advisory warning"]
        assert p.unsupported_changes == ["field_x"]

    def test_to_dict_rationale_present(self):
        p = _make_proposal(rationale="testing regularisation difference")
        d = p.to_dict()
        assert d["rationale"] == "testing regularisation difference"

    def test_proposal_id_in_to_dict(self):
        p = _make_proposal()
        d = p.to_dict()
        assert d["proposal_id"] == "p5d_testid"


# ---------------------------------------------------------------------------
# B. ResearchSpecProposalResult
# ---------------------------------------------------------------------------


class TestResearchSpecProposalResult:
    def test_to_dict_returns_plain_dict(self):
        r = _make_result()
        d = r.to_dict()
        assert isinstance(d, dict)

    def test_status_present(self):
        r = _make_result("proposal_created")
        assert r.status == "proposal_created"

    def test_status_in_to_dict(self):
        r = _make_result("needs_user_input")
        d = r.to_dict()
        assert d["status"] == "needs_user_input"

    def test_proposal_present_in_to_dict(self):
        r = _make_result("proposal_created")
        d = r.to_dict()
        assert d["proposal"] is not None
        assert isinstance(d["proposal"], dict)

    def test_proposal_none_for_unsupported(self):
        r = ResearchSpecProposalResult(
            status="unsupported_action",
            proposal=None,
            summary="unsupported",
        )
        d = r.to_dict()
        assert d["proposal"] is None

    def test_resolution_result_none_when_absent(self):
        r = _make_result()
        d = r.to_dict()
        assert d["resolution_result"] is None

    def test_warnings_and_errors_are_lists(self):
        r = ResearchSpecProposalResult(
            status="proposal_created",
            proposal=_make_proposal(),
            warnings=["w1", "w2"],
            errors=["e1"],
            summary="x",
        )
        assert isinstance(r.warnings, list)
        assert isinstance(r.errors, list)

    def test_summary_is_string(self):
        r = _make_result()
        assert isinstance(r.summary, str)

    def test_all_required_keys_in_to_dict(self):
        r = _make_result()
        d = r.to_dict()
        for key in ("status", "proposal", "resolution_result", "warnings", "errors", "summary"):
            assert key in d


# ---------------------------------------------------------------------------
# C. summarise_spec_proposal
# ---------------------------------------------------------------------------


class TestSummariseSpecProposal:
    def test_returns_string(self):
        r = _make_result()
        s = summarise_spec_proposal(r)
        assert isinstance(s, str)

    def test_includes_status(self):
        r = _make_result("proposal_created")
        s = summarise_spec_proposal(r)
        assert "proposal_created" in s

    def test_includes_action_type(self):
        r = _make_result()
        s = summarise_spec_proposal(r)
        assert "model_comparison" in s

    def test_includes_proposed_changes(self):
        r = _make_result()
        s = summarise_spec_proposal(r)
        # The proposal has model=LassoRegression
        assert "LassoRegression" in s or "model" in s

    def test_includes_governance_no_draft(self):
        r = _make_result()
        s = summarise_spec_proposal(r)
        assert "No draft created" in s

    def test_includes_governance_no_execution(self):
        r = _make_result()
        s = summarise_spec_proposal(r)
        assert "No execution performed" in s

    def test_includes_user_confirmation_required(self):
        r = _make_result()
        s = summarise_spec_proposal(r)
        assert "confirmation required" in s.lower() or "user confirmation" in s.lower()

    def test_mentions_rationale(self):
        r = _make_result()
        s = summarise_spec_proposal(r)
        assert "Rationale" in s or "regularisation" in s

    def test_mentions_evidence_ids(self):
        r = _make_result()
        s = summarise_spec_proposal(r)
        # supporting_evidence_ids=["mem_001"]
        assert "mem_001" in s

    def test_mentions_unsupported_when_present(self):
        proposal = _make_proposal(unsupported_changes=["some_field not supported"])
        r = _make_result(proposal=proposal)
        s = summarise_spec_proposal(r)
        assert "Unsupported" in s or "unsupported" in s

    def test_mentions_resolution_status(self):
        proposal = _make_proposal(resolution_status="needs_user_input")
        r = _make_result("needs_user_input", proposal=proposal)
        s = summarise_spec_proposal(r)
        assert "needs_user_input" in s or "Resolution" in s

    def test_mentions_errors_when_present(self):
        r = ResearchSpecProposalResult(
            status="resolution_failed",
            proposal=None,
            errors=["panel signal on single-asset universe"],
            summary="",
        )
        s = summarise_spec_proposal(r)
        assert "Errors" in s or "panel signal" in s

    def test_works_for_proposal_none(self):
        r = ResearchSpecProposalResult(
            status="unsupported_action",
            proposal=None,
            errors=[],
            warnings=[],
            summary="",
        )
        s = summarise_spec_proposal(r)
        assert isinstance(s, str)
        assert "No draft created" in s

    def test_repeat_baseline_mentions_no_changes(self):
        proposal = _make_proposal(
            source_action_type="repeat_baseline",
            proposed_changes={},
        )
        r = _make_result(proposal=proposal)
        s = summarise_spec_proposal(r)
        assert "none" in s.lower() or "repeat baseline" in s.lower() or "Proposed changes: none" in s
