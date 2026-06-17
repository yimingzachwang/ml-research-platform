"""Next-action workflow façade tests — Phase 5C.

Covers:
  A. Structured spec → recommendations (canonical preset with dates)
  B. Natural-language request → recommendations
  C. Incomplete spec → needs_user_input (no next-action recommendations)
  D. Empty memory → needs_more_evidence
  E. Truncation metadata when generated > max_candidates
  F. Execution-keyword warning in NL request
  G. Governance invariants on all result paths
  H. WorkflowResult structure (next_actions, memory_context, governance)
  I. No spec / objective-only path
  J. Resolution failure paths (invalid spec)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.memory.memory_context import ResearchMemoryContext
from src.agent.memory.memory_query import ResearchMemoryQuery
from src.agent.workflow.next_action_workflow import (
    run_recommend_next_actions,
    run_recommend_next_actions_from_request,
)
from src.agent.workflow.workflow_result import WorkflowResult

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

_MOD_RETRIEVE = "src.agent.memory.memory_connector.retrieve_research_memory"

_CANONICAL_SPEC_DICT = {
    "preset": "canonical_multi_asset_etf",
    "start_date": "2015-01-01",
    "end_date": "2024-12-31",
}


def _empty_ctx() -> ResearchMemoryContext:
    return ResearchMemoryContext(
        query=ResearchMemoryQuery(keyword_query=None, semantic_query=None, source="test"),
    )


def _ctx_with_failure(mode: str = "poor_oos_consistency") -> ResearchMemoryContext:
    from src.agent.memory.memory_evidence import ResearchMemoryEvidence

    ev = ResearchMemoryEvidence(
        memory_id="mem_001",
        experiment_name="exp_a",
        artefact_type="review",
        short_summary=f"Signal showed {mode} across splits.",
        failure_modes=[mode],
        tags=["ridge"],
        path="/artefacts/exp_a/review.json",
        context_hash="abc123",
        matched_terms=[mode],
    )
    ctx = ResearchMemoryContext(
        query=ResearchMemoryQuery(keyword_query=mode, semantic_query=None, source="test"),
        keyword_count=1,
    )
    ctx.evidence_items.append(ev)
    return ctx


def _govflags_ok(wr: WorkflowResult) -> bool:
    """Check all governance flags are correct for Phase 5C tools."""
    return (
        wr.execution_performed is False
        and wr.draft_approved is False
        and wr.approval_required is True
    )


# ---------------------------------------------------------------------------
# A. Structured spec → recommendations
# ---------------------------------------------------------------------------


class TestStructuredSpecRecommendations:
    @patch(_MOD_RETRIEVE, return_value=[])
    def test_returns_workflow_result(self, _m):
        from src.agent.intent.research_spec import ResearchSpec

        spec = ResearchSpec.from_dict(_CANONICAL_SPEC_DICT)
        result = run_recommend_next_actions(spec=spec)
        assert isinstance(result, WorkflowResult)

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_status_is_valid(self, _m):
        from src.agent.intent.research_spec import ResearchSpec

        spec = ResearchSpec.from_dict(_CANONICAL_SPEC_DICT)
        result = run_recommend_next_actions(spec=spec)
        assert result.status in (
            "recommended", "needs_more_evidence", "planned", "needs_review"
        )

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_next_actions_key_present(self, _m):
        from src.agent.intent.research_spec import ResearchSpec

        spec = ResearchSpec.from_dict(_CANONICAL_SPEC_DICT)
        result = run_recommend_next_actions(spec=spec)
        assert result.next_actions is not None

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_next_actions_has_candidates_key(self, _m):
        from src.agent.intent.research_spec import ResearchSpec

        spec = ResearchSpec.from_dict(_CANONICAL_SPEC_DICT)
        result = run_recommend_next_actions(spec=spec)
        assert "candidates" in result.next_actions

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_resolution_populated(self, _m):
        from src.agent.intent.research_spec import ResearchSpec

        spec = ResearchSpec.from_dict(_CANONICAL_SPEC_DICT)
        result = run_recommend_next_actions(spec=spec)
        assert result.resolution is not None

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_governance_flags(self, _m):
        from src.agent.intent.research_spec import ResearchSpec

        spec = ResearchSpec.from_dict(_CANONICAL_SPEC_DICT)
        result = run_recommend_next_actions(spec=spec)
        assert _govflags_ok(result)

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_no_draft_created(self, _m):
        from src.agent.intent.research_spec import ResearchSpec

        spec = ResearchSpec.from_dict(_CANONICAL_SPEC_DICT)
        result = run_recommend_next_actions(spec=spec)
        assert result.status != "draft_created"

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_to_dict_includes_governance(self, _m):
        from src.agent.intent.research_spec import ResearchSpec

        spec = ResearchSpec.from_dict(_CANONICAL_SPEC_DICT)
        d = run_recommend_next_actions(spec=spec).to_dict()
        gov = d["governance"]
        assert gov["execution_performed"] is False
        assert gov["draft_approved"] is False
        assert gov["draft_created"] is False
        assert gov["approval_required"] is True

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_to_dict_includes_next_actions(self, _m):
        from src.agent.intent.research_spec import ResearchSpec

        spec = ResearchSpec.from_dict(_CANONICAL_SPEC_DICT)
        d = run_recommend_next_actions(spec=spec).to_dict()
        assert "next_actions" in d
        assert d["next_actions"] is not None


# ---------------------------------------------------------------------------
# B. Natural-language request → recommendations
# ---------------------------------------------------------------------------


class TestNLRequestRecommendations:
    @patch(_MOD_RETRIEVE, return_value=[])
    def test_nl_returns_workflow_result(self, _m):
        result = run_recommend_next_actions_from_request(
            "Use canonical multi-asset ETF preset from 2015 to 2024. What should we try next?",
            provider="stub",
        )
        assert isinstance(result, WorkflowResult)

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_nl_status_valid(self, _m):
        result = run_recommend_next_actions_from_request(
            "Use canonical multi-asset ETF preset from 2015 to 2024.",
            provider="stub",
        )
        assert result.status in (
            "recommended", "needs_more_evidence",
            "needs_user_input", "needs_clarification",
        )

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_nl_parse_metadata_present(self, _m):
        result = run_recommend_next_actions_from_request(
            "Use canonical multi-asset ETF preset from 2015 to 2024.",
            provider="stub",
        )
        # parse is populated regardless of outcome
        assert result.parse is not None
        assert "status" in result.parse

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_nl_governance_flags(self, _m):
        result = run_recommend_next_actions_from_request(
            "Use canonical multi-asset ETF preset from 2015 to 2024.",
            provider="stub",
        )
        assert _govflags_ok(result)

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_nl_no_draft_created(self, _m):
        result = run_recommend_next_actions_from_request(
            "Use canonical multi-asset ETF preset from 2015 to 2024.",
            provider="stub",
        )
        assert result.status != "draft_created"

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_nl_to_dict_governance_draft_created_false(self, _m):
        d = run_recommend_next_actions_from_request(
            "Use canonical multi-asset ETF preset from 2015 to 2024.",
            provider="stub",
        ).to_dict()
        assert d["governance"]["draft_created"] is False


# ---------------------------------------------------------------------------
# C. Incomplete spec → needs_user_input
# ---------------------------------------------------------------------------


class TestIncompleteSpec:
    @patch(_MOD_RETRIEVE, return_value=[])
    def test_incomplete_spec_status(self, _m):
        from src.agent.intent.research_spec import ResearchSpec

        # custom_ticker_template needs end_date; no default exists
        spec = ResearchSpec.from_dict({"preset": "custom_ticker_template"})
        result = run_recommend_next_actions(spec=spec)
        assert result.status == "needs_user_input"

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_incomplete_spec_has_missing_fields(self, _m):
        from src.agent.intent.research_spec import ResearchSpec

        spec = ResearchSpec.from_dict({"preset": "custom_ticker_template"})
        result = run_recommend_next_actions(spec=spec)
        assert len(result.missing_fields) > 0

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_incomplete_spec_no_next_actions(self, _m):
        from src.agent.intent.research_spec import ResearchSpec

        spec = ResearchSpec.from_dict({"preset": "custom_ticker_template"})
        result = run_recommend_next_actions(spec=spec)
        assert result.next_actions is None

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_incomplete_spec_governance_still_correct(self, _m):
        from src.agent.intent.research_spec import ResearchSpec

        spec = ResearchSpec.from_dict({"preset": "custom_ticker_template"})
        result = run_recommend_next_actions(spec=spec)
        assert result.execution_performed is False
        assert result.draft_approved is False


# ---------------------------------------------------------------------------
# D. Empty memory → needs_more_evidence
# ---------------------------------------------------------------------------


class TestEmptyMemory:
    def test_empty_memory_status(self):
        result = run_recommend_next_actions(
            memory_context=_empty_ctx(),
        )
        assert result.status == "needs_more_evidence"

    def test_empty_memory_has_candidates(self):
        result = run_recommend_next_actions(
            memory_context=_empty_ctx(),
        )
        na = result.next_actions or {}
        assert len(na.get("candidates", [])) > 0

    def test_empty_memory_no_crash(self):
        result = run_recommend_next_actions(
            memory_context=_empty_ctx(),
        )
        assert result is not None

    def test_empty_memory_governance_ok(self):
        result = run_recommend_next_actions(
            memory_context=_empty_ctx(),
        )
        assert _govflags_ok(result)

    def test_empty_memory_summary_non_empty(self):
        result = run_recommend_next_actions(
            memory_context=_empty_ctx(),
        )
        assert result.summary


# ---------------------------------------------------------------------------
# E. Truncation metadata
# ---------------------------------------------------------------------------


class TestTruncation:
    def _ctx_overfitting_and_signal(self) -> ResearchMemoryContext:
        """Creates context with both signal-quality and overfitting failure modes —
        enough to trigger Rules A and D, producing 4 distinct action_type candidates."""
        from src.agent.memory.memory_evidence import ResearchMemoryEvidence

        ev = ResearchMemoryEvidence(
            memory_id="mem_t1",
            experiment_name="exp_t",
            artefact_type="review",
            short_summary="high split sharpe variance and coefficient instability observed.",
            failure_modes=["high_split_sharpe_variance", "coefficient_instability"],
            tags=[],
            path="/p",
            context_hash="cc",
            matched_terms=["high_split_sharpe_variance"],
        )
        ctx = ResearchMemoryContext(
            query=ResearchMemoryQuery(
                keyword_query="instability", semantic_query=None, source="test"
            ),
            keyword_count=1,
        )
        ctx.evidence_items.append(ev)
        return ctx

    def test_truncation_honours_max_candidates(self):
        ctx = self._ctx_overfitting_and_signal()
        result = run_recommend_next_actions(memory_context=ctx, max_candidates=2)
        na = result.next_actions or {}
        assert len(na.get("candidates", [])) <= 2

    def test_truncation_warning_present_when_truncated(self):
        ctx = self._ctx_overfitting_and_signal()
        result = run_recommend_next_actions(memory_context=ctx, max_candidates=1)
        # Should have at least one truncation warning
        all_warnings = result.warnings + (result.next_actions or {}).get("warnings", [])
        assert any("max_candidates" in w for w in all_warnings)

    def test_no_truncation_warning_when_under_limit(self):
        ctx = _ctx_with_failure("high_turnover")  # only 1 candidate from Rule C
        result = run_recommend_next_actions(memory_context=ctx, max_candidates=5)
        all_warnings = result.warnings + (result.next_actions or {}).get("warnings", [])
        assert not any("max_candidates" in w for w in all_warnings)

    def test_top_candidate_consistent_with_candidates(self):
        ctx = self._ctx_overfitting_and_signal()
        result = run_recommend_next_actions(memory_context=ctx, max_candidates=2)
        na = result.next_actions or {}
        cands = na.get("candidates", [])
        top = na.get("top_candidate")
        if cands and top:
            assert top["action_id"] == cands[0]["action_id"]


# ---------------------------------------------------------------------------
# F. Execution-keyword warning in NL request
# ---------------------------------------------------------------------------


class TestExecutionKeywordWarning:
    @patch(_MOD_RETRIEVE, return_value=[])
    def test_exec_keyword_warning_added(self, _m):
        result = run_recommend_next_actions_from_request(
            "Run the next experiment now.",
            provider="stub",
        )
        # Warning about execution keywords must be present
        assert any(
            "execution" in w.lower() or "run" in w.lower()
            for w in result.warnings
        )

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_exec_keyword_no_execution(self, _m):
        result = run_recommend_next_actions_from_request(
            "Run the next experiment now.",
            provider="stub",
        )
        assert result.execution_performed is False

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_exec_keyword_no_draft_approved(self, _m):
        result = run_recommend_next_actions_from_request(
            "Run the next experiment now.",
            provider="stub",
        )
        assert result.draft_approved is False

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_exec_keyword_status_not_draft_created(self, _m):
        result = run_recommend_next_actions_from_request(
            "Run the next experiment now.",
            provider="stub",
        )
        assert result.status != "draft_created"


# ---------------------------------------------------------------------------
# G. Governance invariants — exhaustive check across all paths
# ---------------------------------------------------------------------------


class TestGovernanceInvariants:
    @pytest.mark.parametrize("ctx_factory,kwargs", [
        (lambda: None, {"objective": "improve signal quality"}),
        (lambda: _empty_ctx(), {}),
        (lambda: _ctx_with_failure("poor_oos_consistency"), {}),
        (lambda: _ctx_with_failure("high_turnover"), {}),
    ])
    def test_execution_performed_always_false(self, ctx_factory, kwargs):
        ctx = ctx_factory()
        result = run_recommend_next_actions(memory_context=ctx, **kwargs)
        assert result.execution_performed is False

    @pytest.mark.parametrize("ctx_factory,kwargs", [
        (lambda: None, {"objective": "improve signal quality"}),
        (lambda: _empty_ctx(), {}),
        (lambda: _ctx_with_failure("poor_oos_consistency"), {}),
    ])
    def test_draft_approved_always_false(self, ctx_factory, kwargs):
        ctx = ctx_factory()
        result = run_recommend_next_actions(memory_context=ctx, **kwargs)
        assert result.draft_approved is False

    @pytest.mark.parametrize("ctx_factory,kwargs", [
        (lambda: None, {"objective": "improve signal quality"}),
        (lambda: _empty_ctx(), {}),
        (lambda: _ctx_with_failure("poor_oos_consistency"), {}),
    ])
    def test_approval_required_always_true(self, ctx_factory, kwargs):
        ctx = ctx_factory()
        result = run_recommend_next_actions(memory_context=ctx, **kwargs)
        assert result.approval_required is True

    @pytest.mark.parametrize("ctx_factory,kwargs", [
        (lambda: None, {"objective": "improve signal quality"}),
        (lambda: _empty_ctx(), {}),
        (lambda: _ctx_with_failure("poor_oos_consistency"), {}),
    ])
    def test_to_dict_governance_draft_created_false(self, ctx_factory, kwargs):
        ctx = ctx_factory()
        d = run_recommend_next_actions(memory_context=ctx, **kwargs).to_dict()
        assert d["governance"]["draft_created"] is False

    def test_next_action_candidates_governance_flags(self):
        ctx = _ctx_with_failure("poor_oos_consistency")
        result = run_recommend_next_actions(memory_context=ctx)
        na = result.next_actions or {}
        for c in na.get("candidates", []):
            assert c.get("creates_draft") is False
            assert c.get("executes_experiment") is False
            assert c.get("requires_user_confirmation") is True


# ---------------------------------------------------------------------------
# H. WorkflowResult structure
# ---------------------------------------------------------------------------


class TestWorkflowResultStructure:
    def test_next_actions_status_field(self):
        ctx = _ctx_with_failure("poor_oos_consistency")
        result = run_recommend_next_actions(memory_context=ctx)
        na = result.next_actions or {}
        assert "status" in na
        assert na["status"] in ("recommended", "needs_more_evidence", "invalid_input")

    def test_next_actions_evidence_summary(self):
        ctx = _ctx_with_failure("poor_oos_consistency")
        result = run_recommend_next_actions(memory_context=ctx)
        na = result.next_actions or {}
        assert "evidence_summary" in na
        assert isinstance(na["evidence_summary"], str)

    def test_next_actions_rejected_directions(self):
        ctx = _ctx_with_failure("poor_oos_consistency")
        result = run_recommend_next_actions(memory_context=ctx)
        na = result.next_actions or {}
        assert "rejected_directions" in na
        assert isinstance(na["rejected_directions"], list)

    def test_next_actions_recommended_cautions(self):
        ctx = _ctx_with_failure("poor_oos_consistency")
        result = run_recommend_next_actions(memory_context=ctx)
        na = result.next_actions or {}
        assert "recommended_cautions" in na
        assert isinstance(na["recommended_cautions"], list)

    def test_memory_context_key_when_evidence_present(self):
        ctx = _ctx_with_failure("poor_oos_consistency")
        result = run_recommend_next_actions(memory_context=ctx)
        # memory_context_summary is exposed as "memory_context" in to_dict()
        d = result.to_dict()
        assert "memory_context" in d

    def test_memory_context_structure_when_present(self):
        ctx = _ctx_with_failure("poor_oos_consistency")
        result = run_recommend_next_actions(memory_context=ctx)
        mc = result.memory_context_summary
        if mc is not None:
            assert "total_items" in mc
            assert "failure_mode_summary" in mc
            assert "is_empty" in mc

    def test_to_dict_includes_all_required_keys(self):
        ctx = _ctx_with_failure("poor_oos_consistency")
        d = run_recommend_next_actions(memory_context=ctx).to_dict()
        for key in ("status", "summary", "next_actions", "governance"):
            assert key in d

    def test_next_actions_approval_governance(self):
        ctx = _ctx_with_failure("poor_oos_consistency")
        result = run_recommend_next_actions(memory_context=ctx)
        na = result.next_actions or {}
        assert na.get("approval_required") is True
        assert na.get("execution_performed") is False
        assert na.get("draft_created") is False


# ---------------------------------------------------------------------------
# I. No spec / objective-only path
# ---------------------------------------------------------------------------


class TestObjectiveOnly:
    @patch(_MOD_RETRIEVE, return_value=[])
    def test_objective_only_returns_result(self, _m):
        result = run_recommend_next_actions(objective="Improve momentum signal quality")
        assert isinstance(result, WorkflowResult)

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_objective_only_no_resolution(self, _m):
        result = run_recommend_next_actions(objective="Improve momentum signal quality")
        # No spec provided → no resolution dict
        assert result.resolution is None

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_objective_only_governance(self, _m):
        result = run_recommend_next_actions(objective="Improve momentum signal quality")
        assert _govflags_ok(result)

    def test_no_inputs_is_needs_more_evidence(self):
        result = run_recommend_next_actions()
        assert result.status == "needs_more_evidence"


# ---------------------------------------------------------------------------
# J. Resolution failure paths
# ---------------------------------------------------------------------------


class TestResolutionFailure:
    @patch(_MOD_RETRIEVE, return_value=[])
    def test_invalid_spec_returns_no_next_actions(self, _m):
        from src.agent.intent.research_spec import ResearchSpec

        # Single-asset with panel signal — compatibility error
        spec = ResearchSpec.from_dict({
            "preset": "single_asset_ml",
            "end_date": "2024-12-31",
            "signal": "cross_sectional_momentum",  # panel signal on single-asset
        })
        result = run_recommend_next_actions(spec=spec)
        assert result.status in ("needs_user_input", "invalid", "error")
        assert result.next_actions is None

    @patch(_MOD_RETRIEVE, return_value=[])
    def test_invalid_spec_governance_still_safe(self, _m):
        from src.agent.intent.research_spec import ResearchSpec

        spec = ResearchSpec.from_dict({
            "preset": "single_asset_ml",
            "end_date": "2024-12-31",
            "signal": "cross_sectional_momentum",
        })
        result = run_recommend_next_actions(spec=spec)
        assert result.execution_performed is False
        assert result.draft_approved is False


# ---------------------------------------------------------------------------
# Internal: _apply_truncation helper
# ---------------------------------------------------------------------------


class TestApplyTruncationHelper:
    def test_no_truncation_when_under_limit(self):
        from src.agent.workflow.next_action_workflow import _apply_truncation
        from src.agent.next_actions.next_action_result import NextActionRecommendationResult
        from src.agent.next_actions.next_action import NextActionCandidate

        c = NextActionCandidate(
            action_id="x1", title="T", action_type="feature_ablation",
            description="d", rationale="r", main_risks=["r1"],
        )
        na = NextActionRecommendationResult(status="recommended", objective=None, candidates=[c])
        warnings = _apply_truncation(na, max_candidates=5)
        assert warnings == []
        assert len(na.candidates) == 1

    def test_truncation_emits_warning(self):
        from src.agent.workflow.next_action_workflow import _apply_truncation
        from src.agent.next_actions.next_action_result import NextActionRecommendationResult
        from src.agent.next_actions.next_action import NextActionCandidate

        candidates = [
            NextActionCandidate(
                action_id=f"x{i}", title=f"T{i}", action_type="feature_ablation",
                description="d", rationale="r", main_risks=["r"],
            )
            for i in range(4)
        ]
        # Give each a unique action_type so dedup doesn't remove them
        action_types = [
            "feature_ablation", "model_comparison",
            "validation_robustness", "cost_sensitivity",
        ]
        for c, at in zip(candidates, action_types):
            c.action_type = at

        na = NextActionRecommendationResult(
            status="recommended", objective=None, candidates=candidates,
            summary="original",
        )
        warnings = _apply_truncation(na, max_candidates=2)
        assert len(warnings) == 1
        assert "max_candidates=2" in warnings[0]
        assert len(na.candidates) == 2
