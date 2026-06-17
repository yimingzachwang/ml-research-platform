"""critique_next_action_recommendations unit tests — Phase 5B.

Covers:
  A. CRITIC_A — no performance guarantees
  B. CRITIC_B — high-priority candidates need evidence IDs when evidence exists
  C. CRITIC_C — creates_draft=True or executes_experiment=True are errors
  D. CRITIC_D — requires_user_confirmation=False is an error
  E. CRITIC_E — no risks and no failure_modes_to_monitor is an error
  F. CRITIC_F — recommended status without candidates is an error
  G. CRITIC_G — needs_more_evidence without explanation in summary is a warning
  H. CRITIC_H — invalid priority is an error
  I. CRITIC_I — invalid confidence is an error
  J. Valid result passes all checks
"""

from __future__ import annotations

import pytest

from src.agent.memory.memory_context import ResearchMemoryContext
from src.agent.memory.memory_evidence import ResearchMemoryEvidence
from src.agent.memory.memory_query import ResearchMemoryQuery
from src.agent.next_actions.next_action import NextActionCandidate
from src.agent.next_actions.next_action_critic import (
    NextActionCriticIssue,
    NextActionCriticResult,
    critique_next_action_recommendations,
)
from src.agent.next_actions.next_action_result import NextActionRecommendationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidate(
    action_id: str = "cnd001",
    action_type: str = "feature_ablation",
    priority: str = "medium",
    confidence: str = "medium",
    rationale: str = "Prior evidence suggests this direction.",
    description: str = "Run an ablation study.",
    expected_benefit: str | None = None,
    supporting_evidence_ids: list[str] | None = None,
    main_risks: list[str] | None = None,
    failure_modes_to_monitor: list[str] | None = None,
    requires_user_confirmation: bool = True,
    creates_draft: bool = False,
    executes_experiment: bool = False,
) -> NextActionCandidate:
    return NextActionCandidate(
        action_id=action_id,
        title="Test candidate",
        action_type=action_type,
        description=description,
        rationale=rationale,
        supporting_evidence_ids=supporting_evidence_ids or [],
        main_risks=main_risks or ["test risk"],
        failure_modes_to_monitor=failure_modes_to_monitor or [],
        priority=priority,
        confidence=confidence,
        expected_benefit=expected_benefit,
        requires_user_confirmation=requires_user_confirmation,
        creates_draft=creates_draft,
        executes_experiment=executes_experiment,
    )


def _result(
    status: str = "recommended",
    candidates: list[NextActionCandidate] | None = None,
    memory_context: ResearchMemoryContext | None = None,
    summary: str = "Test summary with evidence baseline",
) -> NextActionRecommendationResult:
    cands = candidates if candidates is not None else [_candidate()]
    top = cands[0] if cands else None
    return NextActionRecommendationResult(
        status=status,
        objective=None,
        candidates=cands,
        top_candidate=top,
        memory_context=memory_context,
        summary=summary,
    )


def _empty_ctx() -> ResearchMemoryContext:
    return ResearchMemoryContext(
        query=ResearchMemoryQuery(keyword_query=None, semantic_query=None),
    )


def _ctx_with_evidence() -> ResearchMemoryContext:
    ev = ResearchMemoryEvidence(
        memory_id="m001",
        experiment_name="exp_v1",
        artefact_type="experiment_metrics",
        short_summary="OOS Sharpe 0.31, consistency issues",
        failure_modes=["poor_oos_consistency"],
        tags=[],
        retrieval_method="keyword",
    )
    return ResearchMemoryContext(
        query=ResearchMemoryQuery(keyword_query="ridge", semantic_query="ridge"),
        evidence_items=[ev],
        keyword_count=1,
    )


def _check_ids(critic_result: NextActionCriticResult) -> list[str]:
    return [i.check_id for i in critic_result.issues]


def _error_ids(critic_result: NextActionCriticResult) -> list[str]:
    return [i.check_id for i in critic_result.errors]


def _warning_ids(critic_result: NextActionCriticResult) -> list[str]:
    return [i.check_id for i in critic_result.warnings]


# ---------------------------------------------------------------------------
# A. CRITIC_A — performance guarantees
# ---------------------------------------------------------------------------


class TestCriticA:
    def test_will_improve_in_rationale_is_error(self):
        c = _candidate(rationale="This will improve performance significantly.")
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_A" in _error_ids(cr)

    def test_guaranteed_in_description_is_error(self):
        c = _candidate(description="This approach is guaranteed to work.")
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_A" in _error_ids(cr)

    def test_guaranteed_in_expected_benefit_is_error(self):
        c = _candidate(
            expected_benefit="This is guaranteed to improve Sharpe.",
            main_risks=["test risk"],
        )
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_A" in _error_ids(cr)

    def test_cautious_language_is_valid(self):
        c = _candidate(rationale="Prior evidence suggests this direction may help.")
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_A" not in _error_ids(cr)

    def test_will_outperform_in_rationale_is_error(self):
        c = _candidate(rationale="This will outperform the baseline.")
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_A" in _error_ids(cr)


# ---------------------------------------------------------------------------
# B. CRITIC_B — high priority with evidence needs evidence IDs
# ---------------------------------------------------------------------------


class TestCriticB:
    def test_high_priority_no_evidence_ids_with_memory_is_warning(self):
        c = _candidate(priority="high", supporting_evidence_ids=[])
        r = _result(candidates=[c], memory_context=_ctx_with_evidence())
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_B" in _warning_ids(cr)

    def test_high_priority_with_evidence_ids_is_valid(self):
        c = _candidate(priority="high", supporting_evidence_ids=["m001"])
        r = _result(candidates=[c], memory_context=_ctx_with_evidence())
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_B" not in _warning_ids(cr)

    def test_medium_priority_no_evidence_ids_no_warning(self):
        c = _candidate(priority="medium", supporting_evidence_ids=[])
        r = _result(candidates=[c], memory_context=_ctx_with_evidence())
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_B" not in _warning_ids(cr)

    def test_high_priority_no_evidence_when_no_memory_no_warning(self):
        # No memory context → no warning about missing evidence IDs
        c = _candidate(priority="high", supporting_evidence_ids=[])
        r = _result(candidates=[c], memory_context=_empty_ctx())
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_B" not in _warning_ids(cr)


# ---------------------------------------------------------------------------
# C. CRITIC_C — creates_draft / executes_experiment
# ---------------------------------------------------------------------------


class TestCriticC:
    def test_creates_draft_true_is_error(self):
        c = _candidate(creates_draft=True)
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_C" in _error_ids(cr)

    def test_executes_experiment_true_is_error(self):
        c = _candidate(executes_experiment=True)
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_C" in _error_ids(cr)

    def test_both_false_is_valid(self):
        c = _candidate(creates_draft=False, executes_experiment=False)
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_C" not in _error_ids(cr)

    def test_creates_draft_error_references_candidate_id(self):
        c = _candidate(action_id="bad-cnd", creates_draft=True)
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        critic_c_issues = [i for i in cr.issues if i.check_id == "CRITIC_C"]
        assert any(i.candidate_id == "bad-cnd" for i in critic_c_issues)


# ---------------------------------------------------------------------------
# D. CRITIC_D — requires_user_confirmation
# ---------------------------------------------------------------------------


class TestCriticD:
    def test_requires_confirmation_false_is_error(self):
        c = _candidate(requires_user_confirmation=False)
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_D" in _error_ids(cr)

    def test_requires_confirmation_true_is_valid(self):
        c = _candidate(requires_user_confirmation=True)
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_D" not in _error_ids(cr)


# ---------------------------------------------------------------------------
# E. CRITIC_E — no risks / no failure modes
# ---------------------------------------------------------------------------


class TestCriticE:
    def test_no_risks_and_no_failure_modes_is_error(self):
        c = NextActionCandidate(
            action_id="bare",
            title="bare",
            action_type="custom",
            description="desc",
            rationale="Prior evidence suggests.",
            main_risks=[],
            failure_modes_to_monitor=[],
        )
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_E" in _error_ids(cr)

    def test_only_main_risks_is_valid(self):
        c = _candidate(main_risks=["some risk"], failure_modes_to_monitor=[])
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_E" not in _error_ids(cr)

    def test_only_failure_modes_is_valid(self):
        c = _candidate(main_risks=[], failure_modes_to_monitor=["weak_ic"])
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_E" not in _error_ids(cr)


# ---------------------------------------------------------------------------
# F. CRITIC_F — recommended status without candidates
# ---------------------------------------------------------------------------


class TestCriticF:
    def test_recommended_status_no_candidates_is_error(self):
        r = _result(status="recommended", candidates=[])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_F" in _error_ids(cr)

    def test_recommended_status_with_candidates_no_error(self):
        r = _result(status="recommended", candidates=[_candidate()])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_F" not in _error_ids(cr)

    def test_needs_more_evidence_no_candidates_no_critic_f_error(self):
        r = NextActionRecommendationResult(
            status="needs_more_evidence",
            objective=None,
            candidates=[],
            top_candidate=None,
            summary="No prior memory evidence or baseline is available.",
        )
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_F" not in _error_ids(cr)


# ---------------------------------------------------------------------------
# G. CRITIC_G — needs_more_evidence without explanation
# ---------------------------------------------------------------------------


class TestCriticG:
    def test_needs_more_evidence_no_explanation_is_warning(self):
        r = NextActionRecommendationResult(
            status="needs_more_evidence",
            objective=None,
            candidates=[],
            top_candidate=None,
            summary="No actionable path forward.",
        )
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_G" in _warning_ids(cr)

    def test_needs_more_evidence_with_evidence_keyword_no_warning(self):
        r = NextActionRecommendationResult(
            status="needs_more_evidence",
            objective=None,
            candidates=[],
            top_candidate=None,
            summary="No prior research evidence is available. Run a baseline first.",
        )
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_G" not in _warning_ids(cr)

    def test_needs_more_evidence_with_baseline_keyword_no_warning(self):
        r = NextActionRecommendationResult(
            status="needs_more_evidence",
            objective=None,
            candidates=[],
            top_candidate=None,
            summary="A baseline run is required before specific recommendations.",
        )
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_G" not in _warning_ids(cr)

    def test_empty_summary_does_not_trigger_critic_g(self):
        # Empty summary means we can't judge it — don't warn
        r = NextActionRecommendationResult(
            status="needs_more_evidence",
            objective=None,
            summary="",
        )
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_G" not in _warning_ids(cr)


# ---------------------------------------------------------------------------
# H. CRITIC_H — invalid priority
# ---------------------------------------------------------------------------


class TestCriticH:
    def test_invalid_priority_is_error(self):
        c = _candidate(priority="urgent")
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_H" in _error_ids(cr)

    def test_valid_high_priority_no_error(self):
        c = _candidate(priority="high")
        r = _result(candidates=[c], memory_context=_ctx_with_evidence())
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_H" not in _error_ids(cr)

    def test_valid_medium_priority_no_error(self):
        c = _candidate(priority="medium")
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_H" not in _error_ids(cr)

    def test_valid_low_priority_no_error(self):
        c = _candidate(priority="low")
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_H" not in _error_ids(cr)


# ---------------------------------------------------------------------------
# I. CRITIC_I — invalid confidence
# ---------------------------------------------------------------------------


class TestCriticI:
    def test_invalid_confidence_is_error(self):
        c = _candidate(confidence="certain")
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_I" in _error_ids(cr)

    def test_valid_confidence_high_no_error(self):
        c = _candidate(confidence="high")
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_I" not in _error_ids(cr)

    def test_valid_confidence_low_no_error(self):
        c = _candidate(confidence="low")
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        assert "CRITIC_I" not in _error_ids(cr)


# ---------------------------------------------------------------------------
# J. Valid result passes all checks
# ---------------------------------------------------------------------------


class TestCriticValidResult:
    def test_valid_recommended_result_has_no_errors(self):
        c = _candidate(
            action_id="v001",
            priority="high",
            confidence="medium",
            rationale="Prior evidence suggests this cautious direction.",
            supporting_evidence_ids=["m001"],
            main_risks=["some risk"],
        )
        r = _result(
            status="recommended",
            candidates=[c],
            memory_context=_ctx_with_evidence(),
            summary="Top recommendation: feature ablation with evidence from memory.",
        )
        cr = critique_next_action_recommendations(r)
        assert cr.is_valid
        assert cr.errors == []

    def test_valid_needs_more_evidence_result_has_no_errors(self):
        r = NextActionRecommendationResult(
            status="needs_more_evidence",
            objective=None,
            candidates=[
                _candidate(
                    action_id="nme001",
                    priority="low",
                    confidence="low",
                    action_type="repeat_baseline",
                    main_risks=["baseline may not provide enough evidence"],
                )
            ],
            summary="No prior memory evidence or baseline is available.",
        )
        cr = critique_next_action_recommendations(r)
        assert cr.is_valid

    def test_is_valid_true_when_no_errors(self):
        r = _result(status="recommended", candidates=[_candidate()])
        cr = critique_next_action_recommendations(r)
        assert cr.is_valid is (not cr.errors)

    def test_to_dict_has_all_keys(self):
        r = _result(candidates=[_candidate()])
        cr = critique_next_action_recommendations(r)
        d = cr.to_dict()
        assert {"issues", "error_count", "warning_count", "is_valid"} == set(d.keys())

    def test_to_dict_issues_are_dicts(self):
        c = _candidate(rationale="This will improve performance.")
        r = _result(candidates=[c])
        cr = critique_next_action_recommendations(r)
        d = cr.to_dict()
        for issue in d["issues"]:
            assert isinstance(issue, dict)
            assert "check_id" in issue
            assert "severity" in issue
            assert "message" in issue

    def test_critic_result_dataclass_properties(self):
        r = _result(candidates=[_candidate(rationale="This will improve performance.")])
        cr = critique_next_action_recommendations(r)
        assert cr.errors == [i for i in cr.issues if i.severity == "error"]
        assert cr.warnings == [i for i in cr.issues if i.severity == "warning"]

    def test_planner_output_passes_critic(self):
        """End-to-end: planner output should pass critic without errors."""
        from src.agent.memory.memory_context import ResearchMemoryContext
        from src.agent.memory.memory_evidence import ResearchMemoryEvidence
        from src.agent.memory.memory_query import ResearchMemoryQuery
        from src.agent.next_actions.next_action_planner import recommend_next_actions

        ev = ResearchMemoryEvidence(
            memory_id="m001",
            experiment_name="exp",
            artefact_type="experiment_metrics",
            short_summary="OOS Sharpe 0.31",
            failure_modes=["poor_oos_consistency"],
            tags=[],
            retrieval_method="keyword",
        )
        ctx = ResearchMemoryContext(
            query=ResearchMemoryQuery(keyword_query="ridge", semantic_query="ridge"),
            evidence_items=[ev],
            keyword_count=1,
        )
        result = recommend_next_actions(memory_context=ctx)
        cr = critique_next_action_recommendations(result)
        assert cr.is_valid, f"Critic found errors: {[i.message for i in cr.errors]}"
