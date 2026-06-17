"""recommend_next_actions unit and integration tests — Phase 5B.

Covers:
  A. Empty memory → needs_more_evidence
  B. Weak IC / poor OOS consistency
  C. Allocation-temperature negligible impact
  D. High turnover / cost drag
  E. Overfitting / high split variance
  F. Strong baseline — no critical failure modes
  G. Governance invariants on every result
  H. Result structure and serialisation
  I. Integration: spec → resolve → plan → recommend (with stub context)
"""

from __future__ import annotations

import pytest

from src.agent.memory.memory_context import ResearchMemoryContext
from src.agent.memory.memory_evidence import ResearchMemoryEvidence
from src.agent.memory.memory_query import ResearchMemoryQuery
from src.agent.next_actions.next_action_planner import recommend_next_actions
from src.agent.next_actions.next_action_result import (
    NextActionRecommendationResult,
    summarise_next_action_result,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_query() -> ResearchMemoryQuery:
    return ResearchMemoryQuery(keyword_query="ridge", semantic_query="ridge regression")


def _make_ev(
    memory_id: str,
    experiment_name: str = "exp_v1",
    artefact_type: str = "experiment_metrics",
    short_summary: str = "Test evidence item",
    failure_modes: list[str] | None = None,
    tags: list[str] | None = None,
) -> ResearchMemoryEvidence:
    return ResearchMemoryEvidence(
        memory_id=memory_id,
        experiment_name=experiment_name,
        artefact_type=artefact_type,
        short_summary=short_summary,
        failure_modes=failure_modes or [],
        tags=tags or [],
        path=f"/results/{experiment_name}/metrics.json",
        context_hash="hash_" + memory_id,
        retrieval_method="keyword",
    )


def _ctx(*evidence: ResearchMemoryEvidence) -> ResearchMemoryContext:
    return ResearchMemoryContext(
        query=_make_query(),
        evidence_items=list(evidence),
        keyword_count=len(evidence),
    )


def _ctx_empty() -> ResearchMemoryContext:
    return ResearchMemoryContext(
        query=ResearchMemoryQuery(keyword_query=None, semantic_query=None),
    )


def _ctx_weak_oos() -> ResearchMemoryContext:
    return _ctx(
        _make_ev("mem001", failure_modes=["poor_oos_consistency", "weak_ic"]),
        _make_ev("mem002", failure_modes=["poor_oos_consistency"]),
    )


def _ctx_allocation_negligible() -> ResearchMemoryContext:
    return _ctx(
        _make_ev(
            "mem_alloc",
            experiment_name="allocation_study",
            artefact_type="comparison_evidence",
            short_summary=(
                "Allocation temperature sweep showed negligible impact on Sharpe ratio"
            ),
            failure_modes=[],
            tags=["comparison"],
        ),
    )


def _ctx_high_turnover() -> ResearchMemoryContext:
    return _ctx(
        _make_ev("mem_to", failure_modes=["high_turnover"]),
    )


def _ctx_overfitting() -> ResearchMemoryContext:
    return _ctx(
        _make_ev(
            "mem_of",
            failure_modes=["high_split_sharpe_variance", "coefficient_instability"],
        ),
    )


def _ctx_strong_baseline() -> ResearchMemoryContext:
    return _ctx(
        _make_ev(
            "mem_base",
            experiment_name="canonical_v1",
            short_summary="OOS Sharpe 0.52, moderate consistency across 8 splits",
            failure_modes=[],
        ),
    )


# ---------------------------------------------------------------------------
# A. Empty memory
# ---------------------------------------------------------------------------


class TestEmptyMemory:
    def test_status_is_needs_more_evidence(self):
        result = recommend_next_actions(memory_context=_ctx_empty())
        assert result.status == "needs_more_evidence"

    def test_recommends_repeat_baseline_or_diagnostic(self):
        result = recommend_next_actions(memory_context=_ctx_empty())
        types = [c.action_type for c in result.candidates]
        assert "repeat_baseline" in types or "diagnostic_review" in types

    def test_all_candidates_low_confidence(self):
        result = recommend_next_actions(memory_context=_ctx_empty())
        for c in result.candidates:
            assert c.confidence == "low"

    def test_summary_explains_evidence_missing(self):
        result = recommend_next_actions(memory_context=_ctx_empty())
        summary_lower = result.summary.lower()
        assert "evidence" in summary_lower or "memory" in summary_lower or "baseline" in summary_lower

    def test_no_crash_with_no_args(self):
        result = recommend_next_actions()
        assert isinstance(result, NextActionRecommendationResult)

    def test_no_crash_with_objective_only(self):
        result = recommend_next_actions(
            objective="improve OOS consistency",
            memory_context=_ctx_empty(),
        )
        assert result.status == "needs_more_evidence"

    def test_candidates_have_risks(self):
        result = recommend_next_actions(memory_context=_ctx_empty())
        for c in result.candidates:
            assert c.main_risks or c.failure_modes_to_monitor

    def test_governance_flags_on_empty_result(self):
        result = recommend_next_actions(memory_context=_ctx_empty())
        assert result.approval_required is True
        assert result.execution_performed is False
        assert result.draft_created is False


# ---------------------------------------------------------------------------
# B. Weak IC / poor OOS consistency
# ---------------------------------------------------------------------------


class TestWeakOOSConsistency:
    def test_status_is_recommended(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        assert result.status == "recommended"

    def test_recommends_feature_ablation(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        types = [c.action_type for c in result.candidates]
        assert "feature_ablation" in types

    def test_recommends_diagnostic_review(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        types = [c.action_type for c in result.candidates]
        assert "diagnostic_review" in types

    def test_top_candidate_is_feature_ablation_or_diagnostic(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        assert result.top_candidate is not None
        assert result.top_candidate.action_type in {
            "feature_ablation",
            "diagnostic_review",
        }

    def test_evidence_ids_attached_to_top_candidate(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        assert result.top_candidate is not None
        assert result.top_candidate.supporting_evidence_ids

    def test_failure_modes_to_monitor_populated(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        assert result.top_candidate is not None
        fm = result.top_candidate.failure_modes_to_monitor
        assert "poor_oos_consistency" in fm or "weak_ic" in fm

    def test_top_priority_is_high(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        assert result.top_candidate is not None
        assert result.top_candidate.priority == "high"

    def test_rationale_does_not_claim_guarantee(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        for c in result.candidates:
            text = (c.rationale + c.description).lower()
            assert "will improve" not in text
            assert "guaranteed" not in text

    def test_evidence_summary_references_failure_modes(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        assert "poor_oos_consistency" in result.evidence_summary


# ---------------------------------------------------------------------------
# C. Allocation-temperature negligible impact
# ---------------------------------------------------------------------------


class TestAllocationNegligible:
    def test_rejected_directions_contains_allocation_caution(self):
        result = recommend_next_actions(memory_context=_ctx_allocation_negligible())
        assert result.rejected_directions
        rd_text = " ".join(result.rejected_directions).lower()
        assert "allocation" in rd_text

    def test_top_candidate_is_not_portfolio_construction(self):
        result = recommend_next_actions(memory_context=_ctx_allocation_negligible())
        if result.top_candidate:
            assert result.top_candidate.action_type != "portfolio_construction_test"

    def test_recommends_diagnostic_or_robustness_not_allocation(self):
        result = recommend_next_actions(memory_context=_ctx_allocation_negligible())
        types = {c.action_type for c in result.candidates}
        assert types & {"diagnostic_review", "validation_robustness", "cost_sensitivity"}

    def test_cautions_include_allocation_note(self):
        result = recommend_next_actions(memory_context=_ctx_allocation_negligible())
        caution_text = " ".join(result.recommended_cautions).lower()
        assert "allocation" in caution_text


# ---------------------------------------------------------------------------
# D. High turnover / cost drag
# ---------------------------------------------------------------------------


class TestHighTurnover:
    def test_status_is_recommended(self):
        result = recommend_next_actions(memory_context=_ctx_high_turnover())
        assert result.status == "recommended"

    def test_recommends_cost_sensitivity(self):
        result = recommend_next_actions(memory_context=_ctx_high_turnover())
        types = [c.action_type for c in result.candidates]
        assert "cost_sensitivity" in types

    def test_cost_candidate_monitors_high_turnover(self):
        result = recommend_next_actions(memory_context=_ctx_high_turnover())
        cost_c = next(
            (c for c in result.candidates if c.action_type == "cost_sensitivity"), None
        )
        assert cost_c is not None
        assert "high_turnover" in cost_c.failure_modes_to_monitor

    def test_cost_candidate_has_high_priority(self):
        result = recommend_next_actions(memory_context=_ctx_high_turnover())
        cost_c = next(
            (c for c in result.candidates if c.action_type == "cost_sensitivity"), None
        )
        assert cost_c is not None
        assert cost_c.priority == "high"

    def test_cost_candidate_has_evidence_ids(self):
        result = recommend_next_actions(memory_context=_ctx_high_turnover())
        cost_c = next(
            (c for c in result.candidates if c.action_type == "cost_sensitivity"), None
        )
        assert cost_c is not None
        assert cost_c.supporting_evidence_ids


# ---------------------------------------------------------------------------
# E. Overfitting / high split variance
# ---------------------------------------------------------------------------


class TestOverfitting:
    def test_status_is_recommended(self):
        result = recommend_next_actions(memory_context=_ctx_overfitting())
        assert result.status == "recommended"

    def test_recommends_validation_robustness(self):
        result = recommend_next_actions(memory_context=_ctx_overfitting())
        types = [c.action_type for c in result.candidates]
        assert "validation_robustness" in types

    def test_recommends_model_comparison(self):
        # Need max_candidates=4 because overfitting overlaps with signal-quality rule
        # (high_split_sharpe_variance is in both groups), producing 4 distinct candidates.
        result = recommend_next_actions(memory_context=_ctx_overfitting(), max_candidates=4)
        types = [c.action_type for c in result.candidates]
        assert "model_comparison" in types

    def test_overfitting_failure_modes_monitored(self):
        result = recommend_next_actions(memory_context=_ctx_overfitting())
        top = result.top_candidate
        assert top is not None
        fm = set(top.failure_modes_to_monitor)
        assert fm & {
            "high_split_sharpe_variance",
            "coefficient_instability",
            "catastrophic_split",
        }

    def test_validation_candidate_has_risks(self):
        result = recommend_next_actions(memory_context=_ctx_overfitting())
        val_c = next(
            (c for c in result.candidates if c.action_type == "validation_robustness"),
            None,
        )
        assert val_c is not None
        assert val_c.main_risks


# ---------------------------------------------------------------------------
# F. Strong baseline — no critical failure modes
# ---------------------------------------------------------------------------


class TestStrongBaseline:
    def test_status_is_recommended(self):
        result = recommend_next_actions(memory_context=_ctx_strong_baseline())
        assert result.status == "recommended"

    def test_recommends_validation_robustness(self):
        result = recommend_next_actions(memory_context=_ctx_strong_baseline())
        types = [c.action_type for c in result.candidates]
        assert "validation_robustness" in types

    def test_recommends_cost_sensitivity(self):
        result = recommend_next_actions(memory_context=_ctx_strong_baseline())
        types = [c.action_type for c in result.candidates]
        assert "cost_sensitivity" in types

    def test_no_feature_ablation_without_signal_failure_modes(self):
        result = recommend_next_actions(memory_context=_ctx_strong_baseline())
        types = [c.action_type for c in result.candidates]
        assert "feature_ablation" not in types

    def test_no_model_comparison_without_overfitting_evidence(self):
        result = recommend_next_actions(memory_context=_ctx_strong_baseline())
        types = [c.action_type for c in result.candidates]
        assert "model_comparison" not in types


# ---------------------------------------------------------------------------
# G. Governance invariants
# ---------------------------------------------------------------------------


class TestGovernanceInvariants:
    @pytest.mark.parametrize(
        "ctx_fn",
        [
            _ctx_empty,
            _ctx_weak_oos,
            _ctx_allocation_negligible,
            _ctx_high_turnover,
            _ctx_overfitting,
            _ctx_strong_baseline,
        ],
    )
    def test_approval_required_always_true(self, ctx_fn):
        result = recommend_next_actions(memory_context=ctx_fn())
        assert result.approval_required is True

    @pytest.mark.parametrize(
        "ctx_fn",
        [
            _ctx_empty,
            _ctx_weak_oos,
            _ctx_high_turnover,
            _ctx_overfitting,
            _ctx_strong_baseline,
        ],
    )
    def test_execution_performed_always_false(self, ctx_fn):
        result = recommend_next_actions(memory_context=ctx_fn())
        assert result.execution_performed is False

    @pytest.mark.parametrize(
        "ctx_fn",
        [
            _ctx_empty,
            _ctx_weak_oos,
            _ctx_high_turnover,
            _ctx_overfitting,
            _ctx_strong_baseline,
        ],
    )
    def test_draft_created_always_false(self, ctx_fn):
        result = recommend_next_actions(memory_context=ctx_fn())
        assert result.draft_created is False

    @pytest.mark.parametrize(
        "ctx_fn",
        [_ctx_weak_oos, _ctx_high_turnover, _ctx_overfitting, _ctx_strong_baseline],
    )
    def test_all_candidates_never_create_draft(self, ctx_fn):
        result = recommend_next_actions(memory_context=ctx_fn())
        for c in result.candidates:
            assert c.creates_draft is False

    @pytest.mark.parametrize(
        "ctx_fn",
        [_ctx_weak_oos, _ctx_high_turnover, _ctx_overfitting, _ctx_strong_baseline],
    )
    def test_all_candidates_never_execute_experiment(self, ctx_fn):
        result = recommend_next_actions(memory_context=ctx_fn())
        for c in result.candidates:
            assert c.executes_experiment is False

    @pytest.mark.parametrize(
        "ctx_fn",
        [_ctx_weak_oos, _ctx_high_turnover, _ctx_overfitting, _ctx_strong_baseline],
    )
    def test_all_candidates_require_confirmation(self, ctx_fn):
        result = recommend_next_actions(memory_context=ctx_fn())
        for c in result.candidates:
            assert c.requires_user_confirmation is True


# ---------------------------------------------------------------------------
# H. Result structure and serialisation
# ---------------------------------------------------------------------------


class TestResultStructure:
    def test_to_dict_has_all_keys(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        d = result.to_dict()
        expected_keys = {
            "status",
            "objective",
            "candidates",
            "top_candidate",
            "memory_context",
            "evidence_summary",
            "rejected_directions",
            "recommended_cautions",
            "warnings",
            "errors",
            "summary",
            "approval_required",
            "execution_performed",
            "draft_created",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_candidates_are_dicts(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        d = result.to_dict()
        for c in d["candidates"]:
            assert isinstance(c, dict)

    def test_top_candidate_in_to_dict_is_dict_or_none(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        d = result.to_dict()
        assert isinstance(d["top_candidate"], dict) or d["top_candidate"] is None

    def test_memory_context_in_to_dict_is_dict_or_none(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        d = result.to_dict()
        assert isinstance(d["memory_context"], dict) or d["memory_context"] is None

    def test_summary_is_string(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_summary_mentions_user_confirmation(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        summary = result.summary.lower()
        assert "confirm" in summary or "confirmation" in summary

    def test_max_candidates_respected(self):
        result = recommend_next_actions(
            memory_context=_ctx_weak_oos(), max_candidates=1
        )
        assert len(result.candidates) <= 1

    def test_max_candidates_default(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        assert len(result.candidates) <= 3

    def test_top_candidate_is_first_candidate(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        if result.candidates and result.top_candidate:
            assert result.top_candidate is result.candidates[0]

    def test_evidence_summary_populated(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        assert result.evidence_summary
        assert len(result.evidence_summary) > 0

    def test_objective_stored_when_provided(self):
        result = recommend_next_actions(
            objective="test objective xyz",
            memory_context=_ctx_weak_oos(),
        )
        assert result.objective == "test objective xyz"

    def test_objective_none_when_not_provided(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        # No objective was passed, no spec with objective
        assert result.objective is None


# ---------------------------------------------------------------------------
# I. Integration: plan → query → stub context → recommend
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_integration_with_spec_and_stub_context(self):
        """ResearchSpec → ResolvedResearchSpec → ExperimentPlan → recommend."""
        from src.agent.planner.experiment_plan import ExperimentPlan
        from src.agent.resolver.resolved_spec import ResolvedResearchSpec

        # Construct resolved spec manually (avoids needing registry on disk)
        resolved = ResolvedResearchSpec(
            preset="single_asset_ml",
            model="RidgeRegressionModel",
            feature_families=["momentum", "volatility"],
            label="forward_return_1m",
            objective="improve OOS Sharpe consistency",
            user_specified_fields={"model": "RidgeRegressionModel"},
        )

        plan = ExperimentPlan(
            plan_id="plan-test-001",
            name="ridge-oos-consistency",
            description="Improve OOS consistency for Ridge model",
            resolved_spec=resolved,
            research_objective="Improve OOS consistency",
            hypothesis="Reducing model complexity will reduce inter-split variance",
            experiment_type="model_comparison",
            baseline="canonical_ml_showcase_v2",
            comparison_policy="compare_to_baseline",
            failure_modes_to_monitor=["poor_oos_consistency", "high_split_sharpe_variance"],
            planned_changes=["Reduce regularisation alpha", "Add IC diagnostic"],
        )

        # Stub memory context with relevant evidence
        ctx = _ctx(
            _make_ev(
                "int001",
                experiment_name="canonical_ml_showcase_v2",
                failure_modes=["poor_oos_consistency"],
                short_summary="OOS Sharpe 0.31, split variance 0.18",
            )
        )

        result = recommend_next_actions(
            experiment_plan=plan,
            memory_context=ctx,
        )

        assert result.status == "recommended"
        assert result.approval_required is True
        assert result.execution_performed is False
        assert result.draft_created is False
        assert result.candidates
        assert result.top_candidate is not None
        assert result.objective == "Improve OOS consistency"

    def test_integration_evidence_ids_reference_stub_context(self):
        from src.agent.planner.experiment_plan import ExperimentPlan
        from src.agent.resolver.resolved_spec import ResolvedResearchSpec

        resolved = ResolvedResearchSpec(
            model="RidgeRegressionModel",
            feature_families=["momentum"],
        )
        plan = ExperimentPlan(
            plan_id="plan-test-002",
            name="plan-test",
            description="test",
            resolved_spec=resolved,
            research_objective="test objective",
            hypothesis="test hypothesis",
            experiment_type="model_comparison",
            baseline=None,
            comparison_policy="none",
            failure_modes_to_monitor=["weak_ic"],
        )
        ctx = _ctx(
            _make_ev("ev_int_001", failure_modes=["weak_ic"])
        )
        result = recommend_next_actions(experiment_plan=plan, memory_context=ctx)
        top = result.top_candidate
        assert top is not None
        assert "ev_int_001" in top.supporting_evidence_ids

    def test_summarise_next_action_result_standalone(self):
        result = recommend_next_actions(memory_context=_ctx_weak_oos())
        summary = summarise_next_action_result(result)
        assert isinstance(summary, str)
        assert "confirmation" in summary.lower() or "confirm" in summary.lower()
