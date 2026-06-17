"""ResearchMemoryQuery unit tests — Phase 5A.

Covers:
  A. ResearchMemoryQuery dataclass construction and serialisation
  B. query_from_research_spec — keyword / semantic extraction
  C. query_from_resolved_spec — keyword / semantic extraction
  D. query_from_experiment_plan — failure modes, baseline, objective
"""

from __future__ import annotations

import pytest

from src.agent.intent.research_spec import ResearchSpec
from src.agent.memory.memory_query import (
    ResearchMemoryQuery,
    query_from_experiment_plan,
    query_from_research_spec,
    query_from_resolved_spec,
)
from src.agent.planner.experiment_plan import ExperimentPlan
from src.agent.resolver.resolved_spec import ResolvedResearchSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_resolved(
    model: str | None = None,
    feature_families: list[str] | None = None,
    label: str | None = None,
    objective: str | None = None,
    notes: str | None = None,
    features: list[str] | None = None,
) -> ResolvedResearchSpec:
    return ResolvedResearchSpec(
        model=model,
        feature_families=feature_families or [],
        label=label,
        objective=objective,
        notes=notes,
        features=features or [],
    )


def _make_plan(
    resolved: ResolvedResearchSpec | None = None,
    baseline: str | None = None,
    failure_modes_to_monitor: list[str] | None = None,
    research_objective: str = "test objective",
    hypothesis: str = "test hypothesis",
    planned_changes: list[str] | None = None,
) -> ExperimentPlan:
    if resolved is None:
        resolved = _make_resolved()
    return ExperimentPlan(
        plan_id="plan-001",
        name="test-plan",
        description="test",
        resolved_spec=resolved,
        research_objective=research_objective,
        hypothesis=hypothesis,
        experiment_type="model_comparison",
        baseline=baseline,
        comparison_policy="compare_to_baseline",
        failure_modes_to_monitor=failure_modes_to_monitor or [],
        planned_changes=planned_changes or [],
    )


# ---------------------------------------------------------------------------
# A. ResearchMemoryQuery dataclass
# ---------------------------------------------------------------------------


class TestResearchMemoryQueryDataclass:
    def test_construction_with_required_fields(self):
        q = ResearchMemoryQuery(keyword_query="ridge momentum", semantic_query="test")
        assert q.keyword_query == "ridge momentum"
        assert q.semantic_query == "test"

    def test_defaults(self):
        q = ResearchMemoryQuery(keyword_query=None, semantic_query=None)
        assert q.failure_modes == []
        assert q.experiment_names == []
        assert q.artefact_types == []
        assert q.tags == []
        assert q.top_k_keyword == 5
        assert q.top_k_semantic == 5
        assert q.source == "manual"

    def test_to_dict_contains_all_keys(self):
        q = ResearchMemoryQuery(keyword_query="ridge", semantic_query="ridge model")
        d = q.to_dict()
        expected_keys = {
            "keyword_query",
            "semantic_query",
            "failure_modes",
            "experiment_names",
            "artefact_types",
            "tags",
            "top_k_keyword",
            "top_k_semantic",
            "source",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_values_are_copies(self):
        q = ResearchMemoryQuery(
            keyword_query="ridge",
            semantic_query="ridge",
            failure_modes=["poor_oos_consistency"],
        )
        d = q.to_dict()
        d["failure_modes"].append("weak_ic")
        assert q.failure_modes == ["poor_oos_consistency"]

    def test_none_keyword_query_serialises(self):
        q = ResearchMemoryQuery(keyword_query=None, semantic_query=None)
        d = q.to_dict()
        assert d["keyword_query"] is None
        assert d["semantic_query"] is None


# ---------------------------------------------------------------------------
# B. query_from_research_spec
# ---------------------------------------------------------------------------


class TestQueryFromResearchSpec:
    def test_empty_spec_gives_none_keyword_query(self):
        spec = ResearchSpec()
        q = query_from_research_spec(spec)
        assert q.keyword_query is None

    def test_model_appears_in_keyword_query(self):
        spec = ResearchSpec(model="RidgeRegressionModel")
        q = query_from_research_spec(spec)
        assert q.keyword_query is not None
        assert "ridgeregressionmodel" in q.keyword_query

    def test_feature_families_appear_in_keyword_query(self):
        spec = ResearchSpec(feature_families=["momentum", "volatility"])
        q = query_from_research_spec(spec)
        assert "momentum" in q.keyword_query
        assert "volatility" in q.keyword_query

    def test_objective_appears_in_keyword_query(self):
        spec = ResearchSpec(objective="improve oos sharpe ratio")
        q = query_from_research_spec(spec)
        assert "sharpe" in q.keyword_query

    def test_semantic_query_includes_objective(self):
        spec = ResearchSpec(objective="improve oos consistency")
        q = query_from_research_spec(spec)
        assert "improve oos consistency" in q.semantic_query

    def test_semantic_query_includes_feature_families_label(self):
        spec = ResearchSpec(
            model="RidgeRegressionModel",
            feature_families=["momentum"],
        )
        q = query_from_research_spec(spec)
        assert "momentum" in q.semantic_query

    def test_tags_match_feature_families(self):
        spec = ResearchSpec(feature_families=["momentum", "volatility"])
        q = query_from_research_spec(spec)
        assert q.tags == ["momentum", "volatility"]

    def test_source_is_research_spec(self):
        spec = ResearchSpec(model="RidgeRegressionModel")
        q = query_from_research_spec(spec)
        assert q.source == "ResearchSpec"

    def test_no_failure_modes_from_spec(self):
        spec = ResearchSpec(model="RidgeRegressionModel")
        q = query_from_research_spec(spec)
        assert q.failure_modes == []

    def test_no_experiment_names_from_spec(self):
        spec = ResearchSpec(model="RidgeRegressionModel")
        q = query_from_research_spec(spec)
        assert q.experiment_names == []

    def test_keyword_deduplication(self):
        spec = ResearchSpec(
            model="momentum",
            feature_families=["momentum"],
        )
        q = query_from_research_spec(spec)
        tokens = q.keyword_query.split()
        assert tokens.count("momentum") == 1

    def test_top_k_defaults(self):
        q = query_from_research_spec(ResearchSpec())
        assert q.top_k_keyword == 5
        assert q.top_k_semantic == 5

    def test_top_k_override(self):
        q = query_from_research_spec(ResearchSpec(), top_k_keyword=10, top_k_semantic=8)
        assert q.top_k_keyword == 10
        assert q.top_k_semantic == 8

    def test_empty_spec_semantic_query_is_none(self):
        spec = ResearchSpec()
        q = query_from_research_spec(spec)
        assert q.semantic_query is None


# ---------------------------------------------------------------------------
# C. query_from_resolved_spec
# ---------------------------------------------------------------------------


class TestQueryFromResolvedSpec:
    def test_model_in_keyword_query(self):
        resolved = _make_resolved(model="RidgeRegressionModel")
        q = query_from_resolved_spec(resolved)
        assert "ridgeregressionmodel" in q.keyword_query

    def test_feature_families_in_tags(self):
        resolved = _make_resolved(feature_families=["momentum", "volatility"])
        q = query_from_resolved_spec(resolved)
        assert q.tags == ["momentum", "volatility"]

    def test_source_is_resolved_spec(self):
        resolved = _make_resolved(model="RidgeRegressionModel")
        q = query_from_resolved_spec(resolved)
        assert q.source == "ResolvedResearchSpec"

    def test_empty_resolved_spec_none_keyword_query(self):
        resolved = _make_resolved()
        q = query_from_resolved_spec(resolved)
        assert q.keyword_query is None

    def test_no_failure_modes(self):
        resolved = _make_resolved(model="RidgeRegressionModel")
        q = query_from_resolved_spec(resolved)
        assert q.failure_modes == []

    def test_no_experiment_names(self):
        resolved = _make_resolved(model="RidgeRegressionModel")
        q = query_from_resolved_spec(resolved)
        assert q.experiment_names == []

    def test_objective_in_semantic_query(self):
        resolved = _make_resolved(
            model="RidgeRegressionModel", objective="reduce drawdown"
        )
        q = query_from_resolved_spec(resolved)
        assert "reduce drawdown" in q.semantic_query

    def test_label_in_keyword_query(self):
        resolved = _make_resolved(label="forward_return_1m", model="RidgeRegressionModel")
        q = query_from_resolved_spec(resolved)
        assert "forward" in q.keyword_query

    def test_features_contribute_to_keywords(self):
        resolved = _make_resolved(features=["rsi_14", "macd_signal"])
        q = query_from_resolved_spec(resolved)
        assert "rsi" in q.keyword_query
        assert "macd" in q.keyword_query


# ---------------------------------------------------------------------------
# D. query_from_experiment_plan
# ---------------------------------------------------------------------------


class TestQueryFromExperimentPlan:
    def test_baseline_in_experiment_names(self):
        plan = _make_plan(baseline="canonical_ml_showcase_v2")
        q = query_from_experiment_plan(plan)
        assert "canonical_ml_showcase_v2" in q.experiment_names

    def test_no_baseline_gives_empty_experiment_names(self):
        plan = _make_plan(baseline=None)
        q = query_from_experiment_plan(plan)
        assert q.experiment_names == []

    def test_failure_modes_to_monitor_extracted(self):
        plan = _make_plan(
            failure_modes_to_monitor=["poor_oos_consistency", "weak_ic"]
        )
        q = query_from_experiment_plan(plan)
        assert "poor_oos_consistency" in q.failure_modes
        assert "weak_ic" in q.failure_modes

    def test_empty_failure_modes_to_monitor(self):
        plan = _make_plan(failure_modes_to_monitor=[])
        q = query_from_experiment_plan(plan)
        assert q.failure_modes == []

    def test_research_objective_in_semantic_query(self):
        plan = _make_plan(research_objective="improve out-of-sample consistency")
        q = query_from_experiment_plan(plan)
        assert "improve out-of-sample consistency" in q.semantic_query

    def test_hypothesis_in_semantic_query(self):
        plan = _make_plan(hypothesis="lower alpha reduces overfitting")
        q = query_from_experiment_plan(plan)
        assert "lower alpha reduces overfitting" in q.semantic_query

    def test_source_is_experiment_plan(self):
        plan = _make_plan()
        q = query_from_experiment_plan(plan)
        assert q.source == "ExperimentPlan"

    def test_model_in_keyword_query(self):
        resolved = _make_resolved(model="RidgeRegressionModel")
        plan = _make_plan(resolved=resolved)
        q = query_from_experiment_plan(plan)
        assert "ridgeregressionmodel" in q.keyword_query

    def test_planned_changes_contribute_to_keywords(self):
        plan = _make_plan(planned_changes=["reduce regularisation alpha"])
        q = query_from_experiment_plan(plan)
        assert "regularisation" in q.keyword_query or "alpha" in q.keyword_query

    def test_feature_families_in_tags(self):
        resolved = _make_resolved(feature_families=["momentum"])
        plan = _make_plan(resolved=resolved)
        q = query_from_experiment_plan(plan)
        assert "momentum" in q.tags

    def test_top_k_override(self):
        plan = _make_plan()
        q = query_from_experiment_plan(plan, top_k_keyword=7, top_k_semantic=3)
        assert q.top_k_keyword == 7
        assert q.top_k_semantic == 3

    def test_empty_plan_none_keyword_query(self):
        plan = _make_plan(
            resolved=_make_resolved(),
            research_objective="",
            hypothesis="",
        )
        q = query_from_experiment_plan(plan)
        assert q.keyword_query is None
