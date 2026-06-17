"""ResearchMemoryEvidence and ResearchMemoryContext unit tests — Phase 5A.

Covers:
  A. ResearchMemoryEvidence construction from keyword/semantic dicts
  B. ResearchMemoryContext empty state
  C. ResearchMemoryContext with evidence items — derived properties
  D. ResearchMemoryContext.to_agent_summary() — deterministic text output
"""

from __future__ import annotations

import pytest

from src.agent.memory.memory_context import ResearchMemoryContext
from src.agent.memory.memory_evidence import ResearchMemoryEvidence
from src.agent.memory.memory_query import ResearchMemoryQuery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_query(source: str = "ExperimentPlan") -> ResearchMemoryQuery:
    return ResearchMemoryQuery(
        keyword_query="ridge momentum",
        semantic_query="ridge regression with momentum features",
        source=source,
    )


def _make_kw_item(
    memory_id: str = "abc123",
    experiment_name: str = "exp_v1",
    artefact_type: str = "experiment_metrics",
    short_summary: str = "OOS Sharpe 0.42",
    failure_modes: list[str] | None = None,
    tags: list[str] | None = None,
    matched_terms: list[str] | None = None,
) -> dict:
    return {
        "memory_id": memory_id,
        "experiment_name": experiment_name,
        "artefact_type": artefact_type,
        "short_summary": short_summary,
        "failure_modes": failure_modes or [],
        "tags": tags or [],
        "path": f"/results/experiments/{experiment_name}/metrics.json",
        "context_hash": "hash_" + memory_id,
        "matched_terms": matched_terms or ["ridge", "momentum"],
    }


def _make_sem_item(
    memory_id: str = "sem001",
    experiment_name: str = "exp_v2",
    artefact_type: str = "llm_review",
    short_summary: str = "Critical: poor OOS consistency",
    failure_modes: list[str] | None = None,
    score: float = 0.87,
) -> dict:
    return {
        "memory_id": memory_id,
        "experiment_name": experiment_name,
        "artefact_type": artefact_type,
        "short_summary": short_summary,
        "failure_modes": failure_modes if failure_modes is not None else ["poor_oos_consistency"],
        "tags": ["momentum"],
        "path": f"/results/llm_reviews/{experiment_name}/llm_review.json",
        "context_hash": "hash_" + memory_id,
        "score": score,
    }


def _make_evidence_keyword(**kwargs) -> ResearchMemoryEvidence:
    return ResearchMemoryEvidence.from_keyword_item(_make_kw_item(**kwargs))


def _make_evidence_semantic(**kwargs) -> ResearchMemoryEvidence:
    return ResearchMemoryEvidence.from_semantic_item(_make_sem_item(**kwargs))


# ---------------------------------------------------------------------------
# A. ResearchMemoryEvidence
# ---------------------------------------------------------------------------


class TestResearchMemoryEvidenceFromKeyword:
    def test_memory_id_populated(self):
        ev = ResearchMemoryEvidence.from_keyword_item(_make_kw_item(memory_id="id1"))
        assert ev.memory_id == "id1"

    def test_experiment_name_populated(self):
        ev = ResearchMemoryEvidence.from_keyword_item(
            _make_kw_item(experiment_name="exp_v3")
        )
        assert ev.experiment_name == "exp_v3"

    def test_artefact_type_populated(self):
        ev = ResearchMemoryEvidence.from_keyword_item(
            _make_kw_item(artefact_type="llm_review")
        )
        assert ev.artefact_type == "llm_review"

    def test_retrieval_method_is_keyword(self):
        ev = ResearchMemoryEvidence.from_keyword_item(_make_kw_item())
        assert ev.retrieval_method == "keyword"

    def test_relevance_score_is_zero_for_keyword(self):
        ev = ResearchMemoryEvidence.from_keyword_item(_make_kw_item())
        assert ev.relevance_score == 0.0

    def test_matched_terms_populated(self):
        ev = ResearchMemoryEvidence.from_keyword_item(
            _make_kw_item(matched_terms=["ridge", "sharpe"])
        )
        assert ev.matched_terms == ["ridge", "sharpe"]

    def test_failure_modes_populated(self):
        ev = ResearchMemoryEvidence.from_keyword_item(
            _make_kw_item(failure_modes=["weak_ic"])
        )
        assert ev.failure_modes == ["weak_ic"]

    def test_to_dict_has_all_keys(self):
        ev = ResearchMemoryEvidence.from_keyword_item(_make_kw_item())
        d = ev.to_dict()
        expected = {
            "memory_id",
            "experiment_name",
            "artefact_type",
            "short_summary",
            "failure_modes",
            "tags",
            "path",
            "context_hash",
            "retrieval_method",
            "relevance_score",
            "matched_terms",
        }
        assert expected == set(d.keys())

    def test_none_failure_modes_in_item_gives_empty_list(self):
        item = _make_kw_item()
        item["failure_modes"] = None
        ev = ResearchMemoryEvidence.from_keyword_item(item)
        assert ev.failure_modes == []


class TestResearchMemoryEvidenceFromSemantic:
    def test_retrieval_method_is_semantic(self):
        ev = ResearchMemoryEvidence.from_semantic_item(_make_sem_item())
        assert ev.retrieval_method == "semantic"

    def test_relevance_score_from_item(self):
        ev = ResearchMemoryEvidence.from_semantic_item(_make_sem_item(score=0.753))
        assert ev.relevance_score == pytest.approx(0.753)

    def test_matched_terms_empty_for_semantic(self):
        ev = ResearchMemoryEvidence.from_semantic_item(_make_sem_item())
        assert ev.matched_terms == []

    def test_failure_modes_populated(self):
        ev = ResearchMemoryEvidence.from_semantic_item(
            _make_sem_item(failure_modes=["poor_oos_consistency"])
        )
        assert "poor_oos_consistency" in ev.failure_modes

    def test_to_dict_retrieval_method_semantic(self):
        ev = ResearchMemoryEvidence.from_semantic_item(_make_sem_item())
        assert ev.to_dict()["retrieval_method"] == "semantic"


# ---------------------------------------------------------------------------
# B. ResearchMemoryContext — empty state
# ---------------------------------------------------------------------------


class TestResearchMemoryContextEmpty:
    def test_is_empty_true(self):
        ctx = ResearchMemoryContext(query=_make_query())
        assert ctx.is_empty()

    def test_failure_mode_summary_empty(self):
        ctx = ResearchMemoryContext(query=_make_query())
        assert ctx.failure_mode_summary == {}

    def test_experiment_summary_empty(self):
        ctx = ResearchMemoryContext(query=_make_query())
        assert ctx.experiment_summary == []

    def test_to_dict_keys(self):
        ctx = ResearchMemoryContext(query=_make_query())
        d = ctx.to_dict()
        expected = {
            "query",
            "evidence_items",
            "keyword_count",
            "semantic_count",
            "failure_mode_summary",
            "experiment_summary",
            "warnings",
        }
        assert expected == set(d.keys())

    def test_to_dict_empty_evidence_items(self):
        ctx = ResearchMemoryContext(query=_make_query())
        assert ctx.to_dict()["evidence_items"] == []


# ---------------------------------------------------------------------------
# C. ResearchMemoryContext with items — derived properties
# ---------------------------------------------------------------------------


class TestResearchMemoryContextWithItems:
    def _ctx(self) -> ResearchMemoryContext:
        ev1 = _make_evidence_keyword(
            memory_id="k1",
            experiment_name="exp_v1",
            failure_modes=["poor_oos_consistency", "weak_ic"],
        )
        ev2 = _make_evidence_keyword(
            memory_id="k2",
            experiment_name="exp_v1",
            failure_modes=["poor_oos_consistency"],
        )
        ev3 = _make_evidence_semantic(
            memory_id="s1",
            experiment_name="exp_v2",
            failure_modes=[],
        )
        return ResearchMemoryContext(
            query=_make_query(),
            evidence_items=[ev1, ev2, ev3],
            keyword_count=2,
            semantic_count=1,
        )

    def test_is_empty_false(self):
        assert not self._ctx().is_empty()

    def test_failure_mode_summary_counts(self):
        fm = self._ctx().failure_mode_summary
        assert fm["poor_oos_consistency"] == 2
        assert fm["weak_ic"] == 1

    def test_failure_mode_summary_sorted_by_count_descending(self):
        fm = self._ctx().failure_mode_summary
        counts = list(fm.values())
        assert counts == sorted(counts, reverse=True)

    def test_experiment_summary_deduped(self):
        exp = self._ctx().experiment_summary
        assert exp.count("exp_v1") == 1

    def test_experiment_summary_order_of_first_appearance(self):
        exp = self._ctx().experiment_summary
        assert exp[0] == "exp_v1"
        assert exp[1] == "exp_v2"

    def test_keyword_count_correct(self):
        assert self._ctx().keyword_count == 2

    def test_semantic_count_correct(self):
        assert self._ctx().semantic_count == 1

    def test_to_dict_evidence_items_count(self):
        d = self._ctx().to_dict()
        assert len(d["evidence_items"]) == 3


# ---------------------------------------------------------------------------
# D. ResearchMemoryContext.to_agent_summary()
# ---------------------------------------------------------------------------


class TestToAgentSummary:
    def test_summary_contains_item_count(self):
        ctx = ResearchMemoryContext(
            query=_make_query(),
            evidence_items=[_make_evidence_keyword()],
            keyword_count=1,
        )
        summary = ctx.to_agent_summary()
        assert "1 item" in summary

    def test_summary_contains_source(self):
        ctx = ResearchMemoryContext(
            query=_make_query(source="ExperimentPlan"),
            evidence_items=[],
        )
        summary = ctx.to_agent_summary()
        assert "ExperimentPlan" in summary

    def test_summary_contains_failure_modes(self):
        ev = _make_evidence_keyword(failure_modes=["poor_oos_consistency"])
        ctx = ResearchMemoryContext(
            query=_make_query(),
            evidence_items=[ev],
            keyword_count=1,
        )
        summary = ctx.to_agent_summary()
        assert "poor_oos_consistency" in summary

    def test_summary_contains_experiment_name(self):
        ev = _make_evidence_keyword(experiment_name="canonical_v2")
        ctx = ResearchMemoryContext(
            query=_make_query(),
            evidence_items=[ev],
            keyword_count=1,
        )
        summary = ctx.to_agent_summary()
        assert "canonical_v2" in summary

    def test_summary_warnings_none_label(self):
        ctx = ResearchMemoryContext(query=_make_query(), warnings=[])
        summary = ctx.to_agent_summary()
        assert "Warnings: none" in summary

    def test_summary_warnings_included(self):
        ctx = ResearchMemoryContext(
            query=_make_query(),
            warnings=["Semantic index not available."],
        )
        summary = ctx.to_agent_summary()
        assert "Semantic index not available." in summary

    def test_summary_retrieval_counts(self):
        ev_kw = _make_evidence_keyword()
        ev_sem = _make_evidence_semantic()
        ctx = ResearchMemoryContext(
            query=_make_query(),
            evidence_items=[ev_kw, ev_sem],
            keyword_count=1,
            semantic_count=1,
        )
        summary = ctx.to_agent_summary()
        assert "keyword" in summary
        assert "semantic" in summary

    def test_summary_is_string(self):
        ctx = ResearchMemoryContext(query=_make_query())
        assert isinstance(ctx.to_agent_summary(), str)

    def test_summary_artefact_type_in_item_line(self):
        ev = _make_evidence_keyword(artefact_type="experiment_metrics")
        ctx = ResearchMemoryContext(
            query=_make_query(),
            evidence_items=[ev],
            keyword_count=1,
        )
        summary = ctx.to_agent_summary()
        assert "experiment_metrics" in summary

    def test_summary_semantic_score_shown(self):
        ev = _make_evidence_semantic(score=0.823)
        ctx = ResearchMemoryContext(
            query=_make_query(),
            evidence_items=[ev],
            semantic_count=1,
        )
        summary = ctx.to_agent_summary()
        assert "0.823" in summary

    def test_empty_context_summary(self):
        ctx = ResearchMemoryContext(query=_make_query())
        summary = ctx.to_agent_summary()
        assert "0 items" in summary
        assert "Warnings: none" in summary
