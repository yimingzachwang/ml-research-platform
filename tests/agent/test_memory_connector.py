"""retrieve_research_memory_context unit tests — Phase 5A.

Covers:
  A. Keyword retrieval — basic happy path, deduplication
  B. Per-experiment retrieval — baseline experiment names
  C. Semantic retrieval — enabled/disabled, status handling
  D. Error handling — exception absorption, missing semantic query
  E. ResearchMemoryContext fields populated correctly
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from src.agent.memory.memory_connector import retrieve_research_memory_context
from src.agent.memory.memory_context import ResearchMemoryContext
from src.agent.memory.memory_evidence import ResearchMemoryEvidence
from src.agent.memory.memory_query import ResearchMemoryQuery

_MOD_KW = "src.agent.memory.memory_connector.retrieve_research_memory"
_MOD_SEM = "src.agent.memory.memory_connector.semantic_retrieve_research_memory"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _q(
    keyword_query: str | None = "ridge",
    semantic_query: str | None = "ridge regression",
    failure_modes: list[str] | None = None,
    experiment_names: list[str] | None = None,
    tags: list[str] | None = None,
    top_k_keyword: int = 5,
    top_k_semantic: int = 5,
) -> ResearchMemoryQuery:
    return ResearchMemoryQuery(
        keyword_query=keyword_query,
        semantic_query=semantic_query,
        failure_modes=failure_modes or [],
        experiment_names=experiment_names or [],
        tags=tags or [],
        top_k_keyword=top_k_keyword,
        top_k_semantic=top_k_semantic,
    )


def _kw_item(memory_id: str, experiment_name: str = "exp_v1") -> dict:
    return {
        "memory_id": memory_id,
        "experiment_name": experiment_name,
        "artefact_type": "experiment_metrics",
        "short_summary": f"item {memory_id}",
        "failure_modes": [],
        "tags": [],
        "path": f"/results/{experiment_name}/metrics.json",
        "context_hash": "hash_" + memory_id,
        "matched_terms": ["ridge"],
    }


def _sem_item(memory_id: str, experiment_name: str = "exp_v2") -> dict:
    return {
        "memory_id": memory_id,
        "experiment_name": experiment_name,
        "artefact_type": "llm_review",
        "short_summary": f"semantic item {memory_id}",
        "failure_modes": ["poor_oos_consistency"],
        "tags": ["momentum"],
        "path": f"/results/{experiment_name}/llm_review.json",
        "context_hash": "hash_" + memory_id,
        "score": 0.83,
    }


def _ok_semantic_result(items: list[dict]) -> dict:
    return {"status": "ok", "items": items, "query": "test", "embedding_model": "test-model"}


# ---------------------------------------------------------------------------
# A. Keyword retrieval
# ---------------------------------------------------------------------------


class TestKeywordRetrieval:
    def test_empty_memory_returns_empty_context(self):
        with patch(_MOD_KW, return_value=[]):
            ctx = retrieve_research_memory_context(_q())
        assert isinstance(ctx, ResearchMemoryContext)
        assert ctx.is_empty()

    def test_keyword_items_converted_to_evidence(self):
        with patch(_MOD_KW, return_value=[_kw_item("k1"), _kw_item("k2")]):
            ctx = retrieve_research_memory_context(_q())
        assert len(ctx.evidence_items) == 2

    def test_retrieval_method_is_keyword(self):
        with patch(_MOD_KW, return_value=[_kw_item("k1")]):
            ctx = retrieve_research_memory_context(_q())
        assert ctx.evidence_items[0].retrieval_method == "keyword"

    def test_keyword_count_correct(self):
        with patch(_MOD_KW, return_value=[_kw_item("k1"), _kw_item("k2")]):
            ctx = retrieve_research_memory_context(_q())
        assert ctx.keyword_count == 2

    def test_semantic_count_zero_when_disabled(self):
        with patch(_MOD_KW, return_value=[_kw_item("k1")]):
            ctx = retrieve_research_memory_context(_q(), enable_semantic=False)
        assert ctx.semantic_count == 0

    def test_semantic_not_called_when_disabled(self):
        with patch(_MOD_KW, return_value=[]):
            with patch(_MOD_SEM) as mock_sem:
                retrieve_research_memory_context(_q(), enable_semantic=False)
        mock_sem.assert_not_called()

    def test_deduplication_by_memory_id(self):
        item = _kw_item("k1")
        # General query and per-experiment lookup both return the same item
        with patch(_MOD_KW, return_value=[item]):
            ctx = retrieve_research_memory_context(
                _q(experiment_names=["exp_v1"])
            )
        assert ctx.evidence_items.count(ctx.evidence_items[0]) == 1

    def test_failure_modes_filter_passed_to_retriever(self):
        with patch(_MOD_KW, return_value=[]) as mock_kw:
            retrieve_research_memory_context(
                _q(failure_modes=["poor_oos_consistency"])
            )
        # First call (general query) should pass the failure modes filter
        first_call_kwargs = mock_kw.call_args_list[0][1]
        assert first_call_kwargs["failure_modes"] == ["poor_oos_consistency"]

    def test_no_failure_modes_passes_none(self):
        with patch(_MOD_KW, return_value=[]) as mock_kw:
            retrieve_research_memory_context(_q(failure_modes=[]))
        first_call_kwargs = mock_kw.call_args_list[0][1]
        assert first_call_kwargs["failure_modes"] is None


# ---------------------------------------------------------------------------
# B. Per-experiment retrieval for baseline experiments
# ---------------------------------------------------------------------------


class TestPerExperimentRetrieval:
    def test_experiment_names_trigger_extra_retrieval_call(self):
        with patch(_MOD_KW, return_value=[]) as mock_kw:
            retrieve_research_memory_context(
                _q(experiment_names=["canonical_v2"])
            )
        # General query + 1 per-experiment = 2 calls
        assert mock_kw.call_count == 2

    def test_per_experiment_call_uses_experiment_name(self):
        with patch(_MOD_KW, return_value=[]) as mock_kw:
            retrieve_research_memory_context(
                _q(experiment_names=["canonical_v2"])
            )
        per_exp_call_kwargs = mock_kw.call_args_list[1][1]
        assert per_exp_call_kwargs["experiment_name"] == "canonical_v2"

    def test_per_experiment_call_has_no_keyword_filter(self):
        with patch(_MOD_KW, return_value=[]) as mock_kw:
            retrieve_research_memory_context(
                _q(experiment_names=["canonical_v2"])
            )
        per_exp_call_kwargs = mock_kw.call_args_list[1][1]
        assert per_exp_call_kwargs["query"] is None

    def test_per_experiment_call_has_no_failure_mode_filter(self):
        with patch(_MOD_KW, return_value=[]) as mock_kw:
            retrieve_research_memory_context(
                _q(experiment_names=["canonical_v2"], failure_modes=["weak_ic"])
            )
        per_exp_call_kwargs = mock_kw.call_args_list[1][1]
        assert per_exp_call_kwargs["failure_modes"] is None

    def test_two_experiment_names_give_three_calls(self):
        with patch(_MOD_KW, return_value=[]) as mock_kw:
            retrieve_research_memory_context(
                _q(experiment_names=["exp_a", "exp_b"])
            )
        assert mock_kw.call_count == 3  # general + exp_a + exp_b

    def test_unique_items_from_per_experiment_added(self):
        general_item = _kw_item("k1", experiment_name="exp_v1")
        exp_item = _kw_item("k2", experiment_name="canonical_v2")

        def fake_retrieve(**kwargs):
            exp_name = kwargs.get("experiment_name")
            if exp_name == "canonical_v2":
                return [exp_item]
            return [general_item]

        with patch(_MOD_KW, side_effect=fake_retrieve):
            ctx = retrieve_research_memory_context(
                _q(experiment_names=["canonical_v2"])
            )
        assert len(ctx.evidence_items) == 2
        assert ctx.keyword_count == 2

    def test_no_experiment_names_gives_one_call(self):
        with patch(_MOD_KW, return_value=[]) as mock_kw:
            retrieve_research_memory_context(_q(experiment_names=[]))
        assert mock_kw.call_count == 1


# ---------------------------------------------------------------------------
# C. Semantic retrieval
# ---------------------------------------------------------------------------


class TestSemanticRetrieval:
    def test_semantic_items_added_when_enabled(self):
        sem_item = _sem_item("s1")
        with patch(_MOD_KW, return_value=[]):
            with patch(_MOD_SEM, return_value=_ok_semantic_result([sem_item])):
                ctx = retrieve_research_memory_context(_q(), enable_semantic=True)
        assert len(ctx.evidence_items) == 1
        assert ctx.evidence_items[0].retrieval_method == "semantic"

    def test_semantic_count_correct(self):
        with patch(_MOD_KW, return_value=[]):
            with patch(_MOD_SEM, return_value=_ok_semantic_result([_sem_item("s1")])):
                ctx = retrieve_research_memory_context(_q(), enable_semantic=True)
        assert ctx.semantic_count == 1

    def test_semantic_not_added_if_already_in_keyword(self):
        item = _kw_item("k1")
        sem_dup = {**_sem_item("k1"), "memory_id": "k1"}  # same ID as keyword item
        with patch(_MOD_KW, return_value=[item]):
            with patch(_MOD_SEM, return_value=_ok_semantic_result([sem_dup])):
                ctx = retrieve_research_memory_context(_q(), enable_semantic=True)
        # keyword item takes precedence; duplicate not added
        assert len(ctx.evidence_items) == 1
        assert ctx.evidence_items[0].retrieval_method == "keyword"
        assert ctx.semantic_count == 0

    def test_no_semantic_index_adds_warning(self):
        with patch(_MOD_KW, return_value=[]):
            with patch(
                _MOD_SEM,
                return_value={"status": "no_semantic_index", "items": []},
            ):
                ctx = retrieve_research_memory_context(_q(), enable_semantic=True)
        assert any("Semantic index not available" in w for w in ctx.warnings)

    def test_no_phase1_index_adds_warning(self):
        with patch(_MOD_KW, return_value=[]):
            with patch(
                _MOD_SEM,
                return_value={"status": "no_phase1_index", "items": []},
            ):
                ctx = retrieve_research_memory_context(_q(), enable_semantic=True)
        assert any("Memory index not built" in w for w in ctx.warnings)

    def test_embedding_failed_adds_warning_with_error(self):
        with patch(_MOD_KW, return_value=[]):
            with patch(
                _MOD_SEM,
                return_value={"status": "embedding_failed", "error": "connection refused", "items": []},
            ):
                ctx = retrieve_research_memory_context(_q(), enable_semantic=True)
        assert any("connection refused" in w for w in ctx.warnings)

    def test_unexpected_status_adds_warning(self):
        with patch(_MOD_KW, return_value=[]):
            with patch(
                _MOD_SEM,
                return_value={"status": "unknown_error", "items": []},
            ):
                ctx = retrieve_research_memory_context(_q(), enable_semantic=True)
        assert any("unexpected status" in w for w in ctx.warnings)

    def test_no_warnings_on_success(self):
        with patch(_MOD_KW, return_value=[]):
            with patch(_MOD_SEM, return_value=_ok_semantic_result([])):
                ctx = retrieve_research_memory_context(_q(), enable_semantic=True)
        assert ctx.warnings == []


# ---------------------------------------------------------------------------
# D. Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_keyword_exception_returns_empty_context_no_raise(self):
        with patch(_MOD_KW, side_effect=RuntimeError("disk full")):
            ctx = retrieve_research_memory_context(_q())
        assert isinstance(ctx, ResearchMemoryContext)
        assert ctx.is_empty()

    def test_semantic_exception_absorbed_as_warning(self):
        with patch(_MOD_KW, return_value=[]):
            with patch(_MOD_SEM, side_effect=OSError("network error")):
                ctx = retrieve_research_memory_context(_q(), enable_semantic=True)
        assert any("network error" in w for w in ctx.warnings)

    def test_none_semantic_query_adds_warning_when_semantic_enabled(self):
        q = _q(semantic_query=None)
        with patch(_MOD_KW, return_value=[]):
            with patch(_MOD_SEM) as mock_sem:
                ctx = retrieve_research_memory_context(q, enable_semantic=True)
        mock_sem.assert_not_called()
        assert any("no semantic_query" in w for w in ctx.warnings)

    def test_items_with_empty_memory_id_are_skipped(self):
        item_no_id = {**_kw_item("k1"), "memory_id": ""}
        item_valid = _kw_item("k2")
        with patch(_MOD_KW, return_value=[item_no_id, item_valid]):
            ctx = retrieve_research_memory_context(_q())
        assert ctx.keyword_count == 1
        assert ctx.evidence_items[0].memory_id == "k2"


# ---------------------------------------------------------------------------
# E. Context fields populated correctly
# ---------------------------------------------------------------------------


class TestContextFieldsPopulated:
    def test_query_stored_on_context(self):
        q = _q(keyword_query="test query")
        with patch(_MOD_KW, return_value=[]):
            ctx = retrieve_research_memory_context(q)
        assert ctx.query is q

    def test_context_warnings_empty_on_clean_run(self):
        with patch(_MOD_KW, return_value=[]):
            ctx = retrieve_research_memory_context(_q(), enable_semantic=False)
        assert ctx.warnings == []

    def test_to_dict_returns_dict(self):
        with patch(_MOD_KW, return_value=[_kw_item("k1")]):
            ctx = retrieve_research_memory_context(_q())
        d = ctx.to_dict()
        assert isinstance(d, dict)
        assert "evidence_items" in d

    def test_combined_keyword_and_semantic_counts(self):
        kw_item = _kw_item("k1")
        sem = _sem_item("s1")
        with patch(_MOD_KW, return_value=[kw_item]):
            with patch(_MOD_SEM, return_value=_ok_semantic_result([sem])):
                ctx = retrieve_research_memory_context(_q(), enable_semantic=True)
        assert ctx.keyword_count == 1
        assert ctx.semantic_count == 1
        assert len(ctx.evidence_items) == 2

    def test_to_agent_summary_callable(self):
        with patch(_MOD_KW, return_value=[_kw_item("k1")]):
            ctx = retrieve_research_memory_context(_q())
        summary = ctx.to_agent_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
