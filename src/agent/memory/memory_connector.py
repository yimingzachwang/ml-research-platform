"""Research Memory Evidence Connector.

Connects the existing orchestration memory system to the src/agent/ layer as
an evidence source.  All reads go through the existing
src/orchestration/api/research_api APIs — no new storage is introduced.

Governance invariants:
  - Does not execute experiments.
  - Does not approve drafts.
  - Does not create drafts from memory alone.
  - Does not let semantic retrieval override deterministic validation.
  - Memory evidence informs planning suggestions; it does not trigger actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agent.memory.memory_context import ResearchMemoryContext
from src.agent.memory.memory_evidence import ResearchMemoryEvidence
from src.agent.memory.memory_query import ResearchMemoryQuery
from src.orchestration.api.research_api import (
    retrieve_research_memory,
    semantic_retrieve_research_memory,
)


def retrieve_research_memory_context(
    query: ResearchMemoryQuery,
    memory_base: Path | str | None = None,
    enable_semantic: bool = False,
    provider: str = "openai",
    model: str | None = None,
    base_url: str | None = None,
) -> ResearchMemoryContext:
    """Retrieve and assemble a ResearchMemoryContext from existing memory APIs.

    Phase 1 (keyword): always runs.
      - General keyword query against the full index (respects failure_modes filter).
      - Per-experiment lookup for each name in query.experiment_names (no keyword
        filter — retrieves all artefact types for the baseline).

    Phase 2 (semantic): runs only when enable_semantic=True.
      - Single semantic query; unique results (not already in keyword set) are added.

    Results are deduplicated by memory_id; keyword items take precedence over
    semantic items for the same memory_id.
    """
    warnings: list[str] = []
    seen_ids: dict[str, ResearchMemoryEvidence] = {}
    keyword_count = 0
    semantic_count = 0

    # ------------------------------------------------------------------
    # Phase 1a: general keyword query
    # ------------------------------------------------------------------
    fm_filter = query.failure_modes if query.failure_modes else None
    general_items = _keyword_retrieve(
        query_str=query.keyword_query,
        experiment_name=None,
        failure_modes=fm_filter,
        top_k=query.top_k_keyword,
        memory_base=memory_base,
    )
    for item in general_items:
        mid = item.get("memory_id", "")
        if mid and mid not in seen_ids:
            seen_ids[mid] = ResearchMemoryEvidence.from_keyword_item(item)
            keyword_count += 1

    # ------------------------------------------------------------------
    # Phase 1b: per-experiment lookups for baseline experiments
    # Retrieves all artefact types for the named experiment without a
    # keyword filter — ensures reviews, metrics, and comparisons are found
    # even when the keyword query would not score them highly.
    # ------------------------------------------------------------------
    for exp_name in query.experiment_names:
        exp_items = _keyword_retrieve(
            query_str=None,
            experiment_name=exp_name,
            failure_modes=None,
            top_k=query.top_k_keyword,
            memory_base=memory_base,
        )
        for item in exp_items:
            mid = item.get("memory_id", "")
            if mid and mid not in seen_ids:
                seen_ids[mid] = ResearchMemoryEvidence.from_keyword_item(item)
                keyword_count += 1

    # ------------------------------------------------------------------
    # Phase 2: optional semantic retrieval
    # ------------------------------------------------------------------
    if enable_semantic:
        if query.semantic_query:
            sem_result = _semantic_retrieve(
                query_str=query.semantic_query,
                top_k=query.top_k_semantic,
                failure_modes=query.failure_modes or None,
                tags=query.tags or None,
                provider=provider,
                model=model,
                base_url=base_url,
                memory_base=memory_base,
            )
            status = sem_result.get("status", "")
            if status == "ok":
                for item in sem_result.get("items", []):
                    mid = item.get("memory_id", "")
                    if mid and mid not in seen_ids:
                        seen_ids[mid] = ResearchMemoryEvidence.from_semantic_item(item)
                        semantic_count += 1
            elif status == "no_semantic_index":
                warnings.append(
                    "Semantic index not available. Keyword retrieval only."
                )
            elif status == "no_phase1_index":
                warnings.append(
                    "Memory index not built. Run index_research_memory() first."
                )
            elif status == "embedding_failed":
                err = sem_result.get("error", "unknown error")
                warnings.append(f"Semantic embedding failed: {err}")
            else:
                warnings.append(
                    f"Semantic retrieval returned unexpected status: {status!r}"
                )
        else:
            warnings.append(
                "Semantic retrieval requested but no semantic_query was built "
                "from the spec."
            )

    return ResearchMemoryContext(
        query=query,
        evidence_items=list(seen_ids.values()),
        keyword_count=keyword_count,
        semantic_count=semantic_count,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Private helpers — thin wrappers that absorb exceptions so the connector
# never raises on a missing or malformed index.
# ---------------------------------------------------------------------------


def _keyword_retrieve(
    query_str: str | None,
    experiment_name: str | None,
    failure_modes: list[str] | None,
    top_k: int,
    memory_base: Any,
) -> list[dict[str, Any]]:
    try:
        return retrieve_research_memory(
            query=query_str or None,
            experiment_name=experiment_name or None,
            failure_modes=failure_modes or None,
            artefact_type=None,
            top_k=top_k,
            memory_base=memory_base,
        )
    except Exception:
        return []


def _semantic_retrieve(
    query_str: str,
    top_k: int,
    failure_modes: list[str] | None,
    tags: list[str] | None,
    provider: str,
    model: str | None,
    base_url: str | None,
    memory_base: Any,
) -> dict[str, Any]:
    try:
        return semantic_retrieve_research_memory(
            query=query_str,
            top_k=top_k,
            failure_modes=failure_modes or None,
            tags=tags or None,
            provider=provider,
            model=model,
            base_url=base_url,
            memory_base=memory_base,
        )
    except Exception as exc:
        return {"status": "embedding_failed", "error": str(exc), "items": []}
