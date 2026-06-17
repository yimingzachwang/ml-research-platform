"""ResearchMemoryQuery — structured retrieval parameters built from agent objects.

Query builders convert ResearchSpec / ResolvedResearchSpec / ExperimentPlan
into keyword and semantic retrieval parameters for the existing orchestration
memory APIs.

No LLM calls. No network calls. All extraction is deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agent.intent.research_spec import ResearchSpec
    from src.agent.planner.experiment_plan import ExperimentPlan
    from src.agent.resolver.resolved_spec import ResolvedResearchSpec

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _collect_keywords(*sources: str | list[str] | None) -> list[str]:
    """Deduplicated ordered keyword list from heterogeneous sources."""
    seen: set[str] = set()
    result: list[str] = []
    for src in sources:
        if isinstance(src, list):
            tokens: list[str] = []
            for item in src:
                if item:
                    tokens.extend(_tokenize(item))
        elif src:
            tokens = _tokenize(src)
        else:
            continue
        for t in tokens:
            if t and t not in seen:
                seen.add(t)
                result.append(t)
    return result


@dataclass
class ResearchMemoryQuery:
    """Structured retrieval parameters for the existing memory APIs.

    Construct via the query_from_* builder functions; do not build manually
    unless writing tests.
    """

    keyword_query: str | None
    semantic_query: str | None
    failure_modes: list[str] = field(default_factory=list)
    experiment_names: list[str] = field(default_factory=list)
    artefact_types: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    top_k_keyword: int = 5
    top_k_semantic: int = 5
    source: str = "manual"

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyword_query": self.keyword_query,
            "semantic_query": self.semantic_query,
            "failure_modes": list(self.failure_modes),
            "experiment_names": list(self.experiment_names),
            "artefact_types": list(self.artefact_types),
            "tags": list(self.tags),
            "top_k_keyword": self.top_k_keyword,
            "top_k_semantic": self.top_k_semantic,
            "source": self.source,
        }


def query_from_research_spec(
    spec: ResearchSpec,
    top_k_keyword: int = 5,
    top_k_semantic: int = 5,
) -> ResearchMemoryQuery:
    """Build a ResearchMemoryQuery from a partially-specified ResearchSpec."""
    kw_parts = _collect_keywords(
        spec.model,
        spec.label,
        spec.signal,
        spec.objective,
        spec.notes,
        spec.feature_families,
        spec.features,
    )
    keyword_query = " ".join(kw_parts) if kw_parts else None

    sem_parts: list[str] = []
    if spec.objective:
        sem_parts.append(spec.objective)
    if spec.model:
        sem_parts.append(spec.model)
    if spec.feature_families:
        sem_parts.append("feature families: " + ", ".join(spec.feature_families))
    if spec.notes:
        sem_parts.append(spec.notes)
    semantic_query = ". ".join(sem_parts) if sem_parts else keyword_query

    tags = list(spec.feature_families or [])

    return ResearchMemoryQuery(
        keyword_query=keyword_query,
        semantic_query=semantic_query,
        failure_modes=[],
        experiment_names=[],
        tags=tags,
        top_k_keyword=top_k_keyword,
        top_k_semantic=top_k_semantic,
        source="ResearchSpec",
    )


def query_from_resolved_spec(
    resolved: ResolvedResearchSpec,
    top_k_keyword: int = 5,
    top_k_semantic: int = 5,
) -> ResearchMemoryQuery:
    """Build a ResearchMemoryQuery from a ResolvedResearchSpec."""
    kw_parts = _collect_keywords(
        resolved.model,
        resolved.label,
        resolved.signal,
        resolved.objective,
        resolved.notes,
        resolved.feature_families,
        resolved.features,
    )
    keyword_query = " ".join(kw_parts) if kw_parts else None

    sem_parts: list[str] = []
    if resolved.objective:
        sem_parts.append(resolved.objective)
    if resolved.model:
        sem_parts.append(resolved.model)
    if resolved.feature_families:
        sem_parts.append("feature families: " + ", ".join(resolved.feature_families))
    if resolved.notes:
        sem_parts.append(resolved.notes)
    semantic_query = ". ".join(sem_parts) if sem_parts else keyword_query

    tags = list(resolved.feature_families)

    return ResearchMemoryQuery(
        keyword_query=keyword_query,
        semantic_query=semantic_query,
        failure_modes=[],
        experiment_names=[],
        tags=tags,
        top_k_keyword=top_k_keyword,
        top_k_semantic=top_k_semantic,
        source="ResolvedResearchSpec",
    )


def query_from_experiment_plan(
    plan: ExperimentPlan,
    top_k_keyword: int = 5,
    top_k_semantic: int = 5,
) -> ResearchMemoryQuery:
    """Build a ResearchMemoryQuery from an ExperimentPlan.

    Incorporates failure_modes_to_monitor and baseline experiment name so the
    connector can retrieve targeted evidence for the plan's comparison context.
    """
    spec = plan.resolved_spec

    kw_parts = _collect_keywords(
        spec.model,
        spec.label,
        spec.signal,
        plan.research_objective,
        spec.notes,
        spec.feature_families,
        plan.planned_changes,
    )
    keyword_query = " ".join(kw_parts) if kw_parts else None

    sem_parts: list[str] = []
    if plan.research_objective:
        sem_parts.append(plan.research_objective)
    if plan.hypothesis:
        sem_parts.append(plan.hypothesis)
    if spec.model:
        sem_parts.append(spec.model)
    if spec.feature_families:
        sem_parts.append("feature families: " + ", ".join(spec.feature_families))
    semantic_query = ". ".join(sem_parts) if sem_parts else keyword_query

    experiment_names: list[str] = [plan.baseline] if plan.baseline else []
    tags = list(spec.feature_families)

    return ResearchMemoryQuery(
        keyword_query=keyword_query,
        semantic_query=semantic_query,
        failure_modes=list(plan.failure_modes_to_monitor),
        experiment_names=experiment_names,
        tags=tags,
        top_k_keyword=top_k_keyword,
        top_k_semantic=top_k_semantic,
        source="ExperimentPlan",
    )
