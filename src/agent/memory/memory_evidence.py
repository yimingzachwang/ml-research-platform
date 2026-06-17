"""ResearchMemoryEvidence — one retrieved memory item.

Wraps the raw dict returned by retrieve_research_memory() (keyword) or
semantic_retrieve_research_memory() (semantic) into a typed, serialisable
evidence object.

Keyword retrieval does not expose a relevance score; relevance_score is 0.0
for keyword items.  Semantic retrieval exposes cosine similarity as score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResearchMemoryEvidence:
    """One retrieved memory item tagged with its retrieval method and score."""

    memory_id: str
    experiment_name: str
    artefact_type: str
    short_summary: str
    failure_modes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    path: str = ""
    context_hash: str = ""
    retrieval_method: str = "keyword"   # "keyword" | "semantic"
    relevance_score: float = 0.0        # 0.0 for keyword; cosine similarity for semantic
    matched_terms: list[str] = field(default_factory=list)  # populated for keyword only

    @classmethod
    def from_keyword_item(cls, item: dict[str, Any]) -> ResearchMemoryEvidence:
        """Build from a keyword retrieval result dict.

        Keyword retrieval does not expose a numeric relevance score, so
        relevance_score is always 0.0 for keyword items.
        """
        return cls(
            memory_id=item.get("memory_id", ""),
            experiment_name=item.get("experiment_name", ""),
            artefact_type=item.get("artefact_type", ""),
            short_summary=item.get("short_summary", ""),
            failure_modes=list(item.get("failure_modes") or []),
            tags=list(item.get("tags") or []),
            path=item.get("path", ""),
            context_hash=item.get("context_hash", ""),
            retrieval_method="keyword",
            relevance_score=0.0,
            matched_terms=list(item.get("matched_terms") or []),
        )

    @classmethod
    def from_semantic_item(cls, item: dict[str, Any]) -> ResearchMemoryEvidence:
        """Build from a semantic retrieval result dict.

        Cosine similarity is stored in item["score"] (0.0–1.0, 3 dp).
        """
        return cls(
            memory_id=item.get("memory_id", ""),
            experiment_name=item.get("experiment_name", ""),
            artefact_type=item.get("artefact_type", ""),
            short_summary=item.get("short_summary", ""),
            failure_modes=list(item.get("failure_modes") or []),
            tags=list(item.get("tags") or []),
            path=item.get("path", ""),
            context_hash=item.get("context_hash", ""),
            retrieval_method="semantic",
            relevance_score=float(item.get("score", 0.0)),
            matched_terms=[],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "experiment_name": self.experiment_name,
            "artefact_type": self.artefact_type,
            "short_summary": self.short_summary,
            "failure_modes": list(self.failure_modes),
            "tags": list(self.tags),
            "path": self.path,
            "context_hash": self.context_hash,
            "retrieval_method": self.retrieval_method,
            "relevance_score": self.relevance_score,
            "matched_terms": list(self.matched_terms),
        }
