"""ResearchMemoryContext — aggregated evidence from memory retrieval.

Produced by retrieve_research_memory_context() in memory_connector.py.
Used by future Phase 5B next-action planner as an evidence source.

No LLM calls. All methods are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.memory.memory_evidence import ResearchMemoryEvidence
from src.agent.memory.memory_query import ResearchMemoryQuery


@dataclass
class ResearchMemoryContext:
    """Aggregated evidence from keyword and/or semantic memory retrieval."""

    query: ResearchMemoryQuery
    evidence_items: list[ResearchMemoryEvidence] = field(default_factory=list)
    keyword_count: int = 0
    semantic_count: int = 0
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived properties (computed, not stored)
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        return len(self.evidence_items) == 0

    @property
    def failure_mode_summary(self) -> dict[str, int]:
        """Failure mode name → occurrence count, sorted by count descending."""
        counts: dict[str, int] = {}
        for item in self.evidence_items:
            for fm in item.failure_modes:
                counts[fm] = counts.get(fm, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    @property
    def experiment_summary(self) -> list[str]:
        """Distinct experiment names in evidence, in order of first appearance."""
        seen: set[str] = set()
        result: list[str] = []
        for item in self.evidence_items:
            name = item.experiment_name
            if name and name not in seen:
                seen.add(name)
                result.append(name)
        return result

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "evidence_items": [e.to_dict() for e in self.evidence_items],
            "keyword_count": self.keyword_count,
            "semantic_count": self.semantic_count,
            "failure_mode_summary": self.failure_mode_summary,
            "experiment_summary": self.experiment_summary,
            "warnings": list(self.warnings),
        }

    # ------------------------------------------------------------------
    # Plain-text summary — deterministic, no LLM
    # ------------------------------------------------------------------

    def to_agent_summary(self) -> str:
        """Return a concise plain-text summary of the retrieved evidence.

        Deterministic.  No LLM.  Suitable for inclusion in an agent prompt
        or for display in a research tool response.
        """
        total = len(self.evidence_items)
        kw = self.keyword_count
        sem = self.semantic_count
        src = self.query.source

        lines: list[str] = [
            f"Research Memory Evidence ({total} items, source: {src})",
        ]

        if kw > 0 and sem > 0:
            lines.append(f"Retrieval: {kw} keyword + {sem} semantic")
        elif kw > 0:
            lines.append(f"Retrieval: {kw} keyword")
        elif sem > 0:
            lines.append(f"Retrieval: {sem} semantic")

        exp = self.experiment_summary
        if exp:
            lines.append("Experiments: " + ", ".join(exp))

        fm = self.failure_mode_summary
        if fm:
            fm_parts = [f"{k} ({v}x)" for k, v in fm.items()]
            lines.append("Failure modes: " + ", ".join(fm_parts))

        if total > 0:
            lines.append("")
            for i, item in enumerate(self.evidence_items, 1):
                method = item.retrieval_method
                score_str = (
                    f", score={item.relevance_score:.3f}"
                    if item.retrieval_method == "semantic" and item.relevance_score > 0
                    else ""
                )
                header = (
                    f"{i}. [{item.artefact_type}] {item.experiment_name}"
                    f" ({method}{score_str})"
                )
                lines.append(header)
                if item.short_summary:
                    lines.append(f"   {item.short_summary[:120]}")
                if item.failure_modes:
                    lines.append("   Failure modes: " + ", ".join(item.failure_modes))
                if item.matched_terms:
                    lines.append("   Matched: " + ", ".join(item.matched_terms[:8]))

        lines.append("")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")
        else:
            lines.append("Warnings: none")

        return "\n".join(lines)
