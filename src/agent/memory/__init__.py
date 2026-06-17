"""Research Memory Evidence Connector — Phase 5A.

Connects the existing src/orchestration/ memory system to src/agent/ as an
evidence source.  Does not implement new storage or retrieval logic.

Public API:

    ResearchMemoryQuery         structured retrieval parameters
    ResearchMemoryEvidence      one retrieved evidence item
    ResearchMemoryContext       aggregated context from retrieval

    query_from_research_spec    build query from ResearchSpec
    query_from_resolved_spec    build query from ResolvedResearchSpec
    query_from_experiment_plan  build query from ExperimentPlan

    retrieve_research_memory_context  main connector function
"""

from src.agent.memory.memory_connector import retrieve_research_memory_context
from src.agent.memory.memory_context import ResearchMemoryContext
from src.agent.memory.memory_evidence import ResearchMemoryEvidence
from src.agent.memory.memory_query import (
    ResearchMemoryQuery,
    query_from_experiment_plan,
    query_from_research_spec,
    query_from_resolved_spec,
)

__all__ = [
    "ResearchMemoryQuery",
    "ResearchMemoryEvidence",
    "ResearchMemoryContext",
    "query_from_research_spec",
    "query_from_resolved_spec",
    "query_from_experiment_plan",
    "retrieve_research_memory_context",
]
