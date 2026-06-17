"""Memory-Aware Next-Action Planner — Phase 5B/5D.

Produces structured, evidence-backed next-action recommendations from a
ResearchMemoryContext.  All generation is deterministic; no LLM calls.
No drafts are created; no experiments are executed.

Public API — Phase 5B:

    VALID_ACTION_TYPES              frozenset of supported action_type values
    VALID_PRIORITIES                frozenset["high", "medium", "low"]
    VALID_CONFIDENCES               frozenset["high", "medium", "low"]

    NextActionCandidate             one possible next research action
    NextActionRecommendationResult  result envelope from the planner
    NextActionCriticIssue           one critic check result
    NextActionCriticResult          full critic validation result

    recommend_next_actions          main planner function
    critique_next_action_recommendations  critic function
    summarise_next_action_result    plain-text summary (no LLM)

Public API — Phase 5D:

    ResearchSpecProposal            a proposed ResearchSpec derived from a candidate
    ResearchSpecProposalResult      result envelope from propose_spec_from_next_action
    propose_spec_from_next_action   candidate → spec proposal (deterministic, no LLM)
    summarise_spec_proposal         plain-text summary of a proposal result (no LLM)
"""

from src.agent.next_actions.candidate_to_spec import propose_spec_from_next_action
from src.agent.next_actions.next_action import (
    VALID_ACTION_TYPES,
    VALID_CONFIDENCES,
    VALID_PRIORITIES,
    NextActionCandidate,
)
from src.agent.next_actions.next_action_critic import (
    NextActionCriticIssue,
    NextActionCriticResult,
    critique_next_action_recommendations,
)
from src.agent.next_actions.next_action_planner import recommend_next_actions
from src.agent.next_actions.next_action_result import (
    NextActionRecommendationResult,
    summarise_next_action_result,
)
from src.agent.next_actions.spec_proposal import (
    ResearchSpecProposal,
    ResearchSpecProposalResult,
    summarise_spec_proposal,
)

__all__ = [
    # Phase 5B
    "VALID_ACTION_TYPES",
    "VALID_PRIORITIES",
    "VALID_CONFIDENCES",
    "NextActionCandidate",
    "NextActionRecommendationResult",
    "NextActionCriticIssue",
    "NextActionCriticResult",
    "recommend_next_actions",
    "critique_next_action_recommendations",
    "summarise_next_action_result",
    # Phase 5D
    "ResearchSpecProposal",
    "ResearchSpecProposalResult",
    "propose_spec_from_next_action",
    "summarise_spec_proposal",
]
