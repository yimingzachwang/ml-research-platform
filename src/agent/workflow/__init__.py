"""Agent workflow façade — deterministic spec → plan → draft pipeline (Phase 3C/3D/4B/5C).

Public functions:
    run_resolve_spec()                        — ResearchSpec dict → WorkflowResult
    run_plan_from_spec()                      — ResearchSpec dict → WorkflowResult (adds plan)
    run_draft_from_spec()                     — ResearchSpec dict → WorkflowResult (adds draft preview)
    save_agent_draft_result()                 — WorkflowResult → SavedAgentDraftResult (Phase 3D)
    run_resolve_from_request()                — NL request → parse → resolve (Phase 4B)
    run_plan_from_request()                   — NL request → parse → resolve → plan (Phase 4B)
    run_draft_from_request()                  — NL request → parse → resolve → plan → draft (Phase 4B)
    run_recommend_next_actions()              — spec → resolve → plan → memory → recommendations (Phase 5C)
    run_recommend_next_actions_from_request() — NL request → parse → above (Phase 5C)

These are exposed as MCP tools via src/mcp/zeto_server.py.
No execution. No approval. Phase 5C tools never create drafts.
"""

from src.agent.workflow.draft_persistence import SavedAgentDraftResult, save_agent_draft_result
from src.agent.workflow.natural_language_workflow import (
    run_draft_from_request,
    run_plan_from_request,
    run_resolve_from_request,
)
from src.agent.workflow.next_action_workflow import (
    run_recommend_next_actions,
    run_recommend_next_actions_from_request,
)
from src.agent.workflow.research_workflow import (
    run_draft_from_spec,
    run_plan_from_spec,
    run_resolve_spec,
)
from src.agent.workflow.workflow_result import WorkflowResult

__all__ = [
    "WorkflowResult",
    "run_resolve_spec",
    "run_plan_from_spec",
    "run_draft_from_spec",
    "SavedAgentDraftResult",
    "save_agent_draft_result",
    "run_resolve_from_request",
    "run_plan_from_request",
    "run_draft_from_request",
    "run_recommend_next_actions",
    "run_recommend_next_actions_from_request",
]
