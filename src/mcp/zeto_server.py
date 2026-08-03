"""Zeto MCP server — exposes the governed Research API as MCP tools.

Runs over stdio so an MCP client (e.g. LM Studio) can drive the AI-assisted
research workflow:

    list experiments -> create session -> build context -> review -> proposal
    -> draft -> validate -> approve -> render YAML
    -> (optional, explicitly confirmed) execute -> post-run review -> summary

Governance held by this layer:
  * tools wrap ``src.orchestration.api.research_api`` ONLY — no quant engine
    internals, no subprocess, no shell, no file/eval access;
  * draft approval is its own explicit tool and is never performed implicitly;
  * YAML rendering is separate from execution;
  * execution requires ``confirmation="RUN"`` and runs exactly one config —
    no loop, no retry, no automatic lineage registration.

The quant engine remains authoritative; these tools coordinate and interpret.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from src.orchestration.api import research_api as _api  # noqa: E402
from src.orchestration.session.session_schema import SessionEventType  # noqa: E402
from src.agent.workflow import research_workflow as _agent_workflow  # noqa: E402
from src.agent.workflow.draft_persistence import (  # noqa: E402
    save_agent_draft_result as _save_agent_draft,
)
from src.agent.registry.registry_loader import load_registry as _load_registry  # noqa: E402
from src.agent.intent.intent_parser import (  # noqa: E402
    parse_research_intent as _parse_research_intent,
)
from src.agent.workflow import natural_language_workflow as _nlw  # noqa: E402
from src.agent.workflow import next_action_workflow as _naw  # noqa: E402
from src.agent.workflow import spec_proposal_workflow as _spw  # noqa: E402
from src.agent.workflow import proposal_to_draft_workflow as _ptdw  # noqa: E402
from src.agent.intent.llm_settings import resolve_llm_settings  # noqa: E402

mcp = FastMCP("zeto")


# ---------------------------------------------------------------------------
# Internal helpers (not exposed as tools)
# ---------------------------------------------------------------------------


def _record_event(
    session_id: str | None,
    event_type: str,
    experiment_name: str,
    data: dict[str, Any],
) -> dict[str, Any] | None:
    """Record one session event if a valid session_id is given; else no-op.

    Returns the session summary (or None when there is no session to record on).
    """
    if not session_id:
        return None
    session = _api.load_research_session(session_id)
    if session is None:
        return None
    session = _api.record_session_event(
        session=session,
        event_type=event_type,
        experiment_name=experiment_name,
        data=data,
    )
    return _api.summarize_research_session(session)


def _short_hash(h: str | None, n: int = 8) -> str:
    """First n characters of a hash for display (full hash stays in data)."""
    return (h or "")[:n]


def _short(text: str, limit: int = 200) -> str:
    """Truncate a string so a compact payload never grows unbounded."""
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _mfmt(value: Any) -> str:
    """Render a metric value for display; 'n/a' when missing. Never invents."""
    return "n/a" if value is None else str(value)


def _stage_summary(stage_executions: list[dict[str, Any]] | None) -> str:
    """Compact 'name Xs, name Ys, ...' stage-timing string for display text.

    Observability only — never affects execution_completed/execution_failed
    semantics, only the human-readable summary attached to a successful result.
    """
    if not stage_executions:
        return ""
    parts = [
        f"{s['name']} {s['duration_s']:.1f}s"
        for s in stage_executions
        if s.get("duration_s") is not None
    ]
    return ", ".join(parts)


def _name_sev(failure_modes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact failure-mode/flag form: only name + severity (no descriptions)."""
    return [
        {"name": fm.get("name"), "severity": fm.get("severity")}
        for fm in (failure_modes or [])
    ]


def _compact_diff(changes: list[Any]) -> list[dict[str, Any]]:
    """Compact config diff: dotted field + current/proposed values only."""
    return [
        {
            "field": f"{c.section}.{c.field}",
            "current_value": c.current_value,
            "proposed_value": c.proposed_value,
        }
        for c in changes
    ]


def _compact_multi_diff(changes: list[Any]) -> list[dict[str, Any]]:
    """Compact diff that handles feature add/remove alongside set changes."""
    result = []
    for c in changes:
        if c.section == "features":
            if c.field == "entries.add":
                feat = c.proposed_value
                result.append({
                    "field_path": "features.entries",
                    "operation": "add",
                    "value": (
                        {"name": feat.get("name"), "type": feat.get("type")}
                        if isinstance(feat, dict) else feat
                    ),
                })
            elif c.field == "entries.remove":
                result.append({
                    "field_path": "features.entries",
                    "operation": "remove",
                    "value": c.proposed_value,
                })
        else:
            result.append({
                "field_path": f"{c.section}.{c.field}",
                "operation": "set",
                "current_value": c.current_value,
                "proposed_value": c.proposed_value,
            })
    return result


def _diff_summary_lines(diff: list[dict[str, Any]]) -> list[str]:
    """One concise human-readable line per diff entry."""
    lines = []
    for d in diff:
        op = d.get("operation", "set")
        fp = d.get("field_path", d.get("field", "?"))
        if op == "set":
            lines.append(f"{fp}: {d.get('current_value')} -> {d.get('proposed_value')}")
        elif op == "add":
            v = d.get("value") or {}
            lines.append(
                f"features.entries ADD {v.get('name')!r} (type={v.get('type')!r})"
                if isinstance(v, dict) else f"features.entries ADD {v!r}"
            )
        elif op == "remove":
            lines.append(f"features.entries REMOVE {d.get('value')!r}")
        else:
            lines.append(f"{fp} [{op}]")
    return lines


def _envelope(
    ok: bool,
    stage: str,
    display: str,
    data: dict[str, Any],
    next_suggested_action: str,
) -> dict[str, Any]:
    """The visible-state contract returned by every tool.

    ok:                    whether the tool succeeded (False for errors/refusals)
    stage:                 machine-readable lifecycle stage
    display:               concise human-readable line for LM Studio to show
    data:                  the structured payload
    next_suggested_action: the single recommended next tool (advisory; the
                           researcher/model decides — there is no auto-chaining)
    """
    return {
        "ok": ok,
        "stage": stage,
        "display": display,
        "data": data,
        "next_suggested_action": next_suggested_action,
    }


def _fmt_failure_modes(failure_modes: list[dict[str, Any]]) -> str:
    if not failure_modes:
        return "none"
    return "; ".join(
        f"[{(fm.get('severity') or '?').upper()}] {fm.get('name')}" for fm in failure_modes
    )


def _fmt_validation(validation: dict[str, Any] | None) -> str:
    val = validation or {}
    keys = [
        "mean_oos_sharpe",
        "std_oos_sharpe",
        "n_splits",
        "n_negative_sharpe_splits",
        "worst_split_sharpe",
    ]
    shown = [f"{k}={val[k]}" for k in keys if k in val and val[k] is not None]
    return ", ".join(shown) if shown else "n/a"


def _review_experiment(
    experiment_name: str,
    session_id: str | None,
    provider: str,
    model: str | None,
    base_url: str | None,
    event_type: str,
    stage: str,
    label: str,
    next_action: str,
) -> dict[str, Any]:
    """Shared LLM-review path for pre-run and post-run reviews.

    Compact: returns context_hash, short flags (name+severity), section NAMES,
    and the path to the persisted review.  Full section bodies stay on disk.

    On any LLM failure or timeout this returns a clean ok=false envelope and
    stops — it never auto-retries the review (the user must ask again).
    """
    # Guard: core metrics must be present before calling the LLM review.
    # This prevents TypeError when the context builder encounters None metric
    # values, and stops invented metrics from entering the review narrative.
    _m = _api.get_experiment_metrics(experiment_name)
    if _m.get("status") == "not_found" or _m.get("metrics", {}).get("sharpe_ratio") is None:
        _missing_reason = (
            "experiment not found — run the experiment first"
            if _m.get("status") == "not_found"
            else ", ".join(_m.get("missing_metrics") or ["sharpe_ratio (absent)"])
        )
        return _envelope(
            False,
            stage.replace("_generated", "_blocked_incomplete_metrics"),
            f"Review blocked — metrics are incomplete for '{experiment_name}': "
            f"{_missing_reason}. "
            "Do not infer or invent metrics. "
            "Inspect artefacts with verify_experiment_artifacts first.",
            {"experiment_name": experiment_name, "incomplete_reason": _missing_reason},
            "stop_and_report_to_user",
        )

    try:
        review = _api.run_llm_review(
            experiment_name,
            provider=provider,
            model=model,
            base_url=base_url,
            persist_context=False,
            persist_review=True,
        )
        ctx = _api.build_llm_context(experiment_name, persist=False)
        context_hash = _api.compute_context_hash(ctx)
    except Exception as exc:  # noqa: BLE001 — surface a clean envelope, never a traceback
        fail_stage = stage.replace("_generated", "_failed")
        return _envelope(
            False, fail_stage,
            f"{label} FAILED for '{experiment_name}': {type(exc).__name__}: {exc}. "
            "Stop and report the failure. Do not auto-retry unless the user asks.",
            {"experiment_name": experiment_name, "error": f"{type(exc).__name__}: {exc}"},
            "stop_and_report_to_user",
        )
    _record_event(
        session_id,
        event_type,
        experiment_name,
        {"provider": provider, "context_hash": context_hash},
    )
    flags = _name_sev(ctx.failure_modes)
    section_names = list(review.sections.keys())
    display = (
        f"{label} of '{experiment_name}' (hash {_short_hash(context_hash)}). "
        f"Flags: {_fmt_failure_modes(flags)}. "
        f"Sections: {', '.join(section_names) or '(none parsed)'}."
    )
    data = {
        "experiment_name": experiment_name,
        "context_hash": context_hash,
        "flags": flags,
        "section_names": section_names,
        "review_path": _api.review_artifact_path(experiment_name),
    }
    return _envelope(True, stage, display, data, next_action)


# ---------------------------------------------------------------------------
# Operator manual (fixed, read-only)
# ---------------------------------------------------------------------------

_OPERATOR_MANUAL_PATH = "docs/LM_STUDIO_QWEN_OPERATOR_MANUAL.md"


def _load_operator_manual_text() -> "str | None":
    """Read operator manual markdown from disk; returns None if missing."""
    try:
        return (PROJECT_ROOT / _OPERATOR_MANUAL_PATH).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None


_OPERATOR_MANUAL_TEXT: "str | None" = _load_operator_manual_text()


def _experiment_config_exists(experiment_name: str) -> bool:
    """Return True if a config YAML exists on disk for this experiment name."""
    from src.orchestration.utils.filesystem import experiment_config_path  # noqa: PLC0415
    return experiment_config_path(experiment_name).exists()


def _wrong_tool_new_design_redirect(experiment_name: str, caller: str) -> dict[str, Any]:
    """Return a routing-redirect envelope for non-existent experiment names.

    Fires when a tool that requires an existing config-backed experiment is called
    with what looks like an invented name from a new research-design request.
    """
    return _envelope(
        False,
        "wrong_tool_for_new_research_design_request",
        f"'{experiment_name}' is not a config-backed experiment. "
        f"{caller} requires an existing experiment with a YAML in configs/experiments/. "
        "For a new research-design request, use create_validated_draft_from_request instead.",
        {
            "experiment_name": experiment_name,
            "recommended_tool": "create_validated_draft_from_request",
            "governance": {
                "draft_created": False,
                "draft_approved": False,
                "execution_performed": False,
            },
        },
        "create_validated_draft_from_request",
    )


# ---------------------------------------------------------------------------
# Session creation guard (Phase 5G.3)
# ---------------------------------------------------------------------------

# Terms that strongly suggest an invented experiment name from a new research-design
# request rather than a real config-backed experiment.
_SESSION_DESIGN_SIGNALS: frozenset[str] = frozenset({
    "canonical", "multi asset", "etf", "preset",
    "lasso", "ridge", "elasticnet",
    "2015", "2016", "2017", "2018", "2019", "2020",
    "2021", "2022", "2023", "2024",
})


def _looks_like_new_design_session(root_experiment: str, research_goal: str) -> bool:
    """Return True if the name/goal suggest an invented new-design experiment."""
    # Normalise separators → space so CamelCase ETF and snake_case etf both match.
    combined = (
        f"{root_experiment} {research_goal}"
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )
    return any(sig in combined for sig in _SESSION_DESIGN_SIGNALS)


# Patterns that identify a placeholder config path rather than a real on-disk path.
_PLACEHOLDER_PATH_PREFIXES: tuple[str, ...] = (
    "/path/to/",
    "path/to/",
)


def _is_placeholder_path(config_path: str) -> bool:
    """Return True if config_path is clearly a placeholder string, not a real file."""
    if not config_path or not config_path.strip():
        return True
    stripped = config_path.strip()
    low = stripped.lower()
    if any(low.startswith(p) for p in _PLACEHOLDER_PATH_PREFIXES):
        return True
    if "<" in stripped and ">" in stripped:
        return True
    return False


# ---------------------------------------------------------------------------
# Request router (Phase 5G.2) — deterministic NL routing, no LLM
# ---------------------------------------------------------------------------

_NEW_DESIGN_SIGNALS: frozenset[str] = frozenset({
    "preset", "lasso", "ridge", "random forest", "timeframe", "features",
    "validation", "instead of", "change model", "different model",
    "canonical multi-asset", "canonical etf", "use the canonical", "use canonical",
    "test lasso", "test ridge", "test a different", "from 2015", "from 2016",
    "from 2017", "from 2018", "from 2019", "from 2020", "from 2021", "from 2022",
    "from 2023", "2015 to", "2016 to", "2017 to", "2018 to", "2019 to", "2020 to",
    "2021 to", "2022 to", "2023 to", "2024 to", "to 2024", "to 2023",
})
_MEMORY_NEXT_SIGNALS: frozenset[str] = frozenset({
    "what should we try", "what to try next", "what next", "next action",
    "next step", "recommend", "should we try", "what have we learned",
    "suggest next",
})
_REVIEW_SIGNALS: frozenset[str] = frozenset({
    "review", "compare", "comparison", "inspect evidence", "build context",
    "context summary", "summarise", "summarize", "what did we learn",
})
_GOVERNANCE_SIGNALS: frozenset[str] = frozenset({
    "approve", "approval", "render the draft", "render draft", "render yaml",
})
_EXECUTION_SIGNALS: frozenset[str] = frozenset({
    "run the experiment", "execute the experiment", "start the experiment",
    "run it", "execute it", "start execution", " run ", "execute now",
    "run now",
})


def _classify_request(
    request: str,
) -> tuple[str, str, list[str], list[str], bool]:
    """Return (request_type, recommended_tool, next_tools, discouraged_tools, requires_confirmation)."""
    low = f" {request.lower()} "

    if any(sig in low for sig in _EXECUTION_SIGNALS):
        return (
            "execution_request",
            "execute_approved_config",
            ["approve_experiment_draft", "render_draft_to_yaml"],
            [],
            True,
        )
    if any(sig in low for sig in _GOVERNANCE_SIGNALS):
        return (
            "existing_draft_governance",
            "approve_experiment_draft",
            ["render_draft_to_yaml", "execute_approved_config"],
            [],
            True,
        )
    if any(sig in low for sig in _MEMORY_NEXT_SIGNALS):
        return (
            "memory_next_action",
            "recommend_next_actions_from_request",
            ["propose_spec_from_recommendation_request", "create_validated_draft_from_request"],
            ["generate_experiment_draft", "generate_iteration_proposal"],
            False,
        )
    if any(sig in low for sig in _NEW_DESIGN_SIGNALS):
        return (
            "new_research_design",
            "create_validated_draft_from_request",
            ["parse_research_intent", "resolve_research_request"],
            [
                "create_research_session", "build_context_summary",
                "run_experiment_review", "generate_iteration_proposal",
                "generate_experiment_draft", "execute_approved_config",
            ],
            False,
        )
    if any(sig in low for sig in _REVIEW_SIGNALS):
        return (
            "existing_experiment_review",
            "run_experiment_review",
            ["build_context_summary", "generate_iteration_proposal"],
            ["execute_approved_config", "approve_experiment_draft"],
            False,
        )
    return (
        "unclear",
        "ask_user_clarification",
        ["route_research_request", "get_zeto_operator_manual"],
        [],
        False,
    )


_ROUTING_REASONS: dict[str, str] = {
    "new_research_design": (
        "Request describes a new research design (preset, model, or timeframe). "
        "create_validated_draft_from_request is the correct entry point — "
        "do not call create_research_session, build_context_summary, or "
        "generate_experiment_draft for new designs."
    ),
    "memory_next_action": (
        "Request asks what to try next. "
        "recommend_next_actions_from_request uses prior memory evidence to suggest "
        "ranked candidates without inventing experiment names."
    ),
    "existing_experiment_review": (
        "Request appears to be about reviewing or comparing existing experiments. "
        "run_experiment_review or build_context_summary require an existing "
        "config-backed experiment name."
    ),
    "existing_draft_governance": (
        "Request mentions draft approval or rendering. "
        "These operations require explicit user confirmation (approval_confirmation='APPROVE')."
    ),
    "execution_request": (
        "Request mentions execution. "
        "execute_approved_config requires confirmation='RUN' in a fresh user message "
        "and an approved YAML config path — never infer RUN from prior messages."
    ),
    "unclear": (
        "Request intent is unclear. "
        "Ask the user for clarification or call get_zeto_operator_manual to review routing rules."
    ),
}


def route_research_request(request: str) -> dict[str, Any]:
    """Classify a natural-language research request and recommend the correct tool.

    Returns request_type, recommended_tool, recommended_next_tools,
    discouraged_tools, requires_confirmation, reason, and governance flags.

    Read-only — deterministic keyword classification, no LLM, no session,
    no state mutation.  Call this for any ambiguous NL request before choosing
    an orchestration tool.  Follow its recommended_tool, not discouraged_tools.

    Governance flags are always:
        draft_created=false, draft_approved=false, execution_performed=false.
    """
    req_type, rec_tool, rec_next, disc_tools, req_confirm = _classify_request(request)
    reason = _ROUTING_REASONS[req_type]
    data: dict[str, Any] = {
        "request_type": req_type,
        "recommended_tool": rec_tool,
        "recommended_next_tools": rec_next,
        "discouraged_tools": disc_tools,
        "requires_confirmation": req_confirm,
        "reason": reason,
        "governance": {
            "approval_required": req_type == "existing_draft_governance",
            "draft_created": False,
            "draft_approved": False,
            "execution_performed": False,
        },
    }
    display = (
        f"Request type: {req_type}. "
        f"Use: {rec_tool}. "
        f"{reason}"
    )
    return _envelope(True, "request_routed", display, data, rec_tool)


# Fixed, in-code rules — returned as-is. This tool never reads files.
_OPERATOR_RULES: tuple[str, ...] = (
    # --- Routing first rule (Phase 5G.2) — MUST appear first ---
    "ROUTING FIRST: For any NL request that is not clearly an existing-experiment "
    "operation, call route_research_request(request=...) first and follow its "
    "recommended_tool. For new preset/model/timeframe requests do NOT call "
    "create_research_session, build_context_summary, run_experiment_review, "
    "generate_iteration_proposal, generate_experiment_draft, or execute_approved_config. "
    "Use create_validated_draft_from_request instead.",
    # --- Session creation guard (Phase 5G.3) ---
    "SESSION GUARD: Do NOT call create_research_session with an invented name like "
    "'CanonicalMultiAssetETF_2015_to_2024'. That handler returns "
    "wrong_tool_for_new_research_design_request for non-existent experiments with "
    "design-like names. Call route_research_request first, then "
    "create_validated_draft_from_request.",
    "Use exact snake_case MCP tool names; do not camelCase tool names.",
    "Show each tool's display field after every tool call.",
    "Use compact tool outputs only; reason from paths, hashes, and IDs.",
    "Before major transitions, call check_research_workflow_state and follow its next_suggested_action.",
    "Store the session_id UUID returned by create_research_session.",
    "Pass session_id to all session-aware tools (review, proposal, draft, "
    "validate, approve, render, execute, post-run review, get_session_summary).",
    "If a session exists, always pass its session_id to approve/render/execute/"
    "post-run review; do not pass session_id=null while an active session exists.",
    "If session_id is lost, call get_latest_research_session before continuing.",
    "Never use experiment_name, context_hash, draft_id, or config_path as session_id.",
    "Never invent metrics; use tool outputs only.",
    "Never invent or hand-write draft configs; drafts come from generate_experiment_draft "
    "or generate_parameter_change_draft only.",
    "For a specific user-requested config change (e.g. 'set alpha to 2'), use "
    "generate_parameter_change_draft, not generate_experiment_draft.",
    "For Sharpe/OOS Sharpe/drawdown/hit-rate or 'did it improve' questions, use "
    "get_experiment_metrics or compare_experiment_metrics, never retrieve_research_memory "
    "or semantic_retrieve_research_memory.",
    "Never invent report contents or sample metrics; performance is authoritative only "
    "from experiment artefacts.",
    "If a tool returns ok=false, report the failure verbatim and stop.",
    "If validation fails, report the failure and stop; do not repeatedly call generate_experiment_draft.",
    "If validation fails because the proposed experiment name already exists, "
    "ask the user whether to use the next available suffix or clean existing local demo artefacts.",
    "Approval requires approval_confirmation='APPROVE'; validation success never authorises approval.",
    "Only a fresh user message that explicitly approves (prefer 'I approve the draft.') "
    "authorises approve_experiment_draft; never infer approval from 'start a research "
    "cycle' or a next_suggested_action.",
    "After approval_refused, stop and wait for the user; do not retry automatically.",
    "Do not render YAML unless the user approves the draft.",
    "Do not execute unless the user explicitly provides RUN (confirmation='RUN').",
    "Do not treat 'execute', 'yes', 'proceed', or 'continue' as RUN; only a fresh "
    "user message containing the literal token RUN authorises execution.",
    "Never infer RUN from a previous approval, a previous render, or a prior "
    "failed/refused execution attempt.",
    "After execution_refused, stop and wait for a new user message; do not retry "
    "execute_approved_config automatically.",
    "Do not run extra experiments, optimise automatically, or loop.",
    "A high-level request such as 'start a research cycle' does NOT authorise "
    "approval, rendering, or execution; it only authorises the read/advisory steps.",
    "Stop after validate_experiment_draft passes and ask the user for approval.",
    "Stop after render_draft_to_yaml and ask the user for RUN.",
    "Stop after the final get_session_summary; the research cycle is complete.",
    "Do not start a second iteration (new proposal, draft, or experiment) unless the user explicitly asks.",
    "Before review/proposal/draft, optionally call retrieve_research_memory or "
    "semantic_retrieve_research_memory for prior related evidence.",
    "Research memory is evidence-only; retrieved memory does not authorise execution or "
    "approval; semantic matches are suggestions, not proof.",
    "Quant metrics remain authoritative. If semantic retrieval fails, report and stop; "
    "do not invent prior evidence.",
    "For local Qwen-backed review/proposal/draft, use provider=openai, model=qwen2.5-7b-instruct, base_url=http://127.0.0.1:1234/v1.",
    "Governed sequence: create session -> check state -> execute baseline if "
    "needed -> build context -> review -> proposal -> draft -> validate -> "
    "approval -> render YAML -> RUN execution -> post-run review -> session summary.",
    "The quant engine remains authoritative; Qwen coordinates and interprets only.",
    "Config routing: 'what can we change?'->list_changeable_config_fields; "
    "'what features?'->list_available_features; 'which models?'->list_supported_models; "
    "'config overview?'->inspect_experiment_config. Never invent feature types or model names.",
    "For multiple changes or feature add/remove/replace, use generate_config_change_draft "
    "(changes is a JSON array). generate_parameter_change_draft handles a single set.",
    "If generate_config_change_draft fails, report the display verbatim and stop. "
    "Do not retry the same change. Use 'value' key in each change dict, never 'proposed_value'.",
    "For 'what did we learn from X vs Y?' or 'inspect evidence for X vs Y', use "
    "inspect_comparison_evidence (direct, authoritative). If no evidence "
    "exists, run compare_experiment_metrics first. Metrics remain authoritative.",
    "Experiment intelligence routing: "
    "'what have we learned?'/'summarise family' -> summarise_experiment_family; "
    "'which experiment is best?'/'rank experiments' -> rank_experiments; "
    "'have we tested this?'/'idea history' -> check_research_idea_history; "
    "'what should we try next?'/'propose plan' -> propose_research_plan. "
    "All use authoritative artefact metrics; never RAG as proof of performance.",
    "rank_experiments/summarise_experiment_family/check_research_idea_history/"
    "propose_research_plan call no LLM, execute nothing, approve nothing, and "
    "render nothing. propose_research_plan is advisory only — its plan steps do "
    "not authorise execution or approval.",
    "After execute_approved_config, call verify_experiment_artifacts before review. "
    "If artefact_status=incomplete, stop and report to user — do not compare, "
    "review, index, or invent metrics.",
    "If get_experiment_metrics returns ok=false, stop — do not compare, review, "
    "index, or invent metrics for that experiment.",
    "If compare_experiment_metrics returns ok=false, stop — do not inspect "
    "evidence or invent conclusions.",
    "If stage contains blocked_incomplete_metrics, stop — never invent "
    "hypothetical Sharpe, OOS Sharpe, drawdown, or hit-rate values. "
    "RAG memory cannot substitute for missing authoritative metrics.",
    # --- Tool routing rules (Phase 5G.1) ---
    "ROUTING A: For a brand-new research-design request (new preset, model, "
    "timeframe, or feature change) use create_validated_draft_from_request. "
    "Do NOT use generate_experiment_draft for new design requests.",
    "ROUTING B: For 'what should we try next?' or memory-aware recommendations, "
    "use recommend_next_actions_from_request then propose_spec_from_recommendation_request.",
    "ROUTING C: Use generate_experiment_draft / generate_iteration_proposal / "
    "run_experiment_review ONLY for existing config-backed experiments that "
    "already have a YAML file in configs/experiments/.",
    "ROUTING D: If generate_experiment_draft returns wrong_tool_for_research_design_request, "
    "do NOT retry it — switch immediately to create_validated_draft_from_request.",
    "ROUTING E: Do NOT invent experiment names. Experiment names come from the "
    "governed pipeline (create_validated_draft_from_request, registry presets), "
    "not from LLM inference. Never generate a name like TestLasso2015_2024.",
    "ROUTING F: If a low-level tool returns 'unknown experiment', 'missing config', "
    "'no YAML file found', 'no IterationProposal found', or "
    "'wrong_tool_for_research_design_request', do NOT retry with the same or "
    "modified experiment name — re-route to the appropriate high-level agent tool.",
    "ROUTING G: Governance gates are unchanged regardless of routing: "
    "approve requires approval_confirmation='APPROVE'; render requires an approved "
    "draft; execute requires confirmation='RUN'. NL wording alone never triggers "
    "these gates.",
)


def get_zeto_operator_manual() -> dict[str, Any]:
    """Return the fixed Zeto operator manual (read-only).

    Loads a compact rules summary so an LM Studio chat does not need long rules
    pasted in. This tool is read-only: it executes nothing, calls no LLM,
    inspects no experiment artefacts, creates no session, mutates no state, and
    reads no files — it returns only the fixed in-code manual (no arbitrary
    paths).
    """
    display = (
        "Zeto operator manual loaded. "
        "ROUTING FIRST: For any natural-language research-design request, call "
        "route_research_request first. "
        "Do not create a research session first. "
        "Do not invent experiment names. "
        "For preset/model/timeframe requests, use create_validated_draft_from_request. "
        "Full rules in data.rules. Execute only with RUN."
    )
    data = {
        "manual_path": _OPERATOR_MANUAL_PATH,
        "rules": list(_OPERATOR_RULES),
        "manual_available": _OPERATOR_MANUAL_TEXT is not None,
    }
    return _envelope(
        True, "operator_manual_loaded", display, data, "check_research_workflow_state"
    )


# ---------------------------------------------------------------------------
# Research memory tools (Phase 1 — metadata/keyword RAG)
#
# Evidence-only. These tools index compact summaries of existing artefacts and
# retrieve prior research evidence by deterministic keyword/metadata matching.
# They run no experiment, call no LLM, approve nothing, render nothing, and
# never authorise execution. Full artefacts stay on disk; only compact pointers
# (summaries, paths, hashes, tags, matched terms) are returned.
# ---------------------------------------------------------------------------


def get_research_memory_status() -> dict[str, Any]:
    """Read-only: report whether the research-memory index exists and its size."""
    status = _api.get_research_memory_status()
    if status["index_exists"]:
        display = (
            f"Research memory contains {status['item_count']} indexed item(s) "
            f"across {status['experiment_count']} experiment(s)."
        )
        next_action = "retrieve_research_memory"
    else:
        display = "Research memory index does not exist yet. Run index_research_memory."
        next_action = "index_research_memory"
    return _envelope(True, "research_memory_status", display, status, next_action)


def index_research_memory() -> dict[str, Any]:
    """Controlled write: build/refresh the memory index from known artefacts only.

    Runs no experiment, calls no LLM, approves/renders/executes nothing, and
    never mutates source artefacts. Reads only known Zeto artefact locations.
    """
    result = _api.index_research_memory()
    display = (
        f"Indexed {result['indexed_count']} research memory item(s) across "
        f"{result['experiment_count']} experiment(s)."
    )
    return _envelope(
        True, "research_memory_indexed", display, result, "retrieve_research_memory"
    )


def retrieve_research_memory(
    query: str | None = None,
    experiment_name: str | None = None,
    failure_modes: list[str] | None = None,
    artefact_type: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Read-only: retrieve prior research evidence by keyword/metadata match.

    Returns up to top_k compact items (summaries, paths, hashes, tags, matched
    terms) — never full artefact contents. Evidence only: a result authorises no
    execution, approval, or rendering; quant metrics remain authoritative.
    """
    items = _api.retrieve_research_memory(
        query=query,
        experiment_name=experiment_name,
        failure_modes=failure_modes,
        artefact_type=artefact_type,
        top_k=top_k,
    )
    focus = ", ".join(failure_modes) if failure_modes else (query or experiment_name or "recent")
    display = f"Retrieved {len(items)} related memory item(s) for {_short(focus, 80)}."
    data = {"item_count": len(items), "items": items}
    return _envelope(True, "research_memory_retrieved", display, data, "run_experiment_review")


# ---------------------------------------------------------------------------
# Semantic research memory tools (Phase 2 — local embedding retrieval)
#
# Evidence-only. These layer a local embedding index over the Phase 1 records
# and retrieve by cosine similarity. They embed compact summaries via a local
# embeddings endpoint only (never a chat/completion LLM), run no experiment,
# approve nothing, render nothing, and never authorise execution. Full artefacts
# stay on disk; only compact pointers (scores, summaries, paths, hashes, tags)
# are returned.
# ---------------------------------------------------------------------------


def get_semantic_research_memory_status() -> dict[str, Any]:
    """Read-only: report whether the semantic memory index exists and its size."""
    status = _api.get_semantic_research_memory_status()
    if status["index_exists"]:
        display = (
            f"Semantic research memory contains {status['item_count']} embedded "
            f"item(s) using {status['embedding_model']}."
        )
        next_action = "semantic_retrieve_research_memory"
    else:
        display = (
            "Semantic research memory index does not exist yet. Run "
            "index_semantic_research_memory."
        )
        next_action = "index_semantic_research_memory"
    return _envelope(True, "semantic_memory_status", display, status, next_action)


def index_semantic_research_memory(
    provider: str = "openai",
    model: str = "text-embedding-nomic-embed-text-v1.5",
    base_url: str = "http://127.0.0.1:1234/v1",
) -> dict[str, Any]:
    """Controlled write: embed Phase 1 memory records into a local semantic index.

    Embeds compact summaries via the local embeddings endpoint only — never a
    chat/completion LLM. Runs no experiment, approves/renders/executes nothing,
    mutates no source artefacts, and reads no arbitrary paths. Requires a Phase 1
    index first; on embedding-endpoint failure it returns a clean ok=false
    envelope and stops (no auto-retry).
    """
    result = _api.index_semantic_research_memory(
        provider=provider, model=model, base_url=base_url
    )
    status = result.get("status")
    if status == "no_phase1_index":
        return _envelope(
            False, "semantic_memory_index_blocked",
            "No Phase 1 research memory index found. Run index_research_memory first.",
            {"error": "Phase 1 memory index not found", **result},
            "index_research_memory",
        )
    if status == "embedding_failed":
        return _envelope(
            False, "semantic_memory_index_failed",
            "Embedding endpoint failed. Report the failure and stop; do not "
            f"auto-retry. Error: {_short(str(result.get('error')), 160)}",
            result,
            "ask_user_to_retry_semantic_index",
        )
    display = (
        f"Embedded {result['embedded_count']} research memory item(s) using "
        f"{result['embedding_model']}."
    )
    return _envelope(
        True, "semantic_memory_indexed", display, result, "semantic_retrieve_research_memory"
    )


def semantic_retrieve_research_memory(
    query: str,
    top_k: int = 5,
    experiment_name: str | None = None,
    failure_modes: list[str] | None = None,
    artefact_type: str | None = None,
    tags: list[str] | None = None,
    provider: str = "openai",
    model: str = "text-embedding-nomic-embed-text-v1.5",
    base_url: str = "http://127.0.0.1:1234/v1",
) -> dict[str, Any]:
    """Read-only: retrieve prior research evidence by local semantic similarity.

    Embeds the query with the same local embedding model and ranks the semantic
    index by cosine similarity (with optional metadata filters). Returns up to
    top_k compact items (scores, summaries, paths, hashes, tags, failure modes)
    — never full artefact bodies. Evidence only: authorises no execution,
    approval, or rendering; quant metrics remain authoritative. On any failure it
    returns a clean ok=false envelope and stops — it never invents evidence.
    """
    result = _api.semantic_retrieve_research_memory(
        query=query,
        top_k=top_k,
        experiment_name=experiment_name,
        failure_modes=failure_modes,
        artefact_type=artefact_type,
        tags=tags,
        provider=provider,
        model=model,
        base_url=base_url,
    )
    status = result.get("status")
    if status == "no_phase1_index":
        return _envelope(
            False, "semantic_memory_unavailable",
            "No Phase 1 research memory index found. Run index_research_memory first.",
            {"error": "Phase 1 memory index not found", "query": query, "items": []},
            "index_research_memory",
        )
    if status == "no_semantic_index":
        return _envelope(
            False, "semantic_memory_unavailable",
            "No semantic memory index found. Run index_semantic_research_memory first.",
            {"error": "Semantic memory index not found", "query": query, "items": []},
            "index_semantic_research_memory",
        )
    if status == "embedding_failed":
        return _envelope(
            False, "semantic_memory_retrieval_failed",
            "Embedding endpoint failed. Report the failure and stop; do not "
            f"auto-retry or invent evidence. Error: {_short(str(result.get('error')), 160)}",
            {"error": result.get("error"), "query": query, "items": []},
            "ask_user_to_retry_semantic_retrieval",
        )
    items = result.get("items", [])
    focus = ", ".join(failure_modes) if failure_modes else _short(query, 80)
    display = f"Retrieved {len(items)} semantically related memory item(s) for {focus}."
    data = {"query": query, "item_count": len(items), "items": items}
    return _envelope(
        True, "semantic_memory_retrieved", display, data, "run_experiment_review"
    )


# ---------------------------------------------------------------------------
# Explicit parameter-change draft (deterministic, no LLM)
#
# Use this when the user requests a specific config change ("set alpha to 2").
# It applies exactly that change, reads the real current value from the base
# config, and persists an unapproved draft. It never approves, renders, executes,
# or calls an LLM.
# ---------------------------------------------------------------------------


def generate_parameter_change_draft(
    experiment_name: str,
    field_path: str,
    proposed_value: Any,
    session_id: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Create a draft from one explicit, user-requested config change.

    Use this (NOT generate_experiment_draft) when the user asks for a specific
    change such as 'set model.params.alpha to 2'. Deterministic and LLM-free:
    applies exactly field_path -> proposed_value, reads the real current value
    from the base config, assigns a unique proposed name, validates against the
    schema, and persists an unapproved draft. Never approves, renders, executes,
    or retries. Invalid field paths and schema-incompatible values are refused.
    """
    result = _api.generate_parameter_change_draft(
        experiment_name=experiment_name,
        field_path=field_path,
        proposed_value=proposed_value,
        reason=reason,
    )
    status = result.get("status")
    if status != "ok":
        errors = [_short(e, 160) for e in result.get("errors", [])][:3]
        display = (
            f"Parameter-change draft refused ({status}): " + ("; ".join(errors) or "see errors")
            + ". Do not invent a fallback field or value; ask the user to correct the request."
        )
        return _envelope(
            False, "parameter_change_draft_failed", display,
            {"experiment_name": experiment_name, "field_path": field_path,
             "status": status, "errors": errors},
            "ask_user_to_correct_parameter_change",
        )

    draft = result["draft"]
    _record_event(
        session_id,
        SessionEventType.DRAFT_GENERATED,
        experiment_name,
        {"draft_id": draft.draft_id, "draft_hash": draft.draft_hash,
         "proposed_name": draft.proposed_name},
    )
    diff = _compact_diff(draft.changes)
    diff_lines = [
        f"{d['field']}: {d['current_value']} -> {d['proposed_value']}" for d in diff
    ]
    data = {
        "experiment_name": experiment_name,
        "draft_id": draft.draft_id,
        "proposed_name": draft.proposed_name,
        "approved": draft.approved,
        "diff": diff,
        "draft_path": _api.draft_artifact_path(experiment_name, draft.draft_id),
    }
    display = (
        f"Draft {draft.draft_id} '{draft.proposed_name}' created (approved="
        f"{draft.approved}). Proposed config diff: " + ("; ".join(diff_lines) or "(none)")
    )
    return _envelope(
        True, "parameter_change_draft_generated", display, data, "validate_experiment_draft"
    )


# ---------------------------------------------------------------------------
# Authoritative experiment metrics (read from artefacts, NOT from RAG memory)
#
# Use these to answer performance questions (Sharpe, OOS Sharpe, drawdown, hit
# rate, "did it improve?"). They read real on-disk diagnostics — never research
# memory. Compact output; full reports stay on disk.
# ---------------------------------------------------------------------------


def get_experiment_metrics(experiment_name: str) -> dict[str, Any]:
    """Read-only: authoritative metrics for one experiment from its artefacts.

    Use this (NOT retrieve_research_memory / semantic_retrieve_research_memory)
    to answer Sharpe / OOS Sharpe / drawdown / hit-rate questions. Returns compact
    metrics from real diagnostics with a missing_metrics list — never invented
    values. No LLM, no execution.
    """
    result = _api.get_experiment_metrics(experiment_name)
    if result.get("status") == "not_found":
        return _envelope(
            False, "experiment_metrics_unavailable",
            f"No metrics artefacts found for '{experiment_name}'. Do not invent "
            "metrics; verify the experiment name or run it first.",
            {"experiment_name": experiment_name, "metrics": {}, "missing_metrics": []},
            "list_experiments",
        )
    m = result["metrics"]
    # Core metrics must be numeric — None here means the artefact is incomplete
    if m.get("sharpe_ratio") is None:
        _core_missing = result.get("missing_metrics") or ["sharpe_ratio"]
        return _envelope(
            False, "experiment_metrics_incomplete",
            f"Metrics artefacts for '{experiment_name}' exist but core metrics are "
            f"absent: {', '.join(_core_missing)}. "
            "Do not compare, review, or invent values. "
            "Use verify_experiment_artifacts to inspect the artefacts.",
            {"experiment_name": experiment_name,
             "missing_metrics": _core_missing,
             "artefact_check": "verify_experiment_artifacts"},
            "stop_and_report_to_user",
        )
    fm = result.get("failure_modes") or []
    display = (
        f"Metrics for {experiment_name}: Sharpe={_mfmt(m.get('sharpe_ratio'))}, "
        f"mean OOS Sharpe={_mfmt(m.get('mean_oos_sharpe'))}, "
        f"MaxDD={_mfmt(m.get('max_drawdown_pct'))}. "
        f"Failure modes: {', '.join(fm) or 'none'}."
    )
    return _envelope(
        True, "experiment_metrics_loaded", display, result, "compare_experiment_metrics"
    )


def compare_experiment_metrics(
    base_experiment_name: str,
    candidate_experiment_name: str,
    session_id: str | None = None,
    research_question: str | None = None,
    tested_change: str | None = None,
) -> dict[str, Any]:
    """Read-only: compare two experiments using authoritative artefact metrics.

    Use this to answer 'did performance improve?' or 'what did we learn?'.
    Computes Sharpe / mean OOS Sharpe / max-drawdown deltas from real diagnostics
    — never RAG memory.  Persists a compact comparison_evidence record so future
    memory queries can retrieve before/after conclusions.  No LLM, no execution.

    Optional context fields enrich the evidence record:
      research_question: the question this comparison answers
      tested_change: what was changed (e.g. "added risk_adjusted_momentum_20")
    """
    result = _api.compare_experiment_metrics(
        base_experiment_name,
        candidate_experiment_name,
        session_id=session_id,
        research_question=research_question,
        tested_change=tested_change,
    )
    if result.get("status") == "not_found":
        missing = result.get("missing_experiments", [])
        return _envelope(
            False, "experiment_metrics_unavailable",
            f"Missing metrics artefacts for: {', '.join(missing) or 'unknown'}. "
            "Do not invent metrics.",
            result,
            "list_experiments",
        )
    if result.get("status") == "incomplete_metrics":
        _incomplete = result.get("incomplete_experiments", [])
        return _envelope(
            False, "experiment_metrics_incomplete",
            f"Comparison refused — core metrics are absent in: "
            f"{', '.join(_incomplete)}. "
            f"Details: {result.get('details', '')}. "
            "Do not invent conclusions or evidence. "
            "Use verify_experiment_artifacts to check the artefacts.",
            result,
            "stop_and_report_to_user",
        )
    data = dict(result)
    data["evidence_path"] = result.get("evidence_path")
    return _envelope(
        True, "experiment_metrics_compared", result["conclusion"], data,
        "get_session_summary",
    )


def inspect_comparison_evidence(
    base_experiment_name: str,
    candidate_experiment_name: str,
) -> dict[str, Any]:
    """Read-only: compact summary of a specific comparison evidence record.

    Directly reads the persisted comparison_evidence.json for the given
    base/candidate pair — no LLM, no RAG, no execution.  Returns what was
    tested, metric deltas, failure modes, and the research conclusion.  If no
    record exists, suggests running compare_experiment_metrics first.
    """
    result = _api.inspect_comparison_evidence(
        base_experiment_name, candidate_experiment_name
    )
    if result.get("status") == "not_found":
        return _envelope(
            False, "comparison_evidence_not_found",
            f"No comparison evidence for {base_experiment_name!r} vs "
            f"{candidate_experiment_name!r}. "
            "Run compare_experiment_metrics first to create the record.",
            result,
            "compare_experiment_metrics",
        )

    deltas = result.get("metric_deltas") or {}
    tested = result.get("tested_change") or ""
    base_modes = set(result.get("failure_modes_base") or [])
    cand_modes = result.get("failure_modes_candidate") or []

    delta_parts: list[str] = []
    d_sharpe = deltas.get("delta_sharpe")
    d_oos = deltas.get("delta_mean_oos_sharpe")
    d_dd = deltas.get("delta_max_drawdown_pct")
    if d_sharpe is not None:
        delta_parts.append(f"Sharpe {d_sharpe:+.3f}")
    if d_oos is not None:
        delta_parts.append(f"OOS Sharpe {d_oos:+.3f}")
    if d_dd is not None:
        dd_str = f"MaxDD {d_dd:+.2f}pp" if d_dd != 0.0 else "MaxDD unchanged"
        delta_parts.append(dd_str)

    if cand_modes:
        modes_str = (
            "failure modes persisted"
            if set(cand_modes) & base_modes
            else "new failure modes detected"
        )
    else:
        modes_str = "no failure modes"

    parts = [f"Comparison {base_experiment_name} -> {candidate_experiment_name}"]
    if tested:
        parts.append(f"tested {tested}")
    parts.append(", ".join(delta_parts) if delta_parts else "no numeric deltas")
    parts.append(modes_str)
    display = "; ".join(parts) + "."

    return _envelope(True, "comparison_evidence_inspected", display, result, None)


# ---------------------------------------------------------------------------
# Experiment intelligence tools (read-only; no LLM, no execution)
# ---------------------------------------------------------------------------


def rank_experiments(
    experiment_prefix: str | None = None,
    metric: str = "robustness_score",
    top_k: int | None = None,
    include_failure_modes: bool = True,
) -> dict[str, Any]:
    """Read-only: rank experiment variants by a chosen authoritative metric.

    Reads real artefact metrics for each experiment.  Missing metrics are
    noted per-experiment and never invented.  No LLM, no RAG, no execution.

    Supported metrics: sharpe_ratio, mean_oos_sharpe, max_drawdown_pct,
    robustness_score.  robustness_score is a transparent composite:
    0.5×mean_oos_sharpe + 0.3×sharpe_ratio + 0.2×(max_drawdown_pp/100) minus
    failure-mode penalties.
    """
    result = _api.rank_experiment_variants(
        experiment_prefix=experiment_prefix,
        metric=metric,
        top_k=top_k,
        include_failure_modes=include_failure_modes,
    )
    if result.get("status") != "ok":
        errs = result.get("errors") or [result.get("status", "error")]
        return _envelope(
            False, "rank_experiments_failed",
            f"rank_experiments failed: {errs[0]}",
            result, "list_experiments",
        )
    n = result["returned"]
    prefix_str = f" {experiment_prefix}" if experiment_prefix else ""
    top_item = result["items"][0] if result["items"] else None
    if top_item:
        top_name = top_item["experiment_name"]
        top_score = top_item.get(metric)
        display = (
            f"Ranked {result['total_experiments']}{prefix_str} experiments "
            f"by {metric}. Best: {top_name} ({metric}={top_score}). "
            f"Returning top {n}."
        )
    else:
        display = f"No{prefix_str} experiments found to rank."
    return _envelope(True, "experiments_ranked", display, result, None)


def summarise_experiment_family(
    experiment_prefix: str,
    include_comparisons: bool = True,
    include_intervention_types: bool = True,
) -> dict[str, Any]:
    """Read-only: summarise what has been learned across a family of experiments.

    Aggregates authoritative artefact metrics, failure modes, comparison
    evidence highlights, and deterministic lessons for all experiments whose
    name starts with ``experiment_prefix``.  No LLM, no RAG, no execution.
    """
    result = _api.summarise_experiment_family(
        experiment_prefix=experiment_prefix,
        include_comparisons=include_comparisons,
        include_intervention_types=include_intervention_types,
    )
    if result.get("status") != "ok":
        errs = result.get("errors") or [result.get("status", "error")]
        return _envelope(
            False, "family_summary_not_found",
            f"No experiments found with prefix '{experiment_prefix}': {errs[0]}",
            result, "list_experiments",
        )
    n = result["experiment_count"]
    best_s = result.get("best_sharpe_experiment")
    best_o = result.get("best_oos_sharpe_experiment")
    cfm = result.get("common_failure_modes") or []
    lessons = result.get("lessons_learned") or []
    cfm_str = (", ".join(cfm) or "none") if cfm else "none"
    display_parts = [
        f"Family '{experiment_prefix}': {n} experiment(s). "
        f"Best Sharpe: {best_s or 'n/a'}. "
        f"Best OOS Sharpe: {best_o or 'n/a'}. "
        f"Common failure modes: {cfm_str}."
    ]
    if lessons:
        display_parts.append(f"Key lesson: {lessons[0]}")
    return _envelope(True, "family_summarised", " ".join(display_parts), result, None)


def check_research_idea_history(
    query: str,
    experiment_prefix: str | None = None,
    intervention_type: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Read-only: check whether a similar research idea has already been tested.

    Scans persisted comparison_evidence records and returns relevant matches
    ranked by keyword relevance.  Advisory only — warns when an idea has failed
    or produced mixed results, but does not block the user.  No LLM, no execution.
    """
    result = _api.check_research_idea_history(
        query=query,
        experiment_prefix=experiment_prefix,
        intervention_type=intervention_type,
        top_k=top_k,
    )
    n = result.get("total_matches", 0)
    discouraged = result.get("repeat_discouraged", False)
    if n == 0:
        display = f"No comparison evidence found matching '{query}'. Run compare_experiment_metrics to build evidence."
        return _envelope(True, "idea_history_no_matches", display, result, "compare_experiment_metrics")

    top = result["items"][0]
    top_cand = top.get("candidate_experiment_name") or ""
    top_rec = top.get("recommendation") or ""
    display = (
        f"Found {n} matching comparison(s) for '{query}'. "
        f"Most relevant: {top_cand} (recommendation={top_rec}). "
        + ("Repeating the exact change is not recommended based on prior evidence." if discouraged else
           "Prior evidence is mixed or insufficient — review before repeating.")
    )
    return _envelope(True, "idea_history_found", display, result, None)


def propose_research_plan(
    research_goal: str,
    experiment_prefix: str,
    max_steps: int = 3,
) -> dict[str, Any]:
    """Read-only: create a compact governed research plan from existing evidence.

    Deterministic from authoritative artefact metrics and comparison evidence.
    No LLM, no RAG, no execution.  This tool does **not** generate drafts,
    approve, render, or execute anything.  Each plan step is advisory only.
    Use generate_config_change_draft or generate_parameter_change_draft to
    act on a step.
    """
    result = _api.propose_research_plan(
        research_goal=research_goal,
        experiment_prefix=experiment_prefix,
        max_steps=max_steps,
    )
    if result.get("status") != "ok":
        errs = result.get("errors") or [result.get("status", "error")]
        return _envelope(
            False, "research_plan_failed",
            f"Could not build plan for '{experiment_prefix}': {errs[0]}",
            result, "list_experiments",
        )
    plan = result.get("plan") or []
    n_steps = len(plan)
    cfm = result.get("common_failure_modes") or []
    cfm_str = ", ".join(cfm) or "none detected"
    candidates = [s for s in plan if s.get("status") == "candidate"]
    tested = [s for s in plan if s.get("status") == "already_tested"]
    display = (
        f"Research plan for '{experiment_prefix}' ({result['experiment_count']} experiments). "
        f"Common failure modes: {cfm_str}. "
        f"{n_steps} step(s): {len(candidates)} candidate, {len(tested)} already tested. "
        "Advisory only — no drafts, approval, rendering, or execution."
    )
    return _envelope(True, "research_plan_proposed", display, result, None)


# ---------------------------------------------------------------------------
# Config introspection tools (read-only; no LLM, no execution)
# ---------------------------------------------------------------------------


def inspect_experiment_config(experiment_name: str) -> dict[str, Any]:
    """Read-only: compact summary of the current YAML config for an experiment.

    Returns model type/params, feature count/names, validation and execution
    settings, universe, and the list of changeable field paths.  Does NOT return
    the raw YAML — the payload is always compact.  Executes nothing, calls no LLM,
    reads no arbitrary paths.
    """
    result = _api.inspect_experiment_config(experiment_name)
    if result.get("status") == "config_not_found":
        return _envelope(
            False, "config_not_found",
            f"No config found for {experiment_name!r}. "
            + (result.get("errors") or [""])[0],
            {"experiment_name": experiment_name, "errors": result.get("errors", [])},
            "list_experiments",
        )
    model_type = (result.get("model") or {}).get("type", "unknown")
    feat_count = (result.get("features") or {}).get("count", 0)
    val_type = (result.get("validation") or {}).get("type", "unknown")
    display = (
        f"Config for {experiment_name!r}: model={model_type}, "
        f"{feat_count} feature(s), validation={val_type}."
    )
    return _envelope(
        True, "experiment_config_inspected", display,
        result,
        "list_changeable_config_fields",
    )


def list_changeable_config_fields(
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Read-only: list the controlled config-change surface — paths, operations, types.

    Returns the authoritative set of field paths that generate_config_change_draft
    will accept, with their operations (set / add / remove / replace) and current
    values when experiment_name is given.  Use this before building a draft so you
    know exactly what can and cannot be changed.
    """
    result = _api.list_changeable_config_fields(experiment_name)
    n = len(result.get("fields", []))
    entry_count = result.get("feature_entry_count")
    feat_info = (
        f"; current feature entries: {entry_count}" if entry_count is not None else ""
    )
    display = (
        f"{n} changeable field(s) across model, features, labels, signal, "
        f"validation, execution, portfolio_construction{feat_info}."
    )
    return _envelope(
        True, "changeable_config_fields_listed", display, result,
        "generate_config_change_draft",
    )


def list_available_features(
    experiment_name: str | None = None,
    family: str | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    """Read-only: valid feature types from the authoritative schema.

    Returns schema-validated feature types only — never invents names.  When
    experiment_name is given, shows which types are currently used.  Optional
    family/query filters narrow the list.  Use this to find valid types before
    calling generate_config_change_draft with an add/remove/replace operation.
    """
    result = _api.list_available_features(
        experiment_name=experiment_name, family=family, query=query
    )
    total = result.get("total_types", 0)
    used_count = result.get("currently_used_count")
    used_str = f"; {used_count} currently used" if used_count is not None else ""
    display = f"{total} valid feature type(s) in schema{used_str}."
    return _envelope(
        True, "available_features_listed", display, result,
        "generate_config_change_draft",
    )


def list_supported_models(
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Read-only: model types supported by the quant engine / config schema.

    Returns schema-authoritative model names only — never invents models.
    Model switching IS supported (via generate_config_change_draft with
    field_path='model.type', operation='set').  When experiment_name is given,
    highlights the current model type.
    """
    result = _api.list_supported_models(experiment_name=experiment_name)
    current = result.get("current_model") or "unknown"
    n = len(result.get("supported_models", []))
    switching = result.get("model_switching_supported", False)
    display = (
        f"Current model: {current}. {n} supported model type(s). "
        f"Model switching: {'supported' if switching else 'not supported'}."
    )
    return _envelope(
        True, "supported_models_listed", display, result,
        "generate_config_change_draft" if switching else None,
    )


def generate_config_change_draft(
    experiment_name: str,
    changes: str,
    session_id: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Deterministic multi-change config draft (NO LLM).

    ``changes`` is a JSON array of change dicts.  Supported per-change shapes:

      set:     {"field_path": "model.params.alpha", "operation": "set", "value": 2.0}
      add:     {"field_path": "features.entries", "operation": "add",
                "value": {"name": "...", "type": "...", "params": {...}}}
      remove:  {"field_path": "features.entries", "operation": "remove",
                "value": "feature_name"}
      replace: {"field_path": "features.entries", "operation": "replace",
                "old_value": "old_name", "value": {"name": "...", "type": "...", ...}}

    All changes are validated before any draft is created.  Returns ok=false on
    invalid field paths, unsupported feature types, absent features, duplicate
    adds, or schema-incompatible values — with no fallback invented.  The draft is
    unapproved.  Never approves, renders, executes, or calls an LLM.
    """
    # Parse the JSON changes string — refuse malformed input rather than guessing.
    try:
        parsed_changes: list[dict[str, Any]] = (
            changes if isinstance(changes, list) else __import__("json").loads(changes)
        )
    except Exception as exc:  # noqa: BLE001
        return _envelope(
            False, "config_change_draft_failed",
            f"'changes' must be a JSON array of change objects: {exc}",
            {"error": str(exc)},
            "list_changeable_config_fields",
        )

    result = _api.generate_config_change_draft(
        experiment_name, parsed_changes, reason=reason
    )
    status = result.get("status", "")

    if status != "ok":
        errors = result.get("errors", [])
        err_display = "; ".join(e for e in errors[:3]) if errors else "see errors"
        return _envelope(
            False, "config_change_draft_failed",
            f"Draft failed for {experiment_name!r}: {err_display}",
            {"experiment_name": experiment_name, "status": status, "errors": errors},
            "stop_and_report_to_user",
        )

    draft = result["draft"]
    _record_event(
        session_id,
        SessionEventType.DRAFT_GENERATED,
        experiment_name,
        {
            "draft_id": draft.draft_id,
            "draft_hash": draft.draft_hash,
            "proposed_name": draft.proposed_name,
        },
    )
    diff = _compact_multi_diff(draft.changes)
    diff_lines = _diff_summary_lines(diff)
    data = {
        "experiment_name": experiment_name,
        "draft_id": draft.draft_id,
        "proposed_name": draft.proposed_name,
        "approved": draft.approved,
        "diff": diff,
        "draft_path": _api.draft_artifact_path(experiment_name, draft.draft_id),
    }
    display = (
        f"Draft {draft.draft_id} '{draft.proposed_name}' "
        f"({len(diff)} change(s); approved={draft.approved}).\n  "
        + ("\n  ".join(diff_lines) or "(no changes)")
    )
    return _envelope(True, "config_change_draft_generated", display, data,
                     "validate_experiment_draft")


# ---------------------------------------------------------------------------
# Phase 1 tools
# ---------------------------------------------------------------------------


def list_experiments() -> dict[str, Any]:
    """List experiments that have persisted result artefacts on disk."""
    experiments = _api.list_all_experiments()
    preview = ", ".join(experiments[:8]) + ("..." if len(experiments) > 8 else "")
    display = f"{len(experiments)} experiment(s) available: {preview or '(none)'}"
    return _envelope(
        True, "experiments_listed", display, {"experiments": experiments},
        "create_research_session",
    )


def create_research_session(root_experiment: str, research_goal: str) -> dict[str, Any]:
    """Create a research session anchored to one experiment and a stated goal.

    Requires a config-backed experiment.  If the root_experiment name does not
    correspond to an existing config YAML and looks like an invented new-design
    name, returns a routing redirect instead of creating a session.
    """
    if not _experiment_config_exists(root_experiment) and _looks_like_new_design_session(
        root_experiment, research_goal
    ):
        return _envelope(
            False,
            "wrong_tool_for_new_research_design_request",
            "This looks like a new research-design request. "
            "Do not create a research session for an invented experiment name. "
            "Call route_research_request first, then use "
            "create_validated_draft_from_request.",
            {
                "recommended_tool": "route_research_request",
                "next_recommended_tool": "create_validated_draft_from_request",
                "discouraged_tools": [
                    "create_research_session",
                    "build_context_summary",
                    "run_experiment_review",
                    "generate_iteration_proposal",
                    "generate_experiment_draft",
                    "execute_approved_config",
                ],
                "governance": {
                    "draft_created": False,
                    "draft_approved": False,
                    "execution_performed": False,
                },
            },
            "route_research_request",
        )
    session = _api.create_research_session(
        root_experiment=root_experiment,
        research_goal=research_goal,
    )
    data = {
        "session_id": session.session_id,
        "research_goal": session.research_goal,
        "root_experiment": session.root_experiment,
        "active_experiment": session.active_experiment,
        "status": session.status,
    }
    display = (
        f"Session {session.session_id} created for '{session.root_experiment}' "
        f"· goal: {session.research_goal}"
    )
    return _envelope(True, "session_created", display, data, "build_context_summary")


def get_session_summary(session_id: str) -> dict[str, Any]:
    """Return the current state of a research session (events, draft, config).

    session_id MUST be the UUID returned by create_research_session — NOT an
    experiment name, context_hash, or draft_id.  If the UUID was lost, call
    get_latest_research_session (or list_research_sessions) to recover it.
    """
    session = _api.load_research_session(session_id)
    if session is None:
        return _envelope(
            False, "session_not_found",
            f"Session '{session_id}' not found. session_id must be the UUID from "
            "create_research_session (not an experiment name / hash / draft id). "
            "Call get_latest_research_session to recover the active session.",
            {"error": f"Session '{session_id}' not found"},
            "get_latest_research_session",
        )
    summary = _api.summarize_research_session(session)
    active_draft = summary.get("active_draft")
    active_draft_id = (
        active_draft.get("draft_id") if isinstance(active_draft, dict) else active_draft
    )
    data = {
        "session_id": summary["session_id"],
        "research_goal": _short(summary["research_goal"]),
        "event_count": summary["event_count"],
        "active_experiment": summary["active_experiment"],
        "active_draft": active_draft_id,
        "approved_config_path": summary["approved_config_path"],
        "experiments_visited": summary["experiments_visited"],
        "status": summary["status"],
    }
    display = (
        f"Session {summary['session_id']}: {summary['event_count']} event(s); "
        f"active draft: {active_draft_id or 'none'}; "
        f"approved config: {summary['approved_config_path'] or 'none'}. "
        "Research cycle complete — stop. Do not start another proposal, draft, or "
        "experiment unless the user explicitly asks."
    )
    return _envelope(True, "session_summary", display, data, "stop_cycle_complete")


def list_research_sessions() -> dict[str, Any]:
    """List existing research session IDs (UUIDs). Read-only.

    Use this to recover a session_id you lost. Pass one of these UUIDs (not an
    experiment name) to the session-aware tools.
    """
    ids = _api.list_research_sessions()
    display = f"{len(ids)} research session(s): " + (", ".join(ids[:8]) or "(none)")
    return _envelope(
        True, "sessions_listed", display, {"session_ids": ids}, "get_latest_research_session",
    )


def get_latest_research_session() -> dict[str, Any]:
    """Return the most recently updated research session (UUID + compact summary).

    Read-only recovery helper for when the session_id was lost. The returned
    session_id is the UUID to pass to all session-aware tools — never use an
    experiment name, context_hash, or draft_id as a session_id.
    """
    session = _api.get_latest_research_session()
    if session is None:
        return _envelope(
            False, "no_sessions",
            "No research sessions found. Call create_research_session first.",
            {"session_id": None}, "create_research_session",
        )
    summary = _api.summarize_research_session(session)
    data = {
        "session_id": session.session_id,
        "research_goal": _short(session.research_goal),
        "active_experiment": session.active_experiment,
        "event_count": summary["event_count"],
        "status": session.status,
        "updated_at": session.updated_at,
    }
    display = (
        f"Latest session {session.session_id} (status: {session.status}, "
        f"{summary['event_count']} event(s)). Use this session_id for session-aware tools."
    )
    return _envelope(True, "latest_session", display, data, "check_research_workflow_state")


def _workflow_next_and_display(state: dict[str, Any]) -> tuple[str, str]:
    """Map a workflow-state dict to (next_suggested_action, concise display).

    Advisory only — names the single recommended next tool based on which
    artefacts already exist; it never runs anything.
    """
    name = state["experiment_name"]
    if not state["context_ready"]:
        nxt = "execute_approved_config"
        head = "baseline artefacts missing — execute the baseline config (confirmation='RUN')"
    elif not state["review_exists"]:
        nxt = "run_experiment_review"
        head = "baseline artefacts exist; review missing"
    elif not state["proposal_exists"]:
        nxt = "generate_iteration_proposal"
        head = "review exists; proposal missing"
    elif not state["draft_exists"]:
        nxt = "generate_experiment_draft"
        head = "proposal exists; draft missing"
    elif not state["latest_draft_approved"]:
        nxt = "approve_experiment_draft"
        head = f"draft {state['latest_draft_id']} exists (unapproved) — validate, then approve"
    elif not state["rendered_yaml_exists"]:
        nxt = "render_draft_to_yaml"
        head = "draft approved; YAML not yet rendered"
    elif not state["revised_artefacts_exist"]:
        nxt = "execute_approved_config"
        head = "YAML rendered; revised experiment not yet executed (confirmation='RUN')"
    else:
        nxt = "review_post_run_result"
        head = "revised artefacts exist; ready for post-run review"
    return nxt, f"[{name}] {head}. Next suggested action: {nxt}."


def check_research_workflow_state(experiment_name: str) -> dict[str, Any]:
    """Read-only preflight: report which research artefacts exist and the next step.

    Inspects on-disk state only (baseline artefacts incl. metadata.json/
    metrics.json, LLM review, iteration proposal, latest draft + approval,
    rendered YAML, revised run artefacts, report/plots paths).  Executes nothing,
    calls no LLM, and touches no quant-engine code.  Call this before major
    transitions so the workflow is never run out of order.
    """
    state = _api.get_research_workflow_state(experiment_name)
    # The next-tool/display logic used to be duplicated here as
    # _workflow_next_and_display(); it now lives once, in research_api, as
    # next_tool_and_display() — the same function Workflow Trace's phase
    # derivation (_api.compute_workflow_trace()) derives from, so there is
    # exactly one authoritative workflow-progression definition.
    next_action, display = _api.next_tool_and_display(state)
    # Compact subset — flags/paths/ids only, no full filesystem inspection.
    data = {
        "experiment_name": state["experiment_name"],
        "baseline_artefacts_exist": state["baseline_artefacts_exist"],
        "context_ready": state["context_ready"],
        "review_exists": state["review_exists"],
        "proposal_exists": state["proposal_exists"],
        "draft_exists": state["draft_exists"],
        "latest_draft_id": state["latest_draft_id"],
        "latest_draft_approved": state["latest_draft_approved"],
        "rendered_yaml_exists": state["rendered_yaml_exists"],
        "rendered_yaml_path": state["rendered_yaml_path"],
        "revised_artefacts_exist": state["revised_artefacts_exist"],
        "report_path": state["report_path"],
        "plots_dir": state["plots_dir"],
    }
    return _envelope(True, "workflow_state_checked", display, data, next_action)


def get_workflow_trace(session_id: str, experiment_name: str | None = None) -> dict[str, Any]:
    """Read-only: who owns this research workflow right now, what phase it's
    in, and what's next — derived from the session's own event log
    (record_session_event) and the same artefact-existence state
    check_research_workflow_state already inspects. Executes nothing, calls
    no LLM, and touches no quant-engine code; introduces no new persisted
    state.

    experiment_name defaults to the session's active_experiment; pass it
    explicitly for a session that has touched more than one experiment.

    'execution' is always null here — a live execution can only be seen
    synchronously, from inside execute_approved_config's own response, while
    that call is in flight; this tool cannot poll for it separately.
    """
    try:
        trace = _api.compute_workflow_trace(session_id, experiment_name=experiment_name)
    except ValueError as exc:
        return _envelope(
            False, "workflow_trace_session_not_found", str(exc),
            {"session_id": session_id, "error": str(exc)},
            "create_research_session",
        )

    data = trace.to_dict()
    if trace.current_phase == "idle":
        display = f"[{trace.workflow_id}] Idle — no active experiment for this session."
        next_action = "no_active_experiment"
    else:
        display = (
            f"[{trace.experiment_name}] Owner: {trace.current_owner}. "
            f"Phase: {trace.current_phase}. Status: {trace.status}. "
            f"Next: {trace.current_tool}."
        )
        next_action = trace.current_tool
    return _envelope(True, "workflow_trace", display, data, next_action)


def build_context_summary(experiment_name: str) -> dict[str, Any]:
    """Assemble deterministic structured context for an experiment (no LLM call).

    Returns a compact view: context hash, detected failure modes, and the
    performance/validation blocks.  No raw tables or large payloads.

    Requires an existing config-backed experiment.  For new research-design
    requests use create_validated_draft_from_request instead.
    """
    if not _experiment_config_exists(experiment_name):
        return _wrong_tool_new_design_redirect(experiment_name, "build_context_summary")
    ctx = _api.build_llm_context(experiment_name, persist=False)
    context_hash = _api.compute_context_hash(ctx)
    failure_modes = _name_sev(ctx.failure_modes)
    perf = ctx.performance or {}
    val = ctx.validation or {}
    key_metrics = {
        "sharpe_ratio": perf.get("sharpe_ratio"),
        "max_drawdown_pct": perf.get("max_drawdown_pct"),
        "mean_oos_sharpe": val.get("mean_oos_sharpe"),
        "std_oos_sharpe": val.get("std_oos_sharpe"),
        "n_splits": val.get("n_splits"),
        "n_negative_sharpe_splits": val.get("n_negative_sharpe_splits"),
    }
    data = {
        "experiment_name": ctx.experiment_name,
        "context_hash": context_hash,
        "failure_modes": failure_modes,
        "key_metrics": key_metrics,
    }
    display = (
        f"Context for {ctx.experiment_name} built (hash {_short_hash(context_hash)}). "
        f"Failure modes: {_fmt_failure_modes(failure_modes)}. "
        f"Validation: {_fmt_validation(ctx.validation)}."
    )
    return _envelope(True, "context_built", display, data, "run_experiment_review")


def run_experiment_review(
    experiment_name: str,
    session_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Run an LLM review of an experiment's persisted diagnostics.

    Advisory only — interprets existing artefacts; computes nothing new and
    runs no experiment.  Records REVIEW_GENERATED when a session_id is given.

    Requires an existing config-backed experiment.  For new research-design
    requests use create_validated_draft_from_request instead.
    """
    if not _experiment_config_exists(experiment_name):
        return _wrong_tool_new_design_redirect(experiment_name, "run_experiment_review")
    _s = resolve_llm_settings(provider, base_url, model)
    return _review_experiment(
        experiment_name,
        session_id,
        _s.provider,
        _s.model,
        _s.base_url,
        SessionEventType.REVIEW_GENERATED,
        "review_generated",
        "Review",
        "generate_iteration_proposal",
    )


def generate_iteration_proposal(
    experiment_name: str,
    session_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Generate an advisory next-step research proposal grounded in diagnostics.

    Does not prescribe parameter values and runs no experiment.  Records
    ITERATION_PROPOSAL_GENERATED when a session_id is given.

    Requires an existing config-backed experiment.  For new research-design
    requests use create_validated_draft_from_request instead.
    """
    if not _experiment_config_exists(experiment_name):
        return _wrong_tool_new_design_redirect(experiment_name, "generate_iteration_proposal")
    _s = resolve_llm_settings(provider, base_url, model)
    proposal = _api.generate_iteration_proposal(
        experiment_name,
        provider=_s.provider,
        model=_s.model,
        base_url=_s.base_url,
        persist=True,
    )
    _record_event(
        session_id,
        SessionEventType.ITERATION_PROPOSAL_GENERATED,
        experiment_name,
        {"context_hash": proposal.context_hash, "research_focus": proposal.research_focus},
    )
    # Compact: short focus/confidence/path + at most 3 short concerns.
    concerns = [_short(c, 160) for c in (proposal.validation_concerns or [])[:3]]
    data = {
        "experiment_name": experiment_name,
        "context_hash": proposal.context_hash,
        "research_focus": _short(proposal.research_focus),
        "confidence": proposal.confidence,
        "validation_concerns": concerns,
        "proposal_path": _api.proposal_artifact_path(experiment_name),
    }
    display = (
        f"Proposal for {experiment_name} (hash {_short_hash(proposal.context_hash)}): "
        f"{_short(proposal.research_focus) or '(none)'} (confidence: {proposal.confidence or 'n/a'})"
    )
    return _envelope(True, "proposal_generated", display, data, "generate_experiment_draft")


def generate_experiment_draft(
    experiment_name: str,
    session_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Synthesise a schema-bounded config draft (typed deltas, NOT YAML).

    Requires a prior persisted iteration proposal.  The draft is unapproved
    (approved=false) and must be validated and explicitly approved before it can
    be rendered.  Records DRAFT_GENERATED when a session_id is given.

    With provider='stub' (default when no env var set) the draft is deterministic
    (model.params.alpha -> 1.0) and no LLM is called.  On any failure this
    returns an ok=false envelope with the error — never a partial or invented draft.
    """
    _s = resolve_llm_settings(provider, base_url, model)
    try:
        draft = _api.generate_experiment_draft(
            experiment_name,
            provider=_s.provider,
            model=_s.model,
            base_url=_s.base_url,
        )
    except Exception as exc:  # noqa: BLE001 — surface a clean envelope, never a traceback
        if not _experiment_config_exists(experiment_name):
            return _envelope(
                False, "wrong_tool_for_research_design_request",
                "This looks like a new research-design request, but "
                "generate_experiment_draft requires an existing config-backed "
                "experiment. Use create_validated_draft_from_request instead.",
                {
                    "experiment_name": experiment_name,
                    "recommended_tool": "create_validated_draft_from_request",
                    "governance": {
                        "draft_created": False,
                        "draft_approved": False,
                        "execution_performed": False,
                    },
                },
                "create_validated_draft_from_request",
            )
        return _envelope(
            False, "draft_generation_failed",
            f"Draft generation failed: {type(exc).__name__}: {exc}. "
            "Do not invent a config — report this and stop.",
            {"error": f"{type(exc).__name__}: {exc}"},
            "generate_iteration_proposal",
        )
    _record_event(
        session_id,
        SessionEventType.DRAFT_GENERATED,
        experiment_name,
        {
            "draft_id": draft.draft_id,
            "draft_hash": draft.draft_hash,
            "proposed_name": draft.proposed_name,
        },
    )
    diff = _compact_diff(draft.changes)
    diff_lines = [
        f"{d['field']}: {d['current_value']} -> {d['proposed_value']}" for d in diff
    ]
    data = {
        "experiment_name": experiment_name,
        "draft_id": draft.draft_id,
        "proposed_name": draft.proposed_name,
        "approved": draft.approved,
        "diff": diff,
        "draft_path": _api.draft_artifact_path(experiment_name, draft.draft_id),
    }
    display = (
        f"Draft {draft.draft_id} '{draft.proposed_name}' (approved={draft.approved}). "
        f"Proposed config diff:\n  " + ("\n  ".join(diff_lines) or "(no changes)")
    )
    return _envelope(True, "draft_generated", display, data, "validate_experiment_draft")


def validate_experiment_draft(
    experiment_name: str,
    draft_id: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Validate a draft against the authoritative config schema (no side effects).

    Records DRAFT_VALIDATED when a session_id is given.
    """
    draft = _api.load_experiment_draft(experiment_name, draft_id)
    if draft is None:
        return _envelope(
            False, "draft_not_found",
            f"Draft '{draft_id}' not found for experiment '{experiment_name}'.",
            {"error": f"Draft '{draft_id}' not found for experiment '{experiment_name}'"},
            "generate_experiment_draft",
        )
    result = _api.validate_experiment_draft(draft)
    _record_event(
        session_id,
        SessionEventType.DRAFT_VALIDATED,
        experiment_name,
        {"draft_id": draft_id, "is_valid": result.is_valid, "error_count": len(result.errors)},
    )
    errors = [_short(e, 160) for e in result.errors[:3]]
    # A duplicate proposed name is the one recoverable failure — it has a clean
    # resolution (a new suffix or cleaning demo artefacts).  Every other failure
    # is non-recoverable here and the model must stop, not regenerate.
    duplicate_name = (not result.is_valid) and any(
        "already exists" in e for e in result.errors
    )
    data = {
        "experiment_name": experiment_name,
        "draft_id": draft_id,
        "is_valid": result.is_valid,
        "rendering_blocked": not result.is_valid,
        "duplicate_proposed_name": duplicate_name,
        "recoverable": duplicate_name,
        "error_count": len(result.errors),
        "errors": errors,
    }
    if result.is_valid:
        # Governance boundary: validation never authorises approval. Stop here and
        # hand control back to the user — do NOT auto-advance to approval.
        display = (
            "Validation PASS — approval still required. Stop and ask the user "
            "before approving or rendering."
        )
        next_action = "ask_user_for_approval"
    elif duplicate_name:
        # Recoverable, but still requires the user to choose — never auto-loop.
        display = (
            "Validation FAILED — proposed experiment name already exists; "
            "rendering is BLOCKED. Stop and ask the user whether to use the next "
            "available name suffix or clean existing local demo artefacts. Do not "
            "regenerate repeatedly."
        )
        next_action = "ask_user_resolve_duplicate_name"
    else:
        # Non-recoverable: stop and report, do NOT regenerate the same draft.
        display = (
            "Validation FAILED — rendering is BLOCKED. Stop and ask the user how "
            "to proceed. Do not regenerate repeatedly. Errors: " + "; ".join(errors)
        )
        next_action = "stop_and_report_to_user"
    return _envelope(result.is_valid, "draft_validated", display, data, next_action)


def approve_experiment_draft(
    experiment_name: str,
    draft_id: str,
    approval_confirmation: str = "",
    session_id: str | None = None,
    render_requested: bool = False,
) -> dict[str, Any]:
    """Explicitly approve a draft so it can be rendered to YAML.

    Requires approval_confirmation='APPROVE'. Approval is performed only by this
    tool — no other tool auto-approves — and only when the latest user message
    explicitly approves the draft.  Records DRAFT_APPROVED when a session_id is
    given.

    Call this ONLY when the latest user instruction explicitly approves the draft.
    Never infer approval from 'start a research cycle', from validation success,
    or from a next_suggested_action; never approve automatically after validation.

    render_requested: set True ONLY when the user explicitly asked to approve AND
    render in the same message. It does not render anything here; it only controls
    whether the suggested next step is render_draft_to_yaml (when True) or
    ask_user_to_render_yaml (when False — the default, which stops for the user).
    """
    if approval_confirmation != "APPROVE":
        # Tool-enforced approval gate: refuse before loading/mutating any state.
        # A refusal must NOT loop back into this tool — stop and wait for the user.
        return _envelope(
            False, "approval_refused",
            "Approval refused — confirmation='APPROVE' is required. Stop. Ask the "
            "user to explicitly approve before calling approve_experiment_draft again.",
            {"approved": False, "error": "Approval requires approval_confirmation='APPROVE'."},
            "ask_user_for_approval",
        )
    draft = _api.load_experiment_draft(experiment_name, draft_id)
    if draft is None:
        return _envelope(
            False, "draft_not_found",
            f"Draft '{draft_id}' not found for experiment '{experiment_name}'.",
            {"error": f"Draft '{draft_id}' not found for experiment '{experiment_name}'"},
            "generate_experiment_draft",
        )
    approved = _api.approve_experiment_draft(draft)
    _record_event(
        session_id,
        SessionEventType.DRAFT_APPROVED,
        experiment_name,
        {"draft_id": approved.draft_id, "draft_hash": approved.draft_hash},
    )
    data = {
        "experiment_name": experiment_name,
        "draft_id": approved.draft_id,
        "approved": approved.approved,
        "approved_at": approved.approved_at,
        "draft_hash": approved.draft_hash,
        "render_requested": render_requested,
    }
    if render_requested:
        display = (
            f"Draft {approved.draft_id} approved at {approved.approved_at}. "
            "User requested approve+render in the same message — rendering may proceed."
        )
        next_action = "render_draft_to_yaml"
    else:
        display = (
            f"Draft {approved.draft_id} approved at {approved.approved_at}. "
            "Stop and ask the user before rendering YAML."
        )
        next_action = "ask_user_to_render_yaml"
    return _envelope(True, "draft_approved", display, data, next_action)


def render_draft_to_yaml(
    experiment_name: str,
    draft_id: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Render an APPROVED draft to a YAML config file. Does NOT execute it.

    Returns a clear error if the draft is not approved.  Execution requires a
    separate, explicitly confirmed tool call.  Records YAML_RENDERED when a
    session_id is given.
    """
    draft = _api.load_experiment_draft(experiment_name, draft_id)
    if draft is None:
        return _envelope(
            False, "draft_not_found",
            f"Draft '{draft_id}' not found for experiment '{experiment_name}'.",
            {"error": f"Draft '{draft_id}' not found for experiment '{experiment_name}'"},
            "generate_experiment_draft",
        )
    if not draft.approved:
        return _envelope(
            False, "render_blocked",
            f"Draft '{draft_id}' must be approved before rendering to YAML.",
            {"error": f"Draft '{draft_id}' must be approved before rendering to YAML"},
            "approve_experiment_draft",
        )

    # Render (and persist) the YAML on disk; do NOT return its contents.
    _api.render_draft_to_yaml(draft)
    config_path = f"configs/experiments/{draft.proposed_name}.yaml"
    _record_event(
        session_id,
        SessionEventType.YAML_RENDERED,
        experiment_name,
        {"draft_id": draft.draft_id, "config_path": config_path},
    )
    data = {
        "experiment_name": experiment_name,
        "draft_id": draft.draft_id,
        "config_path": config_path,
        "draft_hash": draft.draft_hash,
        "execution_has_occurred": False,
    }
    display = (
        f"Rendered {config_path}. Execution has NOT occurred. Stop. Execute only "
        "after the user explicitly says RUN."
    )
    return _envelope(
        True, "yaml_rendered", display, data, "ask_user_for_execution_authorisation"
    )


# ---------------------------------------------------------------------------
# Phase 2 tools (optional execution)
# ---------------------------------------------------------------------------


def verify_experiment_artifacts(experiment_name: str) -> dict[str, Any]:
    """Read-only: verify artefact completeness after execution.

    Checks artefact_root, metrics.json (existence + parseability), required
    metrics, report, and plots.  Returns artefact_status='complete' or
    'incomplete'.  No LLM, no execution.  Call this after execute_approved_config
    and before review_post_run_result to confirm the engine produced valid artefacts.

    If incomplete: stop and report to the user. Do not compare, review, index,
    or invent metrics for an incomplete experiment.
    """
    result = _api.verify_experiment_artifacts(experiment_name)
    artefact_status = result.get("artefact_status", "incomplete")
    missing = result.get("missing_metrics") or []

    if artefact_status == "incomplete":
        problems: list[str] = []
        if not result.get("artefact_root_exists"):
            problems.append("artefact root missing")
        if not result.get("metrics_path_exists"):
            problems.append("metrics.json missing")
        elif not result.get("metrics_parseable"):
            problems.append("metrics.json malformed")
        elif missing:
            problems.append(f"core metrics absent: {', '.join(missing)}")
        display = (
            f"Artefacts for '{experiment_name}' are INCOMPLETE: "
            f"{'; '.join(problems) or 'unknown issue'}. "
            "Execution completed but authoritative metrics are missing. "
            "Do not compare, review, index, or invent metrics for this experiment. "
            "Stop and inspect the artefacts."
        )
        return _envelope(
            False, "artefact_verification_incomplete", display, result,
            "stop_and_report_to_user",
        )

    report_flag = "present" if result.get("report_path_exists") else "absent"
    plots_flag = "present" if result.get("plots_dir_exists") else "absent"
    display = (
        f"Artefacts for '{experiment_name}' are complete: "
        f"metrics present, report {report_flag}, plots {plots_flag}. "
        "Proceed to review_post_run_result."
    )
    return _envelope(
        True, "artefact_verification_complete", display, result,
        "review_post_run_result",
    )


def execute_approved_config(
    config_path: str,
    session_id: str | None = None,
    confirmation: str = "",
    report: bool = True,
    preset: str = "canonical",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run exactly one approved YAML config through the quant research engine.

    Requires confirmation='RUN'. This tool does not optimise, loop, retry,
    trade, or register lineage automatically. It runs a single config and
    returns the result. With dry_run=true it only reports the planned command
    and records nothing.

    Call this ONLY when the latest user instruction explicitly contains the
    literal token RUN. Never infer RUN from a previous approval or render, from
    "execute"/"yes"/"proceed"/"continue", or from a prior failed/refused attempt.
    """
    if _is_placeholder_path(config_path):
        return _envelope(
            False, "invalid_execution_request",
            f"Execution rejected — '{config_path}' is a placeholder path, not a real "
            "config file. Provide the actual rendered YAML path from render_draft_to_yaml. "
            "Do not invent or guess config paths.",
            {
                "success": False,
                "config_path": config_path,
                "error": "Placeholder or empty config_path. Use the path returned by render_draft_to_yaml.",
                "execution_performed": False,
                "draft_approved": False,
            },
            "render_draft_to_yaml",
        )

    if confirmation != "RUN":
        # Governance boundary: a refusal must NOT loop back into this tool. Stop
        # and wait for a fresh user message that explicitly contains RUN.
        return _envelope(
            False, "execution_refused",
            "Execution refused — confirmation='RUN' is required. Stop. Ask the "
            "user to explicitly type RUN before calling execute_approved_config again.",
            {"success": False, "error": "Execution requires confirmation='RUN'."},
            "ask_user_for_execution_authorisation",
        )

    if dry_run:
        execution = _api.execute_approved_config(
            config_path, report=report, preset=preset, dry_run=True
        )
        data = {
            "success": False,
            "planned": True,
            "config_path": config_path,
            "command_hint": execution.command_hint,
        }
        display = f"Planned (dry run, NOT executed): {execution.command_hint}"
        return _envelope(True, "execution_planned", display, data, "execute_approved_config")

    _record_event(
        session_id,
        SessionEventType.EXECUTION_REQUESTED,
        config_path,
        {"config_path": config_path, "preset": preset, "report": report},
    )

    execution = _api.execute_approved_config(
        config_path, report=report, preset=preset, dry_run=False
    )

    if not execution.success:
        data = {"success": False, "config_path": config_path, "error": execution.error}
        return _envelope(
            False, "execution_failed",
            f"Execution failed: {execution.error}", data, "execute_approved_config",
        )

    _record_event(
        session_id,
        SessionEventType.EXECUTION_COMPLETED,
        execution.experiment_name or config_path,
        {
            "config_path": execution.config_path,
            "experiment_name": execution.experiment_name,
            "artefact_root": execution.artefact_root,
            "report_path": execution.report_path,
        },
    )
    plots = f"{execution.artefact_root}/plots" if execution.artefact_root else None
    data = {
        "config_path": execution.config_path,
        "success": True,
        "experiment_name": execution.experiment_name,
        "artefact_root": execution.artefact_root,
        "report_path": execution.report_path,
        "plots_dir": plots,
        "stage_executions": execution.stage_executions,
        "total_duration_s": execution.total_duration_s,
    }
    stage_summary = _stage_summary(execution.stage_executions)
    display = (
        f"Executed one config -> experiment '{execution.experiment_name}'. "
        f"Artefacts: {execution.artefact_root}. (single run; no loop, no retry)"
        + (
            f" Stage timings: {stage_summary} (total {execution.total_duration_s:.1f}s)."
            if stage_summary
            else ""
        )
    )
    return _envelope(True, "execution_completed", display, data, "verify_experiment_artifacts")


def review_post_run_result(
    experiment_name: str,
    session_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Run an LLM review of the artefacts produced by a freshly executed config.

    Advisory only. Records POST_RUN_REVIEW_GENERATED when a session_id is given.
    """
    _s = resolve_llm_settings(provider, base_url, model)
    return _review_experiment(
        experiment_name,
        session_id,
        _s.provider,
        _s.model,
        _s.base_url,
        SessionEventType.POST_RUN_REVIEW_GENERATED,
        "post_run_review_generated",
        "Post-run review",
        "get_session_summary",
    )


# ---------------------------------------------------------------------------
# Tool registration (explicit, safety-aware descriptions)
# ---------------------------------------------------------------------------

# Provider guidance for the LLM-backed tools. Shown to the model so it asks the
# user for the right LM Studio parameters and never requests a real OpenAI key.
_LLM_HINT = (
    " To use a local LM Studio model pass provider='openai', "
    "model='<exact LM Studio model id>', base_url='http://127.0.0.1:1234/v1' "
    "(when base_url is set, no real OpenAI API key is needed). "
    "provider='stub' gives a deterministic offline result with no model."
)

# Appended to every session-aware tool so the model passes the right identifier.
_SESSION_HINT = (
    " Pass session_id — the UUID returned by create_research_session — when "
    "available so this action is recorded in the session. Never use an "
    "experiment name, context_hash, or draft_id as session_id; if the UUID was "
    "lost, call get_latest_research_session to recover it."
)

# ---------------------------------------------------------------------------
# Agent workflow tools (Phase 3C) — deterministic spec → plan → draft
# ---------------------------------------------------------------------------

_AGENT_WORKFLOW_FIELDS = frozenset({
    "asset_classes", "universes", "models", "features", "signals", "labels",
    "validation", "portfolio_construction", "diagnostics", "presets",
    "defaults", "change_paths",
})

_HARD_UNSUPPORTED_NOTE = (
    "Hard-unsupported fields (universe, tickers, frequency, data_source, "
    "asset_class) block draft creation — use a supported preset instead."
)
_SOFT_UNSUPPORTED_NOTE = (
    "Soft-unsupported fields (start_date, end_date) are recorded in "
    "result.unsupported_fields but do not block draft creation."
)


def _mcp_list_research_agent_options(
    field: str | None = None,
    asset_class: str | None = None,
) -> dict[str, Any]:
    """List valid options from the research config registry."""
    try:
        registry = _load_registry()
    except Exception as exc:
        return _envelope(
            False,
            "research_agent_options_error",
            f"Failed to load registry: {exc}",
            {"error": str(exc)},
            "list_research_agent_options",
        )

    if field is None:
        data: dict[str, Any] = {
            "categories": {
                "models": sorted(registry.model_names()),
                "features": sorted(registry.feature_names()),
                "signals": sorted(registry.signal_names()),
                "labels": sorted(registry.label_names()),
                "universes": sorted(registry.universe_names()),
                "asset_classes": sorted(registry.asset_class_names()),
                "validation": sorted(registry.validation_names()),
                "weighting_schemes": sorted(registry.weighting_scheme_names()),
                "prediction_normalizations": sorted(
                    registry.prediction_normalization_names()
                ),
                "diagnostics": sorted(registry.diagnostic_names()),
                "presets": sorted(registry.preset_names()),
                "change_paths": sorted(registry.changeable_paths()),
            },
            "counts": {
                "models": len(registry.model_names()),
                "features": len(registry.feature_names()),
                "signals": len(registry.signal_names()),
                "labels": len(registry.label_names()),
                "universes": len(registry.universe_names()),
                "asset_classes": len(registry.asset_class_names()),
                "presets": len(registry.preset_names()),
                "change_paths": len(registry.changeable_paths()),
            },
            "usage": (
                "Pass field='models' (or other category name) to see detailed "
                "options. Supported fields: "
                + ", ".join(sorted(_AGENT_WORKFLOW_FIELDS))
            ),
        }
        display = (
            f"Registry overview: {len(registry.model_names())} models, "
            f"{len(registry.feature_names())} features, "
            f"{len(registry.preset_names())} presets, "
            f"{len(registry.signal_names())} signals."
        )
        return _envelope(
            True, "research_agent_options", display, data,
            "resolve_research_spec",
        )

    field_lower = field.lower().strip()
    if field_lower not in _AGENT_WORKFLOW_FIELDS:
        return _envelope(
            False,
            "research_agent_options_invalid_field",
            f"Unknown field '{field}'. Supported: {sorted(_AGENT_WORKFLOW_FIELDS)}",
            {"supported_fields": sorted(_AGENT_WORKFLOW_FIELDS), "requested": field},
            "list_research_agent_options",
        )

    if field_lower == "models":
        options = [
            {
                "name": e.name,
                "display_name": e.display_name,
                "description": e.description,
                "supported": e.supported,
                "default_params": e.default_params,
                "tunable_params": e.tunable_params,
            }
            for e in registry._models.values()
        ]
    elif field_lower == "features":
        options = [
            {
                "name": e.name,
                "display_name": e.display_name,
                "family": e.family,
                "description": e.description,
                "required_params": sorted(e.required_params),
                "default_params": e.default_params,
            }
            for e in registry._features.values()
        ]
    elif field_lower == "signals":
        options = [
            {
                "name": e.name,
                "display_name": e.display_name,
                "description": e.description,
                "required_params": sorted(e.required_params),
                "default_params": e.default_params,
                "requires_multi_asset": e.requires_multi_asset,
            }
            for e in registry._signals.values()
        ]
    elif field_lower == "labels":
        options = [
            {
                "name": e.name,
                "display_name": e.display_name,
                "description": e.description,
                "required_params": sorted(e.required_params),
                "default_params": e.default_params,
            }
            for e in registry._labels.values()
        ]
    elif field_lower == "universes":
        entries = registry._universes.values()
        if asset_class:
            entries = [e for e in entries if e.asset_class == asset_class]
        options = [
            {
                "name": e.name,
                "display_name": e.display_name,
                "asset_class": e.asset_class,
                "frequency": e.frequency,
                "ticker_count": len(e.tickers),
                "description": e.description,
            }
            for e in entries
        ]
    elif field_lower == "asset_classes":
        options = [
            {
                "name": e.name,
                "display_name": e.display_name,
                "description": e.description,
            }
            for e in registry._asset_classes.values()
        ]
    elif field_lower == "validation":
        options = [
            {
                "name": e.name,
                "display_name": e.display_name,
                "description": e.description,
                "required_params": sorted(e.required_params),
                "default_params": e.default_params,
                "minimum_data_guidance": e.minimum_data_guidance,
            }
            for e in registry._validations.values()
        ]
    elif field_lower == "portfolio_construction":
        options = {
            "weighting_schemes": [
                {
                    "name": e.name,
                    "display_name": e.display_name,
                    "description": e.description,
                    "required_params": sorted(e.required_params),
                    "default_params": e.default_params,
                }
                for e in registry._weighting_schemes.values()
            ],
            "prediction_normalizations": [
                {
                    "name": e.name,
                    "display_name": e.display_name,
                    "description": e.description,
                }
                for e in registry._prediction_normalizations.values()
            ],
        }
        return _envelope(
            True,
            "research_agent_options",
            f"Portfolio construction options: "
            f"{len(registry._weighting_schemes)} weighting schemes, "
            f"{len(registry._prediction_normalizations)} normalization options.",
            {"field": field_lower, "options": options},
            "resolve_research_spec",
        )
    elif field_lower == "diagnostics":
        options = [
            {
                "name": e.name,
                "display_name": e.display_name,
                "description": e.description,
                "required_for_complete_artefact": e.required_for_complete_artefact,
            }
            for e in registry._diagnostics.values()
        ]
    elif field_lower == "presets":
        options = [
            {
                "name": e.name,
                "display_name": e.display_name,
                "description": e.description,
                "asset_class": e.asset_class,
                "universe": e.universe,
                "frequency": e.frequency,
                "benchmark": e.benchmark,
                "notes": e.notes,
            }
            for e in registry._presets.values()
        ]
    elif field_lower == "defaults":
        policy = registry._default_policy
        options = {
            "default_mode": policy.default_mode if policy else "recommended",
            "field_policies": policy.field_policies if policy else {},
        }
        return _envelope(
            True,
            "research_agent_options",
            f"Default policy: mode={policy.default_mode if policy else 'recommended'}.",
            {"field": field_lower, "options": options},
            "resolve_research_spec",
        )
    elif field_lower == "change_paths":
        options = [
            {
                "dotted_path": e.dotted_path,
                "section": e.section,
                "field": e.field,
                "description": e.description,
            }
            for e in registry.change_path_entries()
        ]
    else:
        options = []

    n = len(options) if isinstance(options, list) else len(options)
    display = f"Field '{field_lower}': {n} option(s) available."
    return _envelope(
        True,
        "research_agent_options",
        display,
        {"field": field_lower, "options": options},
        "resolve_research_spec",
    )


def _mcp_resolve_research_spec(
    spec: str,
    mode: str = "recommended",
) -> dict[str, Any]:
    """Resolve a structured ResearchSpec dict (as JSON string) using registry defaults."""
    import json

    try:
        spec_dict: dict[str, Any] = json.loads(spec)
    except (json.JSONDecodeError, ValueError) as exc:
        return _envelope(
            False,
            "agent_resolution_invalid_input",
            f"spec must be a valid JSON object string: {exc}",
            {"parse_error": str(exc), "hint": "Pass spec as a JSON object string, e.g. '{\"preset\": \"canonical_multi_asset_etf\"}'"},
            "list_research_agent_options",
        )

    result = _agent_workflow.run_resolve_spec(spec_dict, mode=mode)
    d = result.to_dict()
    ok = result.status == "resolved"
    display = _short(
        f"Resolution [{result.status}]: {result.summary}", 160
    )
    next_action = (
        "create_experiment_plan_from_spec"
        if ok
        else "list_research_agent_options"
    )
    return _envelope(ok, f"agent_resolution_{result.status}", display, d, next_action)


def _mcp_create_experiment_plan_from_spec(
    spec: str,
    mode: str = "recommended",
    baseline: str | None = None,
) -> dict[str, Any]:
    """Resolve a ResearchSpec dict and create an ExperimentPlan (no draft, no execution)."""
    import json

    try:
        spec_dict: dict[str, Any] = json.loads(spec)
    except (json.JSONDecodeError, ValueError) as exc:
        return _envelope(
            False,
            "agent_planning_invalid_input",
            f"spec must be a valid JSON object string: {exc}",
            {"parse_error": str(exc)},
            "list_research_agent_options",
        )

    result = _agent_workflow.run_plan_from_spec(spec_dict, mode=mode, baseline=baseline)
    d = result.to_dict()
    ok = result.status in ("planned", "needs_review")
    display = _short(
        f"Plan [{result.status}]: {result.summary}", 160
    )
    next_action = (
        "create_validated_draft_from_spec"
        if ok
        else "list_research_agent_options"
    )
    return _envelope(ok, f"agent_planning_{result.status}", display, d, next_action)


def _mcp_create_validated_draft_from_spec(
    spec: str,
    mode: str = "recommended",
    baseline: str | None = None,
    base_experiment: str | None = None,
    save_draft: bool = False,
) -> dict[str, Any]:
    """Run the full deterministic chain: spec → plan → validated draft preview.

    The draft is validated through the authoritative schema validator.
    It is NOT approved and NOT executed.  Approval is a separate explicit step.

    Governance guarantees (always present in data payload):
        approval_required:   true
        execution_performed: false
        draft_approved:      false

    Hard-unsupported fields (universe, tickers, frequency, data_source, asset_class)
    block draft creation.

    Soft-unsupported fields (start_date, end_date) are reported in
    result.unsupported_fields but do not block a draft_created result.
    """
    import json

    try:
        spec_dict: dict[str, Any] = json.loads(spec)
    except (json.JSONDecodeError, ValueError) as exc:
        return _envelope(
            False,
            "agent_draft_invalid_input",
            f"spec must be a valid JSON object string: {exc}",
            {"parse_error": str(exc)},
            "list_research_agent_options",
        )

    result = _agent_workflow.run_draft_from_spec(
        spec_dict,
        mode=mode,
        baseline=baseline,
        base_experiment=base_experiment,
    )
    d = result.to_dict()

    ok = result.status == "draft_created"
    display_parts = [f"Draft [{result.status}]: {_short(result.summary, 120)}"]
    if result.unsupported_fields:
        display_parts.append(
            f"Unsupported fields (not applied): {result.unsupported_fields}."
        )
    if result.hard_unsupported_fields:
        display_parts.append(
            f"BLOCKED by hard-unsupported: {result.hard_unsupported_fields}."
        )

    if save_draft:
        persist = _save_agent_draft(result)
        d["saved_draft"] = persist.to_dict()
        if persist.status == "saved":
            display_parts.append(
                f"Draft saved: id={persist.draft_id} path={persist.draft_path}."
            )
        else:
            display_parts.append(f"Draft not saved: {_short(persist.summary, 80)}.")

    display = " ".join(display_parts)

    next_action = (
        "approve_experiment_draft"
        if ok
        else "list_research_agent_options"
    )
    return _envelope(ok, f"agent_draft_{result.status}", display, d, next_action)


def _mcp_parse_research_intent(
    request: str,
    provider: str | None = None,
    mode: str = "recommended",
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Parse a natural-language research request into a structured ResearchSpec.

    The parser is registry-constrained: only supported registry values are
    allowed in the ResearchSpec.  Unsupported values go to unrecognised_terms.

    Governance:
        no_plan_created:       true — parser never calls planner.
        no_draft_created:      true — parser never calls drafter.
        no_execution_performed: true — parser never executes anything.
    """
    _s = resolve_llm_settings(provider, base_url, model)
    result = _parse_research_intent(
        user_request=request,
        provider=_s.provider,
        model=_s.model,
        base_url=_s.base_url,
        mode=mode,
    )
    d = result.to_dict()
    ok = result.status == "parsed"
    display_parts = [f"Intent [{result.status}]: {_short(result.summary, 120)}"]
    if result.unrecognised_terms:
        display_parts.append(
            f"Unrecognised terms: {result.unrecognised_terms}."
        )
    if result.registry_warnings:
        display_parts.append(
            f"Registry warnings: {len(result.registry_warnings)}."
        )
    if result.missing_after_parse:
        display_parts.append(
            f"Missing fields: {result.missing_after_parse}."
        )
    display = " ".join(display_parts)

    next_action = (
        "resolve_research_spec"
        if ok
        else "list_research_agent_options"
    )
    return _envelope(ok, f"intent_{result.status}", display, d, next_action)


def _mcp_resolve_research_request(
    request: str,
    provider: str | None = None,
    mode: str = "recommended",
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Parse a natural-language research request and resolve it to a ResearchSpec.

    Combines Phase-4A intent parsing with the deterministic resolver.
    Returns resolution status, resolved spec, missing fields, assumptions, and
    parse metadata (extracted_fields, unrecognised_terms, confidence).

    Governance:
        save_draft is never inferred from natural language.
        no_plan_created: true — stops at resolution.
        no_draft_created: true — stops at resolution.
        no_execution_performed: true — always.
        Execution/approval keywords in request → warning added, not acted on.
    """
    _s = resolve_llm_settings(provider, base_url, model)
    result = _nlw.run_resolve_from_request(
        request=request,
        provider=_s.provider,
        mode=mode,
        model=_s.model,
        base_url=_s.base_url,
    )
    d = result.to_dict()
    ok = result.status == "resolved"
    parse_status = (result.parse or {}).get("status", "")
    display_parts = [f"NL Resolve [{result.status}]: {_short(result.summary, 120)}"]
    if parse_status and parse_status != "parsed":
        display_parts.append(f"Parse: {parse_status}.")
    unrecog = (result.parse or {}).get("unrecognised_terms", [])
    if unrecog:
        display_parts.append(f"Unrecognised: {unrecog}.")
    next_action = (
        "create_experiment_plan_from_request" if ok
        else "parse_research_intent"
    )
    return _envelope(ok, f"nl_resolve_{result.status}", " ".join(display_parts), d, next_action)


def _mcp_create_experiment_plan_from_request(
    request: str,
    provider: str | None = None,
    mode: str = "recommended",
    baseline: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Parse a natural-language request and create a deterministic ExperimentPlan.

    Combines Phase-4A intent parsing with the deterministic resolver and planner.
    Returns plan status, hypothesis, planned changes, success criteria, and
    parse metadata.

    Governance:
        no_draft_created: true — stops at planning.
        no_execution_performed: true — always.
        Execution/approval keywords in request → warning added, not acted on.
    """
    _s = resolve_llm_settings(provider, base_url, model)
    result = _nlw.run_plan_from_request(
        request=request,
        provider=_s.provider,
        mode=mode,
        baseline=baseline,
        model=_s.model,
        base_url=_s.base_url,
    )
    d = result.to_dict()
    ok = result.status in ("planned", "needs_review")
    display_parts = [f"NL Plan [{result.status}]: {_short(result.summary, 120)}"]
    if result.warnings:
        display_parts.append(f"Warnings: {len(result.warnings)}.")
    next_action = (
        "create_validated_draft_from_request" if ok
        else "parse_research_intent"
    )
    return _envelope(ok, f"nl_plan_{result.status}", " ".join(display_parts), d, next_action)


def _mcp_create_validated_draft_from_request(
    request: str,
    provider: str | None = None,
    mode: str = "recommended",
    baseline: str | None = None,
    base_experiment: str | None = None,
    save_draft: bool = False,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Parse a natural-language request and create a validated ExperimentDraft preview.

    Full chain: NL request → parse → resolve → plan → validated draft.
    Returns resolution, plan, and draft results with mapped changes, unsupported
    fields, warnings, governance flags, and parse metadata.

    save_draft=True explicitly persists the draft via the Phase-3D bridge.
    save_draft is NEVER inferred from natural language — it must be passed explicitly.

    Governance:
        approval_required: true — always.
        execution_performed: false — always.
        draft_approved: false — always.
        Hard-unsupported fields (universe/tickers/frequency/data_source/asset_class)
        block draft creation.
        Execution/approval keywords in request → warning added, not acted on.
    """
    _s = resolve_llm_settings(provider, base_url, model)
    result = _nlw.run_draft_from_request(
        request=request,
        provider=_s.provider,
        mode=mode,
        baseline=baseline,
        base_experiment=base_experiment,
        save_draft=save_draft,
        model=_s.model,
        base_url=_s.base_url,
    )
    d = result.to_dict()
    ok = result.status == "draft_created"
    display_parts = [f"NL Draft [{result.status}]: {_short(result.summary, 120)}"]
    if result.unsupported_fields:
        display_parts.append(f"Unsupported (not applied): {result.unsupported_fields}.")
    saved = result.saved_draft or {}
    if saved.get("status") == "saved":
        display_parts.append(
            f"Draft saved: id={saved.get('draft_id')}."
        )
    elif save_draft and saved:
        display_parts.append(f"Draft not saved: {saved.get('summary', '')}.")
    next_action = (
        "approve_experiment_draft" if ok
        else "parse_research_intent"
    )
    return _envelope(
        ok, f"nl_draft_{result.status}", " ".join(display_parts), d, next_action
    )


# ---------------------------------------------------------------------------
# Phase 5C — Memory-aware next-action recommendation tools
# Deterministic: no LLM, no draft creation, no approval, no execution.
# ---------------------------------------------------------------------------

_NA_GOVERNANCE_NOTE = (
    " Governance guarantees (always in data): "
    "approval_required=true, execution_performed=false, "
    "draft_created=false, draft_approved=false. "
    "Recommendations are advisory only — no spec or draft is created unless "
    "the user explicitly requests it in a later step."
)


def _mcp_recommend_next_actions(
    spec: str | None = None,
    objective: str | None = None,
    mode: str = "recommended",
    max_candidates: int = 3,
    max_memory_results: int = 8,
) -> dict[str, Any]:
    """Resolve a ResearchSpec (optional) and recommend deterministic next actions.

    Retrieves prior research memory evidence, applies the Phase 5B rule-based
    planner, and returns ranked next-action candidates with supporting evidence,
    rejected directions, cautions, and governance flags.

    spec: JSON object string describing the ResearchSpec (optional).  When
          omitted, recommendations are based on objective alone or on empty
          memory (status=needs_more_evidence).
    objective: plain-text research objective (optional; enriches memory query).

    Governance:
        approval_required:   true — always.
        execution_performed: false — always.
        draft_created:       false — always.
        No draft is created. No experiment is executed.
    """
    import json

    native_spec = None
    if spec is not None:
        spec_str = spec if isinstance(spec, str) else json.dumps(spec)
        try:
            spec_dict: dict[str, Any] = json.loads(spec_str)
        except (json.JSONDecodeError, ValueError) as exc:
            return _envelope(
                False,
                "next_action_invalid_input",
                f"spec must be a valid JSON object string: {exc}",
                {"parse_error": str(exc)},
                "list_research_agent_options",
            )
        try:
            from src.agent.intent.research_spec import ResearchSpec as _RS
            native_spec = _RS.from_dict(spec_dict)
        except Exception as exc:
            return _envelope(
                False,
                "next_action_invalid_input",
                f"Failed to parse ResearchSpec: {exc}",
                {"error": str(exc)},
                "list_research_agent_options",
            )

    result = _naw.run_recommend_next_actions(
        objective=objective,
        spec=native_spec,
        mode=mode,
        max_candidates=max_candidates,
        max_memory_results=max_memory_results,
    )
    d = result.to_dict()

    ok = result.status in ("recommended", "needs_more_evidence")
    na = result.next_actions or {}
    top = na.get("top_candidate") or {}
    n_cands = len(na.get("candidates") or [])

    if result.status == "needs_user_input":
        display = (
            f"Next-action [needs_user_input]: {_short(result.summary, 120)} "
            f"Missing: {result.missing_fields}."
        )
        next_action = "list_research_agent_options"
    elif result.status in ("invalid", "error"):
        display = f"Next-action [{result.status}]: {_short(result.summary, 120)}"
        next_action = "list_research_agent_options"
    elif result.status == "needs_more_evidence":
        display = (
            f"Next-action [needs_more_evidence]: No prior evidence found. "
            f"{n_cands} baseline action(s) recommended."
        )
        next_action = "retrieve_research_memory"
    else:
        top_title = top.get("title", "(none)")
        top_priority = top.get("priority", "?")
        display = (
            f"Next-action [recommended]: {n_cands} candidate(s). "
            f"Top: {_short(top_title, 80)} (priority={top_priority})."
        )
        if result.warnings:
            display += f" Warnings: {len(result.warnings)}."
        next_action = "resolve_research_spec"

    return _envelope(ok, f"next_action_{result.status}", display, d, next_action)


def _mcp_recommend_next_actions_from_request(
    request: str,
    provider: str | None = None,
    mode: str = "recommended",
    model: str | None = None,
    base_url: str | None = None,
    max_candidates: int = 3,
    max_memory_results: int = 8,
) -> dict[str, Any]:
    """Parse a natural-language request and recommend deterministic next actions.

    Combines Phase-4A intent parsing with the Phase 5B memory-aware planner.
    Returns ranked next-action candidates with evidence, cautions, rejected
    directions, parse metadata, and governance flags.

    Governance:
        approval_required:   true — always.
        execution_performed: false — always.
        draft_created:       false — always.
        Execution/approval keywords (run, execute, approve, …) → warning, not acted on.
        No draft is created. No experiment is executed.
    """
    _s = resolve_llm_settings(provider, base_url, model)
    result = _naw.run_recommend_next_actions_from_request(
        request=request,
        provider=_s.provider,
        mode=mode,
        model=_s.model,
        base_url=_s.base_url,
        max_candidates=max_candidates,
        max_memory_results=max_memory_results,
    )
    d = result.to_dict()

    ok = result.status in ("recommended", "needs_more_evidence")
    na = result.next_actions or {}
    n_cands = len(na.get("candidates") or [])
    top = na.get("top_candidate") or {}
    parse_status = (result.parse or {}).get("status", "")

    display_parts: list[str] = []
    if parse_status and parse_status != "parsed":
        display_parts.append(f"Parse: {parse_status}.")
    if result.status == "needs_clarification":
        display_parts.insert(0, f"NL Next-action [needs_clarification]: {_short(result.summary, 100)}")
        next_action = "parse_research_intent"
    elif result.status == "needs_user_input":
        display_parts.insert(0, f"NL Next-action [needs_user_input]: {_short(result.summary, 100)}")
        display_parts.append(f"Missing: {result.missing_fields}.")
        next_action = "list_research_agent_options"
    elif result.status in ("invalid", "error"):
        display_parts.insert(0, f"NL Next-action [{result.status}]: {_short(result.summary, 100)}")
        next_action = "parse_research_intent"
    elif result.status == "needs_more_evidence":
        display_parts.insert(
            0,
            f"NL Next-action [needs_more_evidence]: {n_cands} baseline action(s).",
        )
        next_action = "retrieve_research_memory"
    else:
        top_title = top.get("title", "(none)")
        top_priority = top.get("priority", "?")
        display_parts.insert(
            0,
            f"NL Next-action [recommended]: {n_cands} candidate(s). "
            f"Top: {_short(top_title, 80)} (priority={top_priority}).",
        )
        next_action = "resolve_research_spec"

    if result.warnings:
        display_parts.append(f"Warnings: {len(result.warnings)}.")

    return _envelope(
        ok,
        f"nl_next_action_{result.status}",
        " ".join(display_parts),
        d,
        next_action,
    )


# ---------------------------------------------------------------------------
# Phase 5E — Candidate-to-spec proposal (deterministic, no LLM)
# ---------------------------------------------------------------------------

_SP_GOVERNANCE_NOTE = (
    " Governance guarantees (always in data.governance): "
    "approval_required=true, requires_user_confirmation=true, "
    "draft_created=false, execution_performed=false, draft_approved=false. "
    "This is a proposal only — no draft is created, no spec is saved, "
    "no experiment is executed, no approval is granted automatically."
)


def _mcp_propose_spec_from_next_action(
    candidate: str,
    base_spec: str | None = None,
    mode: str = "recommended",
) -> dict[str, Any]:
    """Convert a NextActionCandidate dict into a ResearchSpecProposal.

    Resolves the proposed ResearchSpec through the deterministic resolver and
    returns proposed changes, unchanged fields, supporting evidence IDs,
    unsupported changes, resolution status, warnings, and governance flags.

    candidate: JSON object string with required fields action_id, action_type,
               title, description, suggested_spec_changes (plus optional
               rationale, supporting_evidence_ids, main_risks, priority,
               confidence, etc.).
    base_spec: JSON object string describing the current ResearchSpec
               (optional but recommended for meaningful proposed-vs-current diffs).
    mode:      resolver mode, default "recommended".

    Governance:
        approval_required:          true — always.
        requires_user_confirmation: true — always.
        draft_created:              false — always.
        execution_performed:        false — always.
        draft_approved:             false — always.
        No draft is created. No spec is saved. No experiment is executed.
    """
    import json

    # Parse candidate
    cand_str = candidate if isinstance(candidate, str) else json.dumps(candidate)
    try:
        cand_dict: dict[str, Any] = json.loads(cand_str)
    except (json.JSONDecodeError, ValueError) as exc:
        return _envelope(
            False,
            "spec_proposal_invalid_input",
            f"candidate must be a valid JSON object string: {exc}",
            {"parse_error": str(exc)},
            "recommend_next_actions",
        )

    native_cand, parse_err = _spw._parse_candidate_from_dict(cand_dict)
    if parse_err:
        return _envelope(
            False,
            "spec_proposal_invalid_candidate",
            f"Invalid candidate: {parse_err}",
            {"error": parse_err},
            "recommend_next_actions",
        )

    # Parse base_spec (optional)
    native_base: Any = None
    if base_spec is not None:
        bs_str = base_spec if isinstance(base_spec, str) else json.dumps(base_spec)
        try:
            bs_dict: dict[str, Any] = json.loads(bs_str)
        except (json.JSONDecodeError, ValueError) as exc:
            return _envelope(
                False,
                "spec_proposal_invalid_input",
                f"base_spec must be a valid JSON object string: {exc}",
                {"parse_error": str(exc)},
                "list_research_agent_options",
            )
        try:
            from src.agent.intent.research_spec import ResearchSpec as _RS
            native_base = _RS.from_dict(bs_dict)
        except Exception as exc:
            return _envelope(
                False,
                "spec_proposal_invalid_input",
                f"Failed to parse base_spec ResearchSpec: {exc}",
                {"error": str(exc)},
                "list_research_agent_options",
            )

    result = _spw.run_propose_spec_from_candidate(
        candidate=native_cand,
        base_spec=native_base,
        mode=mode,
    )
    d = result.to_dict()

    ok = result.status == "proposal_created"
    sp = result.spec_proposal or {}
    prop_status = sp.get("status", result.status)
    proposed_changes = sp.get("proposed_changes") or {}
    unsupported = sp.get("unsupported_changes") or []

    if result.status == "proposal_created":
        display = (
            f"Spec proposal [proposal_created]. "
            f"Action: {native_cand.action_type}. "
            f"Changes: {list(proposed_changes.keys()) if proposed_changes else 'none'}."
        )
        if unsupported:
            display += f" Unsupported: {len(unsupported)}."
        next_action = "resolve_research_spec"
    elif result.status == "needs_user_input":
        display = f"Spec proposal [needs_user_input]: {_short(result.summary, 120)}"
        next_action = "recommend_next_actions"
    elif result.status == "unsupported_action":
        display = f"Spec proposal [unsupported_action]: {_short(result.summary, 120)}"
        next_action = "recommend_next_actions"
    else:
        display = f"Spec proposal [{result.status}]: {_short(result.summary, 120)}"
        next_action = "list_research_agent_options"

    if result.warnings:
        display += f" Warnings: {len(result.warnings)}."

    return _envelope(ok, f"spec_proposal_{prop_status}", display, d, next_action)


def _mcp_propose_spec_from_recommendation_request(
    request: str,
    candidate_index: int = 0,
    provider: str | None = None,
    mode: str = "recommended",
    model: str | None = None,
    base_url: str | None = None,
    max_candidates: int = 3,
    max_memory_results: int = 8,
) -> dict[str, Any]:
    """Parse a natural-language research request, recommend next actions, select
    a candidate by index, and produce a ResearchSpecProposal.

    Combines Phase-4A intent parsing, Phase 5B memory-aware planning, and
    Phase 5D candidate-to-spec proposal into a single call.  Returns the full
    recommendation context alongside the selected candidate and its spec proposal.

    request:         natural-language research request.
    candidate_index: 0-based index into the ranked candidates list (default 0 = top).
    provider:        LLM provider — reads ZETO_AGENT_LLM_PROVIDER env var if omitted.
    mode:            resolver mode, default "recommended".
    max_candidates:  maximum candidates to generate (default 3).
    max_memory_results: maximum evidence items to retrieve (default 8).

    Governance:
        approval_required:          true — always.
        requires_user_confirmation: true — always.
        draft_created:              false — always.
        execution_performed:        false — always.
        draft_approved:             false — always.
        Execution/approval keywords in request → warning, never acted on.
        No draft is created. No spec is saved. No experiment is executed.
    """
    _s = resolve_llm_settings(provider, base_url, model)
    result = _spw.run_propose_spec_from_request(
        request=request,
        candidate_index=candidate_index,
        provider=_s.provider,
        mode=mode,
        model=_s.model,
        base_url=_s.base_url,
        max_candidates=max_candidates,
        max_memory_results=max_memory_results,
    )
    d = result.to_dict()

    ok = result.status == "proposal_created"
    sp = result.spec_proposal or {}
    prop_status = sp.get("status", result.status)
    sel = result.selected_candidate or {}
    na = result.next_actions or {}
    n_cands = len(na.get("candidates") or [])

    if result.status == "proposal_created":
        action_type = sel.get("action_type", "?")
        proposed_keys = list((sp.get("proposed_changes") or {}).keys())
        display = (
            f"NL Spec proposal [proposal_created]: "
            f"{n_cands} candidate(s), selected index {candidate_index}. "
            f"Action: {action_type}. Changes: {proposed_keys if proposed_keys else 'none'}."
        )
        next_action = "resolve_research_spec"
    elif result.status == "invalid_candidate_index":
        display = (
            f"NL Spec proposal [invalid_candidate_index]: "
            f"{_short(result.summary, 120)}"
        )
        next_action = "recommend_next_actions"
    elif result.status == "needs_more_evidence":
        display = f"NL Spec proposal [needs_more_evidence]: {_short(result.summary, 100)}"
        next_action = "retrieve_research_memory"
    elif result.status in ("needs_clarification", "needs_user_input"):
        display = f"NL Spec proposal [{result.status}]: {_short(result.summary, 100)}"
        next_action = "parse_research_intent"
    elif result.status == "unsupported_action":
        display = f"NL Spec proposal [unsupported_action]: {_short(result.summary, 100)}"
        next_action = "recommend_next_actions"
    else:
        display = f"NL Spec proposal [{result.status}]: {_short(result.summary, 100)}"
        next_action = "list_research_agent_options"

    if result.warnings:
        display += f" Warnings: {len(result.warnings)}."

    return _envelope(
        ok,
        f"nl_spec_proposal_{prop_status}",
        display,
        d,
        next_action,
    )


# ---------------------------------------------------------------------------
# Phase 5F — Proposal-to-draft preview (deterministic, no LLM)
# ---------------------------------------------------------------------------

_PD_GOVERNANCE_NOTE = (
    " Governance guarantees (always in data.governance): "
    "approval_required=true, requires_user_confirmation=true, "
    "draft_approved=false, execution_performed=false. "
    "A draft may only be created when confirm_proposal=true. "
    "Saving requires save_draft=true. "
    "No draft is approved. No experiment is executed."
)


def _mcp_create_validated_draft_from_spec_proposal(
    proposal: str,
    confirm_proposal: bool = False,
    base_experiment: str | None = None,
    save_draft: bool = False,
    mode: str = "recommended",
) -> dict[str, Any]:
    """Convert an explicitly confirmed ResearchSpecProposal into a validated draft preview.

    A draft is only created when confirm_proposal=true is passed explicitly.
    Confirmation cannot be inferred from natural language.

    proposal:         JSON object string of a ResearchSpecProposal (as returned
                      by propose_spec_from_next_action or
                      propose_spec_from_recommendation_request).
    confirm_proposal: Must be true to proceed.  When false, returns
                      status=confirmation_required with no side effects.
    base_experiment:  Optional config name to use as the baseline in the plan.
    save_draft:       If true, persist the validated draft as an unapproved
                      draft (Phase 3D bridge).  Requires confirm_proposal=true
                      and a successful draft_created result.
    mode:             Resolver mode, default "recommended".

    Hard-unsupported fields (universe, tickers, frequency, data_source,
    asset_class) block draft creation.

    Soft-unsupported fields (start_date, end_date) are reported in
    result.unsupported_fields but do not block draft_created.

    Governance:
        approval_required:          true — always.
        requires_user_confirmation: true — always.
        draft_created:              true only if a validated preview was created.
        draft_saved:                true only if save_draft=true and saved.
        draft_approved:             false — always.
        execution_performed:        false — always.
    """
    import json

    # Parse proposal JSON
    prop_str = proposal if isinstance(proposal, str) else json.dumps(proposal)
    try:
        proposal_dict: dict[str, Any] = json.loads(prop_str)
    except (json.JSONDecodeError, ValueError) as exc:
        return _envelope(
            False,
            "proposal_draft_invalid_input",
            f"proposal must be a valid JSON object string: {exc}",
            {"parse_error": str(exc)},
            "propose_spec_from_next_action",
        )

    # Reconstruct ResearchSpecProposal
    try:
        from src.agent.next_actions.spec_proposal import ResearchSpecProposal as _RSP
        native_proposal = _RSP.from_dict(proposal_dict)
    except Exception as exc:
        return _envelope(
            False,
            "proposal_draft_invalid_proposal",
            f"Failed to parse ResearchSpecProposal: {exc}",
            {"error": str(exc)},
            "propose_spec_from_next_action",
        )

    result = _ptdw.run_draft_from_spec_proposal(
        proposal=native_proposal,
        confirm_proposal=confirm_proposal,
        base_experiment=base_experiment,
        save_draft=save_draft,
        mode=mode,
    )
    d = result.to_dict()

    ok = result.status == "draft_created"
    gov = d.get("governance") or {}

    if result.status == "confirmation_required":
        display = (
            "Proposal-to-draft [confirmation_required]: "
            "confirm_proposal=True is required to create a draft. "
            "No draft created. No spec saved."
        )
        next_action = "propose_spec_from_next_action"
    elif result.status == "invalid_proposal":
        display = f"Proposal-to-draft [invalid_proposal]: {_short(result.summary, 120)}"
        next_action = "propose_spec_from_next_action"
    elif result.status == "unsupported_proposal":
        n_unsup = len(result.errors)
        display = (
            f"Proposal-to-draft [unsupported_proposal]: {n_unsup} unsupported change(s). "
            f"{_short(result.summary, 100)}"
        )
        next_action = "propose_spec_from_next_action"
    elif result.status == "draft_created":
        saved_note = ""
        sd = result.saved_draft or {}
        if save_draft:
            if sd.get("status") == "saved":
                saved_note = f" Saved: id={sd.get('draft_id')} path={_short(str(sd.get('draft_path', '')), 60)}."
            else:
                saved_note = f" Not saved: {_short(sd.get('summary', ''), 60)}."
        display = (
            f"Proposal-to-draft [draft_created]: "
            f"Draft '{result.proposed_name}' from proposal."
            + saved_note
        )
        if result.unsupported_fields:
            display += f" Soft-unsupported (not applied): {result.unsupported_fields}."
        if result.hard_unsupported_fields:
            display += f" BLOCKED by hard-unsupported: {result.hard_unsupported_fields}."
        next_action = "approve_experiment_draft"
    elif result.status == "unsupported_plan":
        display = (
            f"Proposal-to-draft [unsupported_plan]: "
            f"Hard-unsupported fields blocked draft: {result.hard_unsupported_fields}."
        )
        next_action = "propose_spec_from_next_action"
    elif result.status in ("needs_user_input", "invalid"):
        display = f"Proposal-to-draft [{result.status}]: {_short(result.summary, 120)}"
        next_action = "resolve_research_spec"
    else:
        display = f"Proposal-to-draft [{result.status}]: {_short(result.summary, 120)}"
        next_action = "list_research_agent_options"

    if result.warnings:
        display += f" Warnings: {len(result.warnings)}."

    return _envelope(
        ok,
        f"proposal_draft_{result.status}",
        display,
        d,
        next_action,
    )


_TOOLS: list[tuple[Any, str, str]] = [
    (get_zeto_operator_manual, "get_zeto_operator_manual",
     "Read-only: load the fixed Zeto operator manual (compact rules) at the "
     "start of a session. Executes nothing, calls no LLM, reads no files, "
     "mutates no state. Returns only the fixed rules — no arbitrary paths."),
    (route_research_request, "route_research_request",
     "Read-only: classify a natural-language request and recommend which tool "
     "to call first. Returns request_type (new_research_design / memory_next_action / "
     "existing_experiment_review / existing_draft_governance / execution_request / unclear), "
     "recommended_tool, recommended_next_tools, discouraged_tools, requires_confirmation, "
     "reason, and governance flags. "
     "Deterministic keyword classification — no LLM, no session, no state mutation. "
     "Call this for any ambiguous NL request before choosing an orchestration tool. "
     "Follow recommended_tool; avoid discouraged_tools for this request type."),
    (get_research_memory_status, "get_research_memory_status",
     "Read-only: report whether the research-memory index exists and how many "
     "items / experiments it holds. Evidence layer only — runs nothing."),
    (index_research_memory, "index_research_memory",
     "Controlled write: build/refresh the research-memory index from known Zeto "
     "artefact locations only (metadata, metrics, reviews, proposals, drafts, "
     "reports, sessions). Runs no experiment, calls no LLM, approves/renders/"
     "executes nothing, mutates no source artefacts, reads no arbitrary paths."),
    (retrieve_research_memory, "retrieve_research_memory",
     "Read-only: retrieve prior research evidence by deterministic keyword / "
     "metadata match (query, experiment_name, failure_modes, artefact_type, "
     "top_k). Returns compact items (summaries, paths, hashes, tags, matched "
     "terms) — never full artefacts. Evidence only: authorises no execution, "
     "approval, or rendering; quant metrics remain authoritative."),
    (get_semantic_research_memory_status, "get_semantic_research_memory_status",
     "Read-only: report whether the semantic (embedding) memory index exists, "
     "how many items it holds, and the embedding model. Evidence layer only."),
    (index_semantic_research_memory, "index_semantic_research_memory",
     "Controlled write: embed Phase 1 memory records into a local semantic index "
     "via a local embeddings endpoint only (provider/model/base_url; default "
     "LM Studio nomic). Calls no chat/completion LLM, runs no experiment, "
     "approves/renders/executes nothing, mutates no source artefacts, reads no "
     "arbitrary paths. Requires a Phase 1 index first."),
    (semantic_retrieve_research_memory, "semantic_retrieve_research_memory",
     "Read-only: retrieve prior research evidence by local semantic similarity "
     "(query embedded with the same model; optional experiment_name / "
     "failure_modes / artefact_type / tags filters; top_k). Returns compact items "
     "with cosine scores (summaries, paths, hashes, tags) — never full artefacts. "
     "Evidence only: suggestions, not proof; authorises no execution, approval, "
     "or rendering. On failure it stops and never invents evidence."),
    (inspect_experiment_config, "inspect_experiment_config",
     "Read-only: compact summary of an experiment's current YAML config — model "
     "type/params, feature count/names, validation settings, universe, and the "
     "list of changeable field paths. Does NOT return raw YAML. Executes nothing, "
     "calls no LLM, reads no arbitrary paths."),
    (list_changeable_config_fields, "list_changeable_config_fields",
     "Read-only: authoritative list of config field paths that generate_config_change_draft "
     "accepts, with their operations (set / add / remove / replace), types, allowed "
     "values, and current values when experiment_name is given. Call this before "
     "building a draft to know what can and cannot be changed."),
    (list_available_features, "list_available_features",
     "Read-only: schema-valid feature types only — never invents names. Includes "
     "family, required_params, and whether each type is currently used when "
     "experiment_name is given. Optional family/query filters. Use this to find valid "
     "feature types before calling generate_config_change_draft with add/remove/replace."),
    (list_supported_models, "list_supported_models",
     "Read-only: model types supported by the quant engine schema — never invents "
     "models. Model switching IS supported via generate_config_change_draft with "
     "field_path='model.type', operation='set'. Highlights the current model when "
     "experiment_name is given."),
    (generate_config_change_draft, "generate_config_change_draft",
     "Use this for multi-change or feature add/remove/replace drafts. Deterministic, "
     "no LLM: accepts a JSON-array 'changes' with set/add/remove/replace operations, "
     "validates all feature types against the schema, refuses unknown features/models/"
     "paths, assigns a unique name, and persists an UNAPPROVED draft. Never approves, "
     "renders, or executes. generate_parameter_change_draft remains available for "
     "simple single-field set operations." + _SESSION_HINT),
    (generate_parameter_change_draft, "generate_parameter_change_draft",
     "Use this (NOT generate_experiment_draft) when the user asks for a specific "
     "config change such as 'set model.params.alpha to 2'. Deterministic, no LLM: "
     "applies exactly field_path -> proposed_value, reads the real current value "
     "from the base config, assigns a unique name, validates, and persists an "
     "UNAPPROVED draft. Refuses invalid field paths and schema-incompatible values "
     "(no fallback field invented). Never approves, renders, or executes." + _SESSION_HINT),
    (get_experiment_metrics, "get_experiment_metrics",
     "Read-only: authoritative metrics (Sharpe, OOS Sharpe, drawdown, hit rate, "
     "failure modes) for one experiment, read from real artefacts. Use this — NOT "
     "retrieve_research_memory / semantic_retrieve_research_memory — to answer "
     "performance questions. Missing metrics are reported, never invented."),
    (compare_experiment_metrics, "compare_experiment_metrics",
     "Read-only: compare two experiments (Sharpe / mean OOS Sharpe / max-drawdown "
     "deltas + failure modes) using real artefact metrics, not RAG memory. Use this "
     "to answer whether performance improved. Missing metrics are reported clearly."),
    (inspect_comparison_evidence, "inspect_comparison_evidence",
     "Read-only: inspect a specific comparison evidence record for a base/candidate "
     "pair. Returns tested_change, metric deltas, failure modes, and conclusion from "
     "the persisted record — no LLM, no RAG. Direct authoritative summary. If no "
     "record exists, run compare_experiment_metrics first to create it."),
    (rank_experiments, "rank_experiments",
     "Read-only: rank experiment variants by sharpe_ratio, mean_oos_sharpe, "
     "max_drawdown_pct, or robustness_score (composite). Uses authoritative artefact "
     "metrics only. Missing metrics noted per-experiment, never invented. "
     "Use to answer 'which experiment is best?' questions."),
    (summarise_experiment_family, "summarise_experiment_family",
     "Read-only: summarise what has been learned across all experiments sharing a "
     "name prefix. Aggregates authoritative metrics, common failure modes, comparison "
     "evidence highlights, and deterministic lessons. No LLM, no execution. "
     "Use to answer 'what have we learned?' questions."),
    (check_research_idea_history, "check_research_idea_history",
     "Read-only: check whether a similar research idea has already been tested. "
     "Scans comparison evidence records by keyword relevance. Returns matched changes, "
     "metric deltas, and a recommendation. Advisory only — warns but does not block. "
     "Use before drafting to avoid repeating failed ideas."),
    (propose_research_plan, "propose_research_plan",
     "Read-only: create a compact governed research plan from existing evidence. "
     "Deterministic from artefact metrics and comparison evidence — no LLM, no RAG. "
     "Does NOT generate drafts, approve, render, or execute. Each step is advisory. "
     "Use generate_config_change_draft to act on a plan step."),
    (list_experiments, "list_experiments",
     "List experiments that have persisted result artefacts. No side effects."),
    (create_research_session, "create_research_session",
     "Create a research session for an experiment with a stated research goal. "
     "Returns session_id (a UUID) — store it and pass it to all session-aware "
     "tools (review, proposal, draft, validate, approve, render, execute, "
     "post-run review, get_session_summary)."),
    (get_session_summary, "get_session_summary",
     "Return the current state of a research session. Requires session_id = the "
     "UUID returned by create_research_session — NOT an experiment name, "
     "context_hash, or draft_id. If the UUID was lost, call "
     "get_latest_research_session first."),
    (list_research_sessions, "list_research_sessions",
     "Read-only: list existing research session IDs (UUIDs). Use to recover a "
     "lost session_id. No side effects."),
    (get_latest_research_session, "get_latest_research_session",
     "Read-only: return the most recently updated research session (its UUID + "
     "a compact summary). Recovery helper for when the session_id was lost. The "
     "returned session_id is the UUID to pass to session-aware tools."),
    (check_research_workflow_state, "check_research_workflow_state",
     "Read-only preflight: report which artefacts exist for an experiment "
     "(baseline incl. metadata.json/metrics.json, review, proposal, latest "
     "draft + approval, rendered YAML, revised run, report/plots) and the "
     "recommended next step. Executes nothing. Call before major transitions "
     "to avoid out-of-order calls (e.g. validating before a draft exists)."),
    (get_workflow_trace, "get_workflow_trace",
     "Read-only: who owns this research workflow right now (Proposal & Draft "
     "Pipeline / Approval Gate / Render & Execution Engine / Review Engine / "
     "Idle), what phase it's in, what the last recorded session action was, "
     "and what's next. session_id is required; experiment_name defaults to "
     "the session's active experiment. Derived entirely from the existing "
     "session event log and check_research_workflow_state's artefact "
     "inspection — introduces no new state. Executes nothing."),
    (build_context_summary, "build_context_summary",
     "Assemble deterministic structured context (failure modes, performance, "
     "validation) for an experiment. No LLM call, no experiment run."),
    (run_experiment_review, "run_experiment_review",
     "Run an advisory LLM review of an experiment's diagnostics. Interprets "
     "existing artefacts only; runs no experiment." + _LLM_HINT + _SESSION_HINT),
    (generate_iteration_proposal, "generate_iteration_proposal",
     "Generate an advisory next-step research proposal grounded in diagnostics. "
     "Prescribes no parameter values and runs no experiment." + _LLM_HINT + _SESSION_HINT),
    (generate_experiment_draft, "generate_experiment_draft",
     "Synthesise a schema-bounded config draft (typed deltas, not YAML). The "
     "draft is unapproved and must be validated and explicitly approved." + _LLM_HINT
     + " With provider='stub' it returns a deterministic draft "
       "(model.params.alpha -> 1.0)." + _SESSION_HINT),
    (validate_experiment_draft, "validate_experiment_draft",
     "Validate a draft against the authoritative config schema. No side effects."
     + _SESSION_HINT),
    (approve_experiment_draft, "approve_experiment_draft",
     "Explicitly approve a draft so it can be rendered. Requires "
     "approval_confirmation='APPROVE'. Only this tool approves; no other tool "
     "auto-approves. Call it ONLY when the latest user message explicitly approves "
     "the draft (e.g. 'I approve the draft.'). Never infer approval from 'start a "
     "research cycle', from validation success, or from a next_suggested_action; "
     "never approve automatically after validation. Pass render_requested=true "
     "ONLY if the user explicitly asked to approve AND render in the same message; "
     "otherwise the suggested next step is to stop and ask the user before "
     "rendering." + _SESSION_HINT),
    (render_draft_to_yaml, "render_draft_to_yaml",
     "Render an already-approved experiment draft to YAML. This does not execute "
     "the experiment. After rendering, STOP and ask the user for RUN — rendering "
     "never authorises execution." + _SESSION_HINT),
    (verify_experiment_artifacts, "verify_experiment_artifacts",
     "Read-only: verify artefact completeness for one experiment after execution. "
     "Checks artefact_root, metrics.json (presence + parseability), required "
     "metric values, report, and plots. Returns artefact_status='complete' or "
     "'incomplete'. If incomplete: ok=false and next_suggested_action=stop_and_report_to_user. "
     "Call this AFTER execute_approved_config and BEFORE review_post_run_result. "
     "Never invents values; no LLM, no execution."),
    (execute_approved_config, "execute_approved_config",
     "Run exactly one approved YAML config through the quant research engine. "
     "Requires confirmation='RUN'. Call this ONLY when the latest user instruction "
     "explicitly contains the literal token RUN. Never infer RUN from a previous "
     "approval, a previous render, the word 'execute'/'yes'/'proceed'/'continue', "
     "or a failed/refused execution attempt; after a refusal, stop and wait for a "
     "new user message. Does not optimise, loop, retry, trade, or register lineage "
     "automatically." + _SESSION_HINT),
    (review_post_run_result, "review_post_run_result",
     "Run an advisory LLM review of artefacts produced by a freshly executed "
     "config. Interprets existing artefacts only." + _LLM_HINT + _SESSION_HINT),
    # -----------------------------------------------------------------------
    # Phase 3C — Agent workflow tools (deterministic, no LLM, no execution)
    # -----------------------------------------------------------------------
    (_mcp_list_research_agent_options, "list_research_agent_options",
     "List valid registry options for the research agent workflow. "
     "Pass field='models' (or features/signals/labels/universes/asset_classes/"
     "validation/portfolio_construction/diagnostics/presets/defaults/change_paths) "
     "to see options for that category. Omit field for an overview of all categories. "
     "No resolver, no planner, no draft, no execution."),
    (_mcp_resolve_research_spec, "resolve_research_spec",
     "Resolve a structured ResearchSpec (JSON string) using registry defaults and "
     "compatibility checks. Returns resolution status (resolved/needs_user_input/"
     "invalid), resolved spec, missing fields, recommended defaults, assumptions, "
     "defaulted fields, and warnings. No plan, no draft, no execution."),
    (_mcp_create_experiment_plan_from_spec, "create_experiment_plan_from_spec",
     "Resolve a ResearchSpec (JSON string) and create a deterministic ExperimentPlan. "
     "Returns resolution summary, plan status (planned/needs_review/invalid), "
     "hypothesis, baseline, planned changes, success criteria, diagnostics, risks, "
     "and critic warnings. No draft, no execution."),
    (_mcp_create_validated_draft_from_spec, "create_validated_draft_from_spec",
     "Run the full deterministic chain: ResearchSpec (JSON string) → resolve → plan "
     "→ validated draft preview. Returns resolution, plan, and draft results with "
     "mapped changes, unsupported fields, warnings, and governance flags. "
     "Hard-unsupported fields (universe/tickers/frequency/data_source/asset_class) "
     "block draft creation. Soft-unsupported fields (start_date/end_date) are "
     "reported but do not block the draft. The draft is unapproved and not executed. "
     "approval_required=true, execution_performed=false, draft_approved=false always."),
    (_mcp_parse_research_intent, "parse_research_intent",
     "Phase 4A: parse a natural-language research request into a structured ResearchSpec "
     "using registry-constrained intent extraction. "
     "The model output is constrained to registry-supported values only. "
     "Unsupported values are rejected and returned in unrecognised_terms/registry_warnings. "
     "Use provider='stub' (default) for keyword-based offline extraction, "
     "or provider='openai'/'anthropic' with a real model for full NL parsing." + _LLM_HINT + " "
     "No plan is created. No draft is created. No execution is performed."),
    # -----------------------------------------------------------------------
    # Phase 4B — Natural-language workflow (parse → resolve → plan → draft)
    # -----------------------------------------------------------------------
    (_mcp_resolve_research_request, "resolve_research_request",
     "Phase 4B: parse a natural-language research request and resolve it to a ResearchSpec. "
     "Combines Phase-4A intent parsing with the deterministic resolver. "
     "Returns resolved spec, missing fields, assumptions, and parse metadata. "
     "Stops at resolution — no plan, no draft, no execution." + _LLM_HINT + " "
     "Execution/approval keywords in request produce a warning, never an action. "
     "save_draft is never inferred from natural language."),
    (_mcp_create_experiment_plan_from_request, "create_experiment_plan_from_request",
     "Phase 4B: parse a natural-language request and create a deterministic ExperimentPlan. "
     "Combines Phase-4A intent parsing with the deterministic resolver and planner. "
     "Returns plan status, hypothesis, success criteria, risks, and parse metadata. "
     "Stops at planning — no draft, no execution." + _LLM_HINT + " "
     "Execution/approval keywords in request produce a warning, never an action."),
    (_mcp_create_validated_draft_from_request, "create_validated_draft_from_request",
     "Phase 4B: parse a natural-language request and run the full NL chain: "
     "parse → resolve → plan → validated draft preview. "
     "Returns resolution, plan, draft results with governance flags and parse metadata. "
     "Hard-unsupported fields (universe/tickers/frequency/data_source/asset_class) "
     "block draft creation. Soft-unsupported fields reported but non-blocking. "
     "approval_required=true, execution_performed=false, draft_approved=false always. "
     "save_draft=True explicitly persists — NEVER inferred from natural language." + _LLM_HINT),
    # -----------------------------------------------------------------------
    # Phase 5C — Memory-aware next-action recommendations (deterministic, no LLM)
    # -----------------------------------------------------------------------
    (_mcp_recommend_next_actions, "recommend_next_actions",
     "Phase 5C: resolve a ResearchSpec (optional JSON string) and recommend "
     "deterministic next research actions backed by prior memory evidence. "
     "Returns ranked candidates (action_type, rationale, supporting evidence IDs, "
     "risks, failure modes to monitor), rejected directions, cautions, memory "
     "context summary, and governance flags. "
     "Pass max_candidates (default 3) to control how many are returned; if more "
     "were generated a truncation warning appears. "
     "Pass max_memory_results (default 8) to limit evidence retrieval. "
     "No draft created. No approval. No execution." + _NA_GOVERNANCE_NOTE),
    (_mcp_recommend_next_actions_from_request, "recommend_next_actions_from_request",
     "Phase 5C: parse a natural-language request and recommend deterministic next "
     "research actions. Combines Phase-4A intent parsing with the Phase 5B "
     "memory-aware planner. Returns candidates, parse metadata, resolution status, "
     "evidence summary, rejected directions, cautions, and governance flags. "
     "Execution/approval keywords (run, execute, approve, …) produce a warning, "
     "never an action. "
     "Pass max_candidates (default 3) and max_memory_results (default 8). "
     "No draft created. No approval. No execution." + _LLM_HINT + _NA_GOVERNANCE_NOTE),
    # -----------------------------------------------------------------------
    # Phase 5E — Candidate-to-spec proposals (deterministic, no LLM)
    # -----------------------------------------------------------------------
    (_mcp_propose_spec_from_next_action, "propose_spec_from_next_action",
     "Phase 5E: convert a selected NextActionCandidate dict into a ResearchSpecProposal. "
     "Validates candidate fields, applies registry-backed validation (models, features, "
     "feature families, validation protocols, universes), runs the deterministic resolver "
     "on the proposed spec, and returns proposed changes, unchanged fields, supporting "
     "evidence IDs, unsupported changes, resolution status, warnings, and governance flags. "
     "candidate must be a JSON object string with action_id, action_type, title, description, "
     "and suggested_spec_changes. base_spec (optional JSON object string) provides the "
     "current ResearchSpec for diff context. "
     "Feature families are validated against registry-derived families: "
     "Trend, Volatility, Mean-Reversion, Market Structure. "
     "No draft created. No spec saved. No approval. No execution." + _SP_GOVERNANCE_NOTE),
    (_mcp_propose_spec_from_recommendation_request, "propose_spec_from_recommendation_request",
     "Phase 5E: parse a natural-language request, recommend next research actions, "
     "select one candidate by index, and produce a ResearchSpecProposal. "
     "Combines Phase-4A intent parsing + Phase 5B memory-aware planning + "
     "Phase 5D candidate-to-spec proposal in one call. "
     "Returns full recommendation context (all candidates, rejected directions, memory "
     "context summary) alongside the selected candidate and its spec proposal. "
     "candidate_index selects from the ranked list (default 0 = top candidate). "
     "Execution/approval keywords in request produce a warning, never an action. "
     "Pass max_candidates (default 3) and max_memory_results (default 8). "
     "No draft created. No spec saved. No approval. No execution." + _LLM_HINT + _SP_GOVERNANCE_NOTE),
    # -----------------------------------------------------------------------
    # Phase 5F — Proposal-to-draft preview (deterministic, confirmation-gated)
    # -----------------------------------------------------------------------
    (_mcp_create_validated_draft_from_spec_proposal, "create_validated_draft_from_spec_proposal",
     "Phase 5F: convert an explicitly confirmed ResearchSpecProposal into a validated "
     "draft preview, and optionally save it as an unapproved draft. "
     "Requires confirm_proposal=true — without this flag the tool returns "
     "status=confirmation_required and takes no action. "
     "proposal must be a JSON object string (as returned by propose_spec_from_next_action "
     "or propose_spec_from_recommendation_request). "
     "If save_draft=true, persists the draft using the Phase 3D unapproved-draft bridge. "
     "Hard-unsupported fields (universe, tickers, frequency, data_source, asset_class) "
     "block draft creation. "
     "Soft-unsupported fields (start_date, end_date) are reported but do not block. "
     "Proposals with unsupported_changes are rejected before any draft is attempted. "
     "No draft is approved. No experiment is executed." + _PD_GOVERNANCE_NOTE),
]

for _fn, _name, _desc in _TOOLS:
    mcp.add_tool(_fn, name=_name, description=_desc)


def main() -> None:
    """Run the Zeto MCP server over stdio (the transport LM Studio expects)."""
    mcp.run()


if __name__ == "__main__":
    main()
