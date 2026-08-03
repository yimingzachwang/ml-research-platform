"""Tests for Workflow Trace (research_api.py): WorkflowPhase, the shared
_workflow_progression_step()/next_tool_and_display() logic, and
compute_workflow_trace().

See docs/architecture/WORKFLOW_TRACE_ARCHITECTURE_DESIGN_PROPOSAL.md.
No quant engine, no LLM — pure derivation over on-disk artefact presence and
session state, isolated to tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from src.orchestration.api.research_api import (
    WorkflowPhase,
    WorkflowTrace,
    _WORKFLOW_OWNER_BY_PHASE,
    _WORKFLOW_STATUS_BY_PHASE,
    _workflow_progression_step,
    compute_workflow_trace,
    create_research_session,
    next_tool_and_display,
    record_session_event,
)
from src.orchestration.session.session_schema import ResearchSession, SessionEventType

_EXP = "exp_a"


def _dirs(tmp_path: Path) -> dict:
    return {
        "base": tmp_path / "experiments",
        "llm_base": tmp_path / "llm",
        "configs_base": tmp_path / "configs",
        "reports_base": tmp_path / "reports",
    }


def _stub_state(**overrides) -> dict:
    state = {
        "experiment_name": "exp_a",
        "baseline_artefacts_exist": True,
        "metadata_exists": True,
        "metrics_exists": True,
        "context_ready": True,
        "review_exists": False,
        "proposal_exists": False,
        "draft_exists": False,
        "latest_draft_id": None,
        "latest_draft_approved": False,
        "proposed_name": None,
        "rendered_yaml_exists": False,
        "rendered_yaml_path": None,
        "revised_artefacts_exist": False,
        "report_path": None,
        "plots_dir": None,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# _workflow_progression_step / next_tool_and_display — phase derivation
# ---------------------------------------------------------------------------


class TestPhaseDerivation:
    @pytest.mark.parametrize(
        "state,expected_phase,expected_tool",
        [
            (_stub_state(context_ready=False, baseline_artefacts_exist=False),
             WorkflowPhase.NEEDS_BASELINE_EXECUTION, "execute_approved_config"),
            (_stub_state(review_exists=False),
             WorkflowPhase.NEEDS_REVIEW, "run_experiment_review"),
            (_stub_state(review_exists=True, proposal_exists=False),
             WorkflowPhase.NEEDS_PROPOSAL, "generate_iteration_proposal"),
            (_stub_state(review_exists=True, proposal_exists=True, draft_exists=False),
             WorkflowPhase.NEEDS_DRAFT, "generate_experiment_draft"),
            (_stub_state(review_exists=True, proposal_exists=True, draft_exists=True,
                         latest_draft_id="d1", latest_draft_approved=False),
             WorkflowPhase.NEEDS_APPROVAL, "approve_experiment_draft"),
            (_stub_state(review_exists=True, proposal_exists=True, draft_exists=True,
                         latest_draft_id="d1", latest_draft_approved=True,
                         proposed_name="exp_a_v2", rendered_yaml_exists=False),
             WorkflowPhase.NEEDS_RENDER, "render_draft_to_yaml"),
            (_stub_state(review_exists=True, proposal_exists=True, draft_exists=True,
                         latest_draft_id="d1", latest_draft_approved=True,
                         proposed_name="exp_a_v2", rendered_yaml_exists=True,
                         revised_artefacts_exist=False),
             WorkflowPhase.NEEDS_EXECUTION, "execute_approved_config"),
            (_stub_state(review_exists=True, proposal_exists=True, draft_exists=True,
                         latest_draft_id="d1", latest_draft_approved=True,
                         proposed_name="exp_a_v2", rendered_yaml_exists=True,
                         revised_artefacts_exist=True),
             WorkflowPhase.NEEDS_POST_RUN_REVIEW, "review_post_run_result"),
        ],
    )
    def test_phase_and_next_tool(self, state, expected_phase, expected_tool):
        phase, next_tool, _fragment = _workflow_progression_step(state)
        assert phase == expected_phase
        assert next_tool == expected_tool

    def test_needs_baseline_execution_and_needs_execution_are_distinct(self):
        """Both branches recommend execute_approved_config, but the phase
        names must differ — 'never executed' vs 'approved revision awaiting
        its first execution' are different points in the same session's
        lifecycle (design proposal §9, row 7b)."""
        never_run = _stub_state(context_ready=False, baseline_artefacts_exist=False)
        awaiting_rerun = _stub_state(
            review_exists=True, proposal_exists=True, draft_exists=True,
            latest_draft_id="d1", latest_draft_approved=True,
            proposed_name="exp_a_v2", rendered_yaml_exists=True,
            revised_artefacts_exist=False,
        )
        phase1, tool1, _ = _workflow_progression_step(never_run)
        phase2, tool2, _ = _workflow_progression_step(awaiting_rerun)
        assert tool1 == tool2 == "execute_approved_config"
        assert phase1 != phase2
        assert phase1 == WorkflowPhase.NEEDS_BASELINE_EXECUTION
        assert phase2 == WorkflowPhase.NEEDS_EXECUTION

    def test_next_tool_and_display_matches_prior_zeto_server_format(self):
        """Byte-for-byte regression against the removed
        zeto_server._workflow_next_and_display format."""
        state = _stub_state(review_exists=True, proposal_exists=True, draft_exists=False)
        next_tool, display = next_tool_and_display(state)
        assert next_tool == "generate_experiment_draft"
        assert display == (
            "[exp_a] proposal exists; draft missing. "
            "Next suggested action: generate_experiment_draft."
        )

    def test_next_tool_and_display_includes_draft_id_in_approval_fragment(self):
        state = _stub_state(
            review_exists=True, proposal_exists=True, draft_exists=True,
            latest_draft_id="d7", latest_draft_approved=False,
        )
        _next_tool, display = next_tool_and_display(state)
        assert "d7" in display


# ---------------------------------------------------------------------------
# Owner / status mapping completeness
# ---------------------------------------------------------------------------


class TestOwnerAndStatusMappings:
    def test_every_phase_has_an_owner(self):
        for phase in WorkflowPhase:
            assert phase in _WORKFLOW_OWNER_BY_PHASE, phase

    def test_every_phase_has_a_status(self):
        for phase in WorkflowPhase:
            assert phase in _WORKFLOW_STATUS_BY_PHASE, phase

    def test_owner_is_not_an_agent_name(self):
        """Design Principle 4 — Workflow Owner, not Current Agent. None of the
        five owner names should look like an agent identity."""
        forbidden_substrings = ("agent", "supervisor", "llm")
        for owner in _WORKFLOW_OWNER_BY_PHASE.values():
            low = owner.lower()
            for bad in forbidden_substrings:
                assert bad not in low, f"{owner!r} reads like an agent name"

    def test_approval_phase_is_waiting_for_user(self):
        assert _WORKFLOW_STATUS_BY_PHASE[WorkflowPhase.NEEDS_APPROVAL] == "waiting_for_user"

    def test_advisory_phases_are_needs_next_step(self):
        advisory = (
            WorkflowPhase.NEEDS_REVIEW, WorkflowPhase.NEEDS_PROPOSAL,
            WorkflowPhase.NEEDS_DRAFT, WorkflowPhase.NEEDS_RENDER,
            WorkflowPhase.NEEDS_POST_RUN_REVIEW,
        )
        for phase in advisory:
            assert _WORKFLOW_STATUS_BY_PHASE[phase] == "needs_next_step"

    def test_idle_status(self):
        assert _WORKFLOW_STATUS_BY_PHASE[WorkflowPhase.IDLE] == "idle"


# ---------------------------------------------------------------------------
# compute_workflow_trace() — the derived projection
# ---------------------------------------------------------------------------


class TestComputeWorkflowTrace:
    def test_unknown_session_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="no_such_session"):
            compute_workflow_trace("no_such_session", sessions_base=tmp_path)

    def test_new_session_defaults_to_baseline_execution_phase(self, tmp_path: Path):
        session = create_research_session(
            root_experiment=_EXP, research_goal="test", sessions_base=tmp_path,
        )
        trace = compute_workflow_trace(
            session.session_id, sessions_base=tmp_path, **_dirs(tmp_path),
        )
        assert isinstance(trace, WorkflowTrace)
        assert trace.workflow_id == session.session_id
        assert trace.experiment_name == _EXP
        assert trace.current_phase == WorkflowPhase.NEEDS_BASELINE_EXECUTION.value
        assert trace.current_owner == "Render & Execution Engine"
        assert trace.status == "waiting_for_user"
        assert trace.current_tool == "execute_approved_config"
        assert trace.execution is None

    def test_phase_advances_as_artefacts_appear(self, tmp_path: Path):
        session = create_research_session(
            root_experiment=_EXP, research_goal="test", sessions_base=tmp_path,
        )
        d = _dirs(tmp_path)
        (d["base"] / _EXP).mkdir(parents=True)
        (d["base"] / _EXP / "metadata.json").write_text("{}")
        (d["base"] / _EXP / "metrics.json").write_text("{}")

        trace = compute_workflow_trace(session.session_id, sessions_base=tmp_path, **d)
        assert trace.current_phase == WorkflowPhase.NEEDS_REVIEW.value
        assert trace.current_owner == "Review Engine"
        assert trace.status == "needs_next_step"

    def test_current_action_reflects_last_session_event(self, tmp_path: Path):
        session = create_research_session(
            root_experiment=_EXP, research_goal="test", sessions_base=tmp_path,
        )
        session = record_session_event(
            session, SessionEventType.DRAFT_GENERATED, _EXP,
            data={"draft_id": "d1"}, sessions_base=tmp_path,
        )
        trace = compute_workflow_trace(
            session.session_id, sessions_base=tmp_path, **_dirs(tmp_path),
        )
        assert trace.current_action == SessionEventType.DRAFT_GENERATED

    def test_no_events_leaves_current_action_none(self, tmp_path: Path):
        session = create_research_session(
            root_experiment=_EXP, research_goal="test", sessions_base=tmp_path,
        )
        trace = compute_workflow_trace(
            session.session_id, sessions_base=tmp_path, **_dirs(tmp_path),
        )
        assert trace.current_action is None

    def test_experiment_name_override_takes_precedence(self, tmp_path: Path):
        session = create_research_session(
            root_experiment=_EXP, research_goal="test", sessions_base=tmp_path,
        )
        d = _dirs(tmp_path)
        (d["base"] / "other_exp").mkdir(parents=True)
        (d["base"] / "other_exp" / "metadata.json").write_text("{}")
        (d["base"] / "other_exp" / "metrics.json").write_text("{}")

        trace = compute_workflow_trace(
            session.session_id, experiment_name="other_exp", sessions_base=tmp_path, **d,
        )
        assert trace.experiment_name == "other_exp"
        assert trace.current_phase == WorkflowPhase.NEEDS_REVIEW.value

    def test_no_active_experiment_is_idle(self, tmp_path: Path):
        idle_session = ResearchSession(
            session_id="s-idle", research_goal="test", root_experiment=_EXP,
            active_experiment="", status="active",
            created_at="2026-01-01T00:00:00+00:00", updated_at="2026-01-01T00:00:00+00:00",
        )
        with patch(
            "src.orchestration.api.research_api.load_research_session",
            return_value=idle_session,
        ):
            trace = compute_workflow_trace("s-idle", sessions_base=tmp_path)
        assert trace.current_phase == WorkflowPhase.IDLE.value
        assert trace.current_owner == "Idle"
        assert trace.status == "idle"
        assert trace.experiment_name is None
        assert trace.current_tool is None
        assert trace.execution is None

    def test_metadata_includes_latest_draft_id_when_present(self, tmp_path: Path):
        session = create_research_session(
            root_experiment=_EXP, research_goal="test", sessions_base=tmp_path,
        )
        d = _dirs(tmp_path)
        (d["base"] / _EXP).mkdir(parents=True)
        (d["base"] / _EXP / "metadata.json").write_text("{}")
        (d["base"] / _EXP / "metrics.json").write_text("{}")
        llm_exp = d["llm_base"] / _EXP
        llm_exp.mkdir(parents=True)
        (llm_exp / "llm_review.json").write_text("{}")
        (llm_exp / "iteration_proposal.json").write_text("{}")
        (llm_exp / "draft_x.json").write_text(json.dumps({
            "draft_id": "d9", "approved": False, "proposed_name": "exp_a_v2",
            "generated_at": "2026-02-01T00:00:00+00:00",
        }))

        trace = compute_workflow_trace(session.session_id, sessions_base=tmp_path, **d)
        assert trace.current_phase == WorkflowPhase.NEEDS_APPROVAL.value
        assert trace.metadata["latest_draft_id"] == "d9"

    def test_metadata_omits_none_valued_keys(self, tmp_path: Path):
        session = create_research_session(
            root_experiment=_EXP, research_goal="test", sessions_base=tmp_path,
        )
        trace = compute_workflow_trace(
            session.session_id, sessions_base=tmp_path, **_dirs(tmp_path),
        )
        assert "latest_draft_id" not in trace.metadata
        assert "proposed_name" not in trace.metadata

    def test_to_dict_is_json_serializable(self, tmp_path: Path):
        session = create_research_session(
            root_experiment=_EXP, research_goal="test", sessions_base=tmp_path,
        )
        trace = compute_workflow_trace(
            session.session_id, sessions_base=tmp_path, **_dirs(tmp_path),
        )
        json.dumps(trace.to_dict())  # must not raise

    def test_execution_is_always_none_in_this_phase_of_the_implementation(self, tmp_path: Path):
        """Design proposal §7/§13 Step 4 (nesting Execution Trace into a live
        execute_approved_config response) is not implemented here — only
        Steps 1-3. execution must stay None regardless of phase."""
        session = create_research_session(
            root_experiment=_EXP, research_goal="test", sessions_base=tmp_path,
        )
        d = _dirs(tmp_path)
        for state_setup in [
            {},  # NEEDS_BASELINE_EXECUTION
        ]:
            trace = compute_workflow_trace(session.session_id, sessions_base=tmp_path, **d)
            assert trace.execution is None
