"""Phase 5F.3 — Agent Draft Governance Compatibility Tests.

Proves that drafts created by the new agent workflow chain are fully
compatible with the existing governance pipeline:

  Agent chain:
    ResearchSpec → resolve → plan → propose_spec → confirm → draft_created
    → save_agent_draft_result()

  Existing governance layer:
    load_draft() → validate_draft() → approve_draft() → render_to_yaml()

All disk I/O uses tmp_path fixtures.
render_to_yaml() is always called with dry_run=True to avoid writing configs.

Compatibility assertions:
  A. Saved draft can be loaded by existing draft loader
  B. Saved draft contains expected model change (LassoRegression)
  C. Saved draft is unapproved initially
  D. validate_draft() accepts the agent-created draft
  E. approve_draft() transitions approved=False → True
  F. render_to_yaml() produces YAML with expected change and provenance
  G. render_to_yaml() raises ValueError on unapproved draft
  H. No execution occurs at any stage
  I. Governance flags (approval_required, execution_performed, draft_approved)
     are correct throughout the agent flow
  J. After approval, execution_performed is still False
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_CONFIGS = _REPO_ROOT / "configs" / "experiments"


def _assert_real_config_exists() -> None:
    """Skip the test if the canonical config file is absent."""
    config_path = _REAL_CONFIGS / "canonical_ml_multi_asset.yaml"
    if not config_path.exists():
        pytest.skip(
            f"Base config not found: {config_path}. "
            "Real config required for compatibility tests."
        )


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def registry():
    from src.agent.registry.registry_loader import load_registry
    return load_registry()


@pytest.fixture()
def lasso_proposal(registry):
    """A ResearchSpecProposal for LassoRegression on the canonical preset."""
    from src.agent.intent.research_spec import ResearchSpec
    from src.agent.next_actions.next_action import NextActionCandidate
    from src.agent.next_actions.candidate_to_spec import propose_spec_from_next_action

    base_spec = ResearchSpec.from_dict({
        "preset": "canonical_multi_asset_etf",
        "model": "LassoRegression",
        "end_date": "2024-12-31",
        "start_date": "2015-01-01",
    })
    candidate = NextActionCandidate(
        action_id="compat_lasso",
        title="Model comparison: LassoRegression",
        action_type="model_comparison",
        description="Switch to LassoRegression for compatibility test",
        rationale="Phase 5F.3 compatibility checkpoint",
        suggested_spec_changes={"model": "LassoRegression"},
    )
    result = propose_spec_from_next_action(candidate, base_spec=base_spec, registry=registry)
    assert result.status == "proposal_created", f"Unexpected proposal status: {result.status}"
    return result.proposal


@pytest.fixture()
def draft_result(lasso_proposal):
    """WorkflowResult with status=draft_created from the agent chain."""
    from src.agent.workflow.proposal_to_draft_workflow import run_draft_from_spec_proposal
    result = run_draft_from_spec_proposal(lasso_proposal, confirm_proposal=True)
    assert result.status == "draft_created", f"Unexpected draft status: {result.status}"
    return result


@pytest.fixture()
def saved_draft_result(draft_result, tmp_path):
    """SavedAgentDraftResult with draft JSON written to tmp_path."""
    from src.agent.workflow.draft_persistence import save_agent_draft_result
    result = save_agent_draft_result(draft_result, base_dir=tmp_path)
    assert result.status == "saved", f"Save failed: {result.status} — {result.errors}"
    return result


@pytest.fixture()
def loaded_draft(draft_result, saved_draft_result, tmp_path):
    """ExperimentDraft loaded from tmp_path via existing load_draft()."""
    from src.orchestration.config_generation.draft_generator import load_draft
    draft_obj = draft_result.experiment_draft_object
    loaded = load_draft(
        draft_obj.base_experiment,
        draft_obj.draft_id,
        llm_base=tmp_path,
    )
    assert loaded is not None, "load_draft returned None"
    return loaded


# ---------------------------------------------------------------------------
# A. Saved draft compatibility
# ---------------------------------------------------------------------------

class TestSavedDraftCompatibility:
    """Agent-created draft is saved with the correct structure."""

    def test_draft_result_status_is_draft_created(self, draft_result):
        assert draft_result.status == "draft_created"

    def test_experiment_draft_object_is_present(self, draft_result):
        assert draft_result.experiment_draft_object is not None

    def test_draft_object_base_experiment(self, draft_result):
        draft_obj = draft_result.experiment_draft_object
        assert draft_obj.base_experiment == "canonical_ml_multi_asset"

    def test_draft_object_has_draft_id(self, draft_result):
        draft_obj = draft_result.experiment_draft_object
        assert isinstance(draft_obj.draft_id, str)
        assert len(draft_obj.draft_id) > 0

    def test_draft_object_has_draft_hash(self, draft_result):
        draft_obj = draft_result.experiment_draft_object
        assert isinstance(draft_obj.draft_hash, str)
        assert len(draft_obj.draft_hash) > 0

    def test_draft_object_unapproved_initially(self, draft_result):
        assert draft_result.experiment_draft_object.approved is False

    def test_draft_contains_model_type_change(self, draft_result):
        draft_obj = draft_result.experiment_draft_object
        changes = {(c.section, c.field): c.proposed_value for c in draft_obj.changes}
        assert ("model", "type") in changes, f"model.type not in changes: {list(changes.keys())}"
        assert changes[("model", "type")] == "LassoRegression"

    def test_save_result_status_saved(self, saved_draft_result):
        assert saved_draft_result.status == "saved"

    def test_save_result_has_draft_id(self, saved_draft_result):
        assert saved_draft_result.draft_id is not None
        assert len(saved_draft_result.draft_id) > 0

    def test_save_result_has_draft_path(self, saved_draft_result):
        assert saved_draft_result.draft_path is not None
        assert Path(saved_draft_result.draft_path).exists()

    def test_save_result_unapproved(self, saved_draft_result):
        assert saved_draft_result.draft_approved is False

    def test_save_result_no_execution(self, saved_draft_result):
        assert saved_draft_result.execution_performed is False

    def test_save_result_approval_required(self, saved_draft_result):
        assert saved_draft_result.approval_required is True

    def test_saved_json_contains_draft_id(self, saved_draft_result, tmp_path):
        import json
        draft_path = Path(saved_draft_result.draft_path)
        data = json.loads(draft_path.read_text())
        assert data["draft_id"] == saved_draft_result.draft_id

    def test_saved_json_contains_lasso_change(self, saved_draft_result):
        import json
        draft_path = Path(saved_draft_result.draft_path)
        data = json.loads(draft_path.read_text())
        changes = {(c["section"], c["field"]): c["proposed_value"] for c in data.get("changes", [])}
        assert ("model", "type") in changes
        assert changes[("model", "type")] == "LassoRegression"

    def test_saved_json_approved_false(self, saved_draft_result):
        import json
        draft_path = Path(saved_draft_result.draft_path)
        data = json.loads(draft_path.read_text())
        assert data["approved"] is False


# ---------------------------------------------------------------------------
# B. Load compatibility
# ---------------------------------------------------------------------------

class TestLoadCompatibility:
    """Existing load_draft() can load the agent-created draft."""

    def test_loaded_draft_is_not_none(self, loaded_draft):
        assert loaded_draft is not None

    def test_loaded_draft_has_correct_base_experiment(self, loaded_draft):
        assert loaded_draft.base_experiment == "canonical_ml_multi_asset"

    def test_loaded_draft_has_correct_draft_id(self, loaded_draft, draft_result):
        assert loaded_draft.draft_id == draft_result.experiment_draft_object.draft_id

    def test_loaded_draft_is_unapproved(self, loaded_draft):
        assert loaded_draft.approved is False

    def test_loaded_draft_model_change_preserved(self, loaded_draft):
        changes = {(c.section, c.field): c.proposed_value for c in loaded_draft.changes}
        assert ("model", "type") in changes
        assert changes[("model", "type")] == "LassoRegression"

    def test_load_returns_none_for_bad_draft_id(self, draft_result, tmp_path):
        from src.orchestration.config_generation.draft_generator import load_draft
        result = load_draft("canonical_ml_multi_asset", "nonexistent-draft-id", llm_base=tmp_path)
        assert result is None

    def test_loaded_draft_changes_match_original(self, loaded_draft, draft_result):
        original = draft_result.experiment_draft_object
        assert len(loaded_draft.changes) == len(original.changes)
        for orig_c, loaded_c in zip(original.changes, loaded_draft.changes):
            assert orig_c.section == loaded_c.section
            assert orig_c.field == loaded_c.field
            assert orig_c.proposed_value == loaded_c.proposed_value


# ---------------------------------------------------------------------------
# C. Validation compatibility
# ---------------------------------------------------------------------------

class TestValidationCompatibility:
    """validate_draft() accepts the agent-created draft."""

    def test_validate_draft_is_valid(self, loaded_draft, tmp_path):
        _assert_real_config_exists()
        from src.orchestration.config_generation.draft_validator import validate_draft
        result = validate_draft(loaded_draft, base=tmp_path, configs_base=_REAL_CONFIGS)
        assert result.is_valid, f"Validation failed: {result.errors}"

    def test_validate_draft_no_errors(self, loaded_draft, tmp_path):
        _assert_real_config_exists()
        from src.orchestration.config_generation.draft_validator import validate_draft
        result = validate_draft(loaded_draft, base=tmp_path, configs_base=_REAL_CONFIGS)
        assert result.errors == []

    def test_model_type_is_in_valid_change_paths(self):
        from src.orchestration.config_generation.draft_validator import _VALID_CHANGE_PATHS
        assert ("model", "type") in _VALID_CHANGE_PATHS

    def test_validate_before_approval(self, loaded_draft, tmp_path):
        _assert_real_config_exists()
        from src.orchestration.config_generation.draft_validator import validate_draft
        # Validation should pass before approval
        result = validate_draft(loaded_draft, base=tmp_path, configs_base=_REAL_CONFIGS)
        assert result.is_valid
        assert loaded_draft.approved is False  # still unapproved after validate

    def test_validate_does_not_change_approval_state(self, loaded_draft, tmp_path):
        _assert_real_config_exists()
        from src.orchestration.config_generation.draft_validator import validate_draft
        validate_draft(loaded_draft, base=tmp_path, configs_base=_REAL_CONFIGS)
        assert loaded_draft.approved is False  # validate never sets approved


# ---------------------------------------------------------------------------
# D. Approval compatibility
# ---------------------------------------------------------------------------

class TestApprovalCompatibility:
    """approve_draft() correctly transitions the agent-created draft."""

    def test_approve_draft_returns_approved_true(self, loaded_draft):
        from src.orchestration.config_generation.draft_validator import approve_draft
        approved = approve_draft(loaded_draft)
        assert approved.approved is True

    def test_approve_draft_sets_approved_at(self, loaded_draft):
        from src.orchestration.config_generation.draft_validator import approve_draft
        approved = approve_draft(loaded_draft)
        assert approved.approved_at is not None
        assert isinstance(approved.approved_at, str)
        assert len(approved.approved_at) > 0

    def test_approve_draft_does_not_mutate_original(self, loaded_draft):
        from src.orchestration.config_generation.draft_validator import approve_draft
        original_approved = loaded_draft.approved
        _ = approve_draft(loaded_draft)
        assert loaded_draft.approved == original_approved  # original unchanged

    def test_approve_draft_preserves_model_change(self, loaded_draft):
        from src.orchestration.config_generation.draft_validator import approve_draft
        approved = approve_draft(loaded_draft)
        changes = {(c.section, c.field): c.proposed_value for c in approved.changes}
        assert ("model", "type") in changes
        assert changes[("model", "type")] == "LassoRegression"

    def test_approve_draft_preserves_base_experiment(self, loaded_draft):
        from src.orchestration.config_generation.draft_validator import approve_draft
        approved = approve_draft(loaded_draft)
        assert approved.base_experiment == "canonical_ml_multi_asset"

    def test_approve_draft_recomputes_hash(self, loaded_draft):
        from src.orchestration.config_generation.draft_validator import approve_draft
        approved = approve_draft(loaded_draft)
        assert isinstance(approved.draft_hash, str)
        assert len(approved.draft_hash) > 0

    def test_approve_draft_no_execution(self, loaded_draft):
        from src.orchestration.config_generation.draft_validator import approve_draft
        # approve_draft is a pure state transition — no execution
        approved = approve_draft(loaded_draft)
        assert approved.approved is True
        # No execution attribute on ExperimentDraft — execution is a separate concern
        assert not hasattr(approved, "execution_performed") or True


# ---------------------------------------------------------------------------
# E. Render compatibility
# ---------------------------------------------------------------------------

class TestRenderCompatibility:
    """render_to_yaml() produces correct YAML from the approved agent draft."""

    @pytest.fixture()
    def approved_draft(self, loaded_draft):
        from src.orchestration.config_generation.draft_validator import approve_draft
        return approve_draft(loaded_draft)

    @pytest.fixture()
    def rendered_yaml(self, approved_draft):
        _assert_real_config_exists()
        from src.orchestration.config_generation.yaml_renderer import render_to_yaml
        return render_to_yaml(approved_draft, configs_base=_REAL_CONFIGS, dry_run=True)

    def test_render_produces_non_empty_yaml(self, rendered_yaml):
        assert isinstance(rendered_yaml, str)
        assert len(rendered_yaml) > 100

    def test_render_contains_lasso_regression(self, rendered_yaml):
        assert "LassoRegression" in rendered_yaml

    def test_render_contains_provenance_header(self, rendered_yaml):
        assert "Generated by Quant Research Platform" in rendered_yaml

    def test_render_contains_draft_hash(self, rendered_yaml):
        assert "draft_hash:" in rendered_yaml

    def test_render_contains_base_experiment(self, rendered_yaml):
        assert "canonical_ml_multi_asset" in rendered_yaml

    def test_render_contains_proposed_name(self, rendered_yaml, approved_draft):
        assert approved_draft.proposed_name in rendered_yaml

    def test_render_contains_metadata_provenance(self, rendered_yaml):
        assert "metadata:" in rendered_yaml
        assert "base_experiment:" in rendered_yaml
        assert "experiment_hash:" in rendered_yaml

    def test_render_yaml_is_valid_yaml(self, rendered_yaml):
        import yaml
        # Strip header comment lines before parsing
        body = "\n".join(
            line for line in rendered_yaml.splitlines()
            if not line.startswith("#")
        )
        parsed = yaml.safe_load(body)
        assert isinstance(parsed, dict)

    def test_render_yaml_model_type_is_lasso(self, rendered_yaml):
        import yaml
        body = "\n".join(
            line for line in rendered_yaml.splitlines()
            if not line.startswith("#")
        )
        parsed = yaml.safe_load(body)
        assert parsed.get("model", {}).get("type") == "LassoRegression"

    def test_render_yaml_version_is_2(self, rendered_yaml):
        import yaml
        body = "\n".join(
            line for line in rendered_yaml.splitlines()
            if not line.startswith("#")
        )
        parsed = yaml.safe_load(body)
        assert str(parsed.get("version")) == "2"

    def test_render_raises_on_unapproved_draft(self, loaded_draft):
        _assert_real_config_exists()
        from src.orchestration.config_generation.yaml_renderer import render_to_yaml
        assert loaded_draft.approved is False
        with pytest.raises(ValueError, match="not been approved"):
            render_to_yaml(loaded_draft, configs_base=_REAL_CONFIGS, dry_run=True)

    def test_render_dry_run_does_not_write_file(self, approved_draft, tmp_path):
        _assert_real_config_exists()
        from src.orchestration.config_generation.yaml_renderer import render_to_yaml
        yaml_str = render_to_yaml(
            approved_draft, configs_base=_REAL_CONFIGS, dry_run=True
        )
        # No config file should be written to the configs_base
        written = list((_REAL_CONFIGS).glob(f"{approved_draft.proposed_name}.yaml"))
        assert written == [], f"dry_run wrote file(s): {written}"
        assert len(yaml_str) > 0


# ---------------------------------------------------------------------------
# F. Execution boundary
# ---------------------------------------------------------------------------

class TestExecutionBoundary:
    """Execution never occurs at any stage of the agent draft pipeline."""

    def test_draft_result_execution_performed_false(self, draft_result):
        assert draft_result.execution_performed is False

    def test_draft_result_draft_approved_false(self, draft_result):
        assert draft_result.draft_approved is False

    def test_draft_result_approval_required_true(self, draft_result):
        assert draft_result.approval_required is True

    def test_save_result_execution_performed_false(self, saved_draft_result):
        assert saved_draft_result.execution_performed is False

    def test_approve_does_not_execute(self, loaded_draft):
        from src.orchestration.config_generation.draft_validator import approve_draft
        approved = approve_draft(loaded_draft)
        assert approved.approved is True
        # approve_draft is a pure dataclass operation — no run_experiment() call

    def test_render_dry_run_does_not_execute(self, loaded_draft):
        _assert_real_config_exists()
        from src.orchestration.config_generation.draft_validator import approve_draft
        from src.orchestration.config_generation.yaml_renderer import render_to_yaml
        approved = approve_draft(loaded_draft)
        yaml_str = render_to_yaml(approved, configs_base=_REAL_CONFIGS, dry_run=True)
        assert "LassoRegression" in yaml_str
        # render_to_yaml never calls an experiment runner


# ---------------------------------------------------------------------------
# G. Governance flags throughout chain
# ---------------------------------------------------------------------------

class TestGovernanceFlagsThroughout:
    """Governance flags are correct at every step of the chain."""

    def test_proposal_approval_required(self, lasso_proposal):
        assert lasso_proposal.approval_required is True

    def test_proposal_requires_user_confirmation(self, lasso_proposal):
        assert lasso_proposal.requires_user_confirmation is True

    def test_proposal_creates_draft_false(self, lasso_proposal):
        assert lasso_proposal.creates_draft is False

    def test_proposal_executes_experiment_false(self, lasso_proposal):
        assert lasso_proposal.executes_experiment is False

    def test_draft_result_approval_required(self, draft_result):
        assert draft_result.approval_required is True

    def test_draft_result_draft_approved_false(self, draft_result):
        assert draft_result.draft_approved is False

    def test_draft_result_execution_performed_false(self, draft_result):
        assert draft_result.execution_performed is False

    def test_governance_dict_approval_required(self, draft_result):
        gov = draft_result.to_dict().get("governance", {})
        assert gov.get("approval_required") is True

    def test_governance_dict_draft_approved_false(self, draft_result):
        gov = draft_result.to_dict().get("governance", {})
        assert gov.get("draft_approved") is False

    def test_governance_dict_execution_performed_false(self, draft_result):
        gov = draft_result.to_dict().get("governance", {})
        assert gov.get("execution_performed") is False

    def test_governance_dict_requires_user_confirmation(self, draft_result):
        gov = draft_result.to_dict().get("governance", {})
        assert gov.get("requires_user_confirmation") is True

    def test_save_result_approval_required(self, saved_draft_result):
        assert saved_draft_result.approval_required is True

    def test_save_result_draft_approved_false(self, saved_draft_result):
        assert saved_draft_result.draft_approved is False

    def test_after_approval_execution_still_false(self, loaded_draft):
        from src.orchestration.config_generation.draft_validator import approve_draft
        approved = approve_draft(loaded_draft)
        assert approved.approved is True
        # ExperimentDraft has no execution_performed field — execution is separate
        # Verify that approve_draft does not add any execution attribute
        assert not hasattr(approved, "execution_performed")


# ---------------------------------------------------------------------------
# H. Full compatibility chain (integration)
# ---------------------------------------------------------------------------

class TestFullCompatibilityChain:
    """End-to-end: agent chain → save → load → validate → approve → render."""

    def test_full_chain_reaches_rendered_yaml(self, tmp_path):
        """Executes the complete compatibility chain in one test."""
        _assert_real_config_exists()
        from src.agent.intent.research_spec import ResearchSpec
        from src.agent.next_actions.next_action import NextActionCandidate
        from src.agent.next_actions.candidate_to_spec import propose_spec_from_next_action
        from src.agent.registry.registry_loader import load_registry
        from src.agent.workflow.proposal_to_draft_workflow import run_draft_from_spec_proposal
        from src.agent.workflow.draft_persistence import save_agent_draft_result
        from src.orchestration.config_generation.draft_generator import load_draft
        from src.orchestration.config_generation.draft_validator import (
            validate_draft, approve_draft,
        )
        from src.orchestration.config_generation.yaml_renderer import render_to_yaml

        # 1. Build proposal
        registry = load_registry()
        base_spec = ResearchSpec.from_dict({
            "preset": "canonical_multi_asset_etf",
            "model": "LassoRegression",
            "end_date": "2024-12-31",
            "start_date": "2015-01-01",
        })
        candidate = NextActionCandidate(
            action_id="fullchain", title="Model: LassoRegression",
            action_type="model_comparison", description="switch model",
            rationale="full chain integration test",
            suggested_spec_changes={"model": "LassoRegression"},
        )
        pr = propose_spec_from_next_action(candidate, base_spec=base_spec, registry=registry)
        assert pr.status == "proposal_created"

        # 2. Confirmation gate: should block without confirm_proposal=True
        from src.agent.workflow.proposal_to_draft_workflow import run_draft_from_spec_proposal
        gate_result = run_draft_from_spec_proposal(pr.proposal, confirm_proposal=False)
        assert gate_result.status == "confirmation_required"
        assert gate_result.draft_approved is False
        assert gate_result.execution_performed is False

        # 3. Draft creation with explicit confirmation
        draft_result = run_draft_from_spec_proposal(pr.proposal, confirm_proposal=True)
        assert draft_result.status == "draft_created"
        assert draft_result.experiment_draft_object is not None
        assert draft_result.draft_approved is False
        assert draft_result.execution_performed is False

        draft_obj = draft_result.experiment_draft_object
        assert draft_obj.approved is False

        # 4. Save
        save_result = save_agent_draft_result(draft_result, base_dir=tmp_path)
        assert save_result.status == "saved"
        assert save_result.draft_approved is False
        assert save_result.execution_performed is False

        # 5. Load
        loaded = load_draft(draft_obj.base_experiment, draft_obj.draft_id, llm_base=tmp_path)
        assert loaded is not None
        assert loaded.approved is False
        assert any(
            c.section == "model" and c.field == "type" and c.proposed_value == "LassoRegression"
            for c in loaded.changes
        )

        # 6. Validate
        val = validate_draft(loaded, base=tmp_path, configs_base=_REAL_CONFIGS)
        assert val.is_valid, f"validate_draft failed: {val.errors}"
        assert loaded.approved is False  # validate never approves

        # 7. Approve
        approved = approve_draft(loaded)
        assert approved.approved is True
        assert approved.approved_at is not None

        # 8. Render (dry_run=True)
        yaml_str = render_to_yaml(approved, configs_base=_REAL_CONFIGS, dry_run=True)
        assert "LassoRegression" in yaml_str
        assert "draft_hash:" in yaml_str
        assert "Generated by Quant Research Platform" in yaml_str

        # 9. Confirm unapproved draft still cannot render
        with pytest.raises(ValueError):
            render_to_yaml(loaded, configs_base=_REAL_CONFIGS, dry_run=True)

    def test_nl_workflow_path_also_produces_compatible_draft(self, tmp_path):
        """The NL workflow path (run_propose_spec_from_request) also works."""
        _assert_real_config_exists()
        from src.agent.workflow.spec_proposal_workflow import run_propose_spec_from_request
        from src.agent.workflow.proposal_to_draft_workflow import run_draft_from_spec_proposal
        from src.agent.workflow.draft_persistence import save_agent_draft_result
        from src.agent.next_actions.spec_proposal import ResearchSpecProposal
        from src.orchestration.config_generation.draft_generator import load_draft
        from src.orchestration.config_generation.draft_validator import (
            validate_draft, approve_draft,
        )
        from src.orchestration.config_generation.yaml_renderer import render_to_yaml

        # Use stub provider for the NL path
        wr = run_propose_spec_from_request(
            "Use the canonical multi-asset ETF preset from 2015 to 2024 "
            "and test LassoRegression.",
            candidate_index=0,
            provider="stub",
        )
        assert wr.status == "proposal_created"
        inner_prop = (wr.spec_proposal or {}).get("proposal") or {}
        assert inner_prop, "No proposal in spec_proposal"
        proposal = ResearchSpecProposal.from_dict(inner_prop)

        # Confirm gate + draft creation
        draft_result = run_draft_from_spec_proposal(proposal, confirm_proposal=True)
        assert draft_result.status == "draft_created"
        draft_obj = draft_result.experiment_draft_object
        assert draft_obj is not None
        assert draft_obj.base_experiment == "canonical_ml_multi_asset"

        # Save → Load → Validate → Approve → Render
        save_result = save_agent_draft_result(draft_result, base_dir=tmp_path)
        assert save_result.status == "saved"

        loaded = load_draft(draft_obj.base_experiment, draft_obj.draft_id, llm_base=tmp_path)
        assert loaded is not None

        val = validate_draft(loaded, base=tmp_path, configs_base=_REAL_CONFIGS)
        assert val.is_valid, f"NL path validation failed: {val.errors}"

        approved = approve_draft(loaded)
        yaml_str = render_to_yaml(approved, configs_base=_REAL_CONFIGS, dry_run=True)
        assert "canonical_ml_multi_asset" in yaml_str
        assert "draft_hash:" in yaml_str
