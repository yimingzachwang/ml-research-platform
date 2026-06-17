"""Tests for src/agent/workflow/draft_persistence.py (Phase 3D).

Covers:
  - SavedAgentDraftResult dataclass and to_dict() contract
  - save_agent_draft_result() happy path (saved)
  - not_saved: wrong workflow status
  - not_saved: hard-unsupported fields present
  - not_saved: experiment_draft_object is None
  - error: save_draft raises an exception
  - soft-unsupported date fields produce a warning (not an error)
  - governance flags always: approval_required=True, execution_performed=False,
    draft_approved=False

All tests are in-memory.  save_draft() is always mocked — no disk I/O.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.workflow.draft_persistence import (
    SavedAgentDraftResult,
    save_agent_draft_result,
)
from src.agent.workflow.workflow_result import WorkflowResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOD_SAVE = "src.agent.workflow.draft_persistence._save_draft"


def _mock_experiment_draft(
    draft_id: str = "d001",
    base_experiment: str = "canonical_ml_multi_asset",
    proposed_name: str = "canonical_ml_multi_asset_v2",
) -> MagicMock:
    d = MagicMock()
    d.draft_id = draft_id
    d.base_experiment = base_experiment
    d.proposed_name = proposed_name
    return d


def _draft_created_result(
    hard_unsupported: list[str] | None = None,
    soft_unsupported: list[str] | None = None,
    experiment_draft: MagicMock | None = None,
) -> WorkflowResult:
    wr = WorkflowResult(
        status="draft_created",
        summary="Draft created.",
        hard_unsupported_fields=hard_unsupported or [],
        unsupported_fields=soft_unsupported or [],
        experiment_draft_object=experiment_draft or _mock_experiment_draft(),
    )
    return wr


# ---------------------------------------------------------------------------
# SavedAgentDraftResult — dataclass contract
# ---------------------------------------------------------------------------


class TestSavedAgentDraftResultContract:
    def test_to_dict_contains_all_required_keys(self):
        r = SavedAgentDraftResult(status="saved", summary="ok")
        d = r.to_dict()
        required = {
            "status", "draft_id", "draft_path", "base_experiment", "proposed_name",
            "approval_required", "execution_performed", "draft_approved",
            "warnings", "errors", "summary",
        }
        assert required.issubset(d.keys()), f"Missing: {required - set(d.keys())}"

    def test_governance_defaults(self):
        r = SavedAgentDraftResult(status="saved")
        assert r.approval_required is True
        assert r.execution_performed is False
        assert r.draft_approved is False

    def test_governance_in_to_dict(self):
        r = SavedAgentDraftResult(status="saved")
        d = r.to_dict()
        assert d["approval_required"] is True
        assert d["execution_performed"] is False
        assert d["draft_approved"] is False

    def test_status_propagated(self):
        for s in ("saved", "not_saved", "error"):
            assert SavedAgentDraftResult(status=s).to_dict()["status"] == s

    def test_warnings_and_errors_default_empty(self):
        r = SavedAgentDraftResult(status="not_saved")
        assert r.warnings == []
        assert r.errors == []


# ---------------------------------------------------------------------------
# save_agent_draft_result — happy path
# ---------------------------------------------------------------------------


class TestSaveAgentDraftResultHappyPath:
    def test_returns_saved_status(self):
        wr = _draft_created_result()
        fake_path = Path("/tmp/results/llm_reviews/canonical/draft_d001.json")
        with patch(_MOD_SAVE, return_value=fake_path):
            result = save_agent_draft_result(wr)
        assert result.status == "saved"

    def test_draft_id_populated(self):
        wr = _draft_created_result()
        fake_path = Path("/tmp/draft_d001.json")
        with patch(_MOD_SAVE, return_value=fake_path):
            result = save_agent_draft_result(wr)
        assert result.draft_id == "d001"

    def test_draft_path_as_string(self):
        wr = _draft_created_result()
        fake_path = Path("/tmp/draft_d001.json")
        with patch(_MOD_SAVE, return_value=fake_path):
            result = save_agent_draft_result(wr)
        assert result.draft_path == str(fake_path)

    def test_base_experiment_propagated(self):
        wr = _draft_created_result()
        fake_path = Path("/tmp/draft.json")
        with patch(_MOD_SAVE, return_value=fake_path):
            result = save_agent_draft_result(wr)
        assert result.base_experiment == "canonical_ml_multi_asset"

    def test_proposed_name_propagated(self):
        wr = _draft_created_result()
        fake_path = Path("/tmp/draft.json")
        with patch(_MOD_SAVE, return_value=fake_path):
            result = save_agent_draft_result(wr)
        assert result.proposed_name == "canonical_ml_multi_asset_v2"

    def test_governance_flags_on_saved(self):
        wr = _draft_created_result()
        with patch(_MOD_SAVE, return_value=Path("/tmp/d.json")):
            result = save_agent_draft_result(wr)
        assert result.approval_required is True
        assert result.execution_performed is False
        assert result.draft_approved is False

    def test_no_errors_on_success(self):
        wr = _draft_created_result()
        with patch(_MOD_SAVE, return_value=Path("/tmp/d.json")):
            result = save_agent_draft_result(wr)
        assert result.errors == []

    def test_save_draft_called_with_experiment_draft(self):
        mock_draft = _mock_experiment_draft(draft_id="xyz")
        wr = _draft_created_result(experiment_draft=mock_draft)
        with patch(_MOD_SAVE, return_value=Path("/tmp/d.json")) as mock_save:
            save_agent_draft_result(wr)
        mock_save.assert_called_once_with(mock_draft, llm_base=None)

    def test_base_dir_forwarded_to_save_draft(self):
        wr = _draft_created_result()
        custom_base = Path("/custom/base")
        with patch(_MOD_SAVE, return_value=Path("/tmp/d.json")) as mock_save:
            save_agent_draft_result(wr, base_dir=custom_base)
        _, kwargs = mock_save.call_args
        assert kwargs["llm_base"] == custom_base

    def test_summary_contains_draft_id(self):
        wr = _draft_created_result()
        with patch(_MOD_SAVE, return_value=Path("/tmp/draft_d001.json")):
            result = save_agent_draft_result(wr)
        assert "d001" in result.summary

    def test_to_dict_on_saved_result(self):
        wr = _draft_created_result()
        with patch(_MOD_SAVE, return_value=Path("/tmp/d.json")):
            result = save_agent_draft_result(wr)
        d = result.to_dict()
        assert d["status"] == "saved"
        assert d["approval_required"] is True
        assert d["execution_performed"] is False
        assert d["draft_approved"] is False


# ---------------------------------------------------------------------------
# save_agent_draft_result — not_saved: wrong status
# ---------------------------------------------------------------------------


class TestNotSavedWrongStatus:
    @pytest.mark.parametrize("bad_status", [
        "planned", "needs_review", "needs_user_input",
        "invalid", "unsupported_plan", "validation_failed", "error",
    ])
    def test_non_draft_created_returns_not_saved(self, bad_status: str):
        wr = WorkflowResult(status=bad_status, summary="not a draft")
        result = save_agent_draft_result(wr)
        assert result.status == "not_saved"

    def test_not_saved_errors_non_empty(self):
        wr = WorkflowResult(status="planned", summary="planning only")
        result = save_agent_draft_result(wr)
        assert result.errors

    def test_save_draft_not_called_on_wrong_status(self):
        wr = WorkflowResult(status="planned", summary="planning only")
        with patch(_MOD_SAVE) as mock_save:
            save_agent_draft_result(wr)
        mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# save_agent_draft_result — not_saved: hard-unsupported fields
# ---------------------------------------------------------------------------


class TestNotSavedHardUnsupported:
    def test_hard_unsupported_returns_not_saved(self):
        wr = _draft_created_result(hard_unsupported=["universe"])
        result = save_agent_draft_result(wr)
        assert result.status == "not_saved"

    def test_hard_unsupported_errors_mention_fields(self):
        wr = _draft_created_result(hard_unsupported=["tickers", "frequency"])
        result = save_agent_draft_result(wr)
        assert any("tickers" in e or "frequency" in e for e in result.errors)

    def test_save_draft_not_called_on_hard_unsupported(self):
        wr = _draft_created_result(hard_unsupported=["universe"])
        with patch(_MOD_SAVE) as mock_save:
            save_agent_draft_result(wr)
        mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# save_agent_draft_result — not_saved: no experiment_draft_object
# ---------------------------------------------------------------------------


class TestNotSavedNoDraftObject:
    def test_none_draft_object_returns_not_saved(self):
        wr = WorkflowResult(
            status="draft_created",
            summary="created",
            experiment_draft_object=None,
        )
        result = save_agent_draft_result(wr)
        assert result.status == "not_saved"

    def test_none_draft_object_has_errors(self):
        wr = WorkflowResult(
            status="draft_created",
            summary="created",
            experiment_draft_object=None,
        )
        result = save_agent_draft_result(wr)
        assert result.errors

    def test_save_draft_not_called_on_missing_draft(self):
        wr = WorkflowResult(
            status="draft_created",
            summary="created",
            experiment_draft_object=None,
        )
        with patch(_MOD_SAVE) as mock_save:
            save_agent_draft_result(wr)
        mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# save_agent_draft_result — error: save_draft raises
# ---------------------------------------------------------------------------


class TestSaveError:
    def test_exception_returns_error_status(self):
        wr = _draft_created_result()
        with patch(_MOD_SAVE, side_effect=OSError("disk full")):
            result = save_agent_draft_result(wr)
        assert result.status == "error"

    def test_exception_message_in_errors(self):
        wr = _draft_created_result()
        with patch(_MOD_SAVE, side_effect=RuntimeError("boom")):
            result = save_agent_draft_result(wr)
        assert any("boom" in e for e in result.errors)

    def test_exception_message_in_summary(self):
        wr = _draft_created_result()
        with patch(_MOD_SAVE, side_effect=OSError("no space left")):
            result = save_agent_draft_result(wr)
        assert "no space left" in result.summary


# ---------------------------------------------------------------------------
# Soft-unsupported date fields produce a warning
# ---------------------------------------------------------------------------


class TestSoftUnsupportedWarning:
    def test_start_date_warning_in_saved(self):
        wr = _draft_created_result(soft_unsupported=["start_date"])
        with patch(_MOD_SAVE, return_value=Path("/tmp/d.json")):
            result = save_agent_draft_result(wr)
        assert result.status == "saved"
        assert any("start_date" in w for w in result.warnings)

    def test_end_date_warning_in_saved(self):
        wr = _draft_created_result(soft_unsupported=["end_date"])
        with patch(_MOD_SAVE, return_value=Path("/tmp/d.json")):
            result = save_agent_draft_result(wr)
        assert result.status == "saved"
        assert any("end_date" in w for w in result.warnings)

    def test_soft_unsupported_does_not_cause_not_saved(self):
        wr = _draft_created_result(soft_unsupported=["start_date", "end_date"])
        with patch(_MOD_SAVE, return_value=Path("/tmp/d.json")):
            result = save_agent_draft_result(wr)
        assert result.status == "saved"

    def test_no_warning_when_no_soft_unsupported(self):
        wr = _draft_created_result(soft_unsupported=[])
        with patch(_MOD_SAVE, return_value=Path("/tmp/d.json")):
            result = save_agent_draft_result(wr)
        assert result.warnings == []


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------


class TestPackageInit:
    def test_imported_from_package(self):
        from src.agent.workflow import SavedAgentDraftResult, save_agent_draft_result
        assert callable(save_agent_draft_result)
        assert SavedAgentDraftResult is not None
