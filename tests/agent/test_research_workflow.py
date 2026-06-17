"""Tests for src/agent/workflow/research_workflow.py

Tests cover:
  - run_resolve_spec: resolved / needs_user_input / invalid / error
  - run_plan_from_spec: planned / needs_review / invalid propagation
  - run_draft_from_spec: draft_created / unsupported_plan / validation_failed /
                         needs_user_input / invalid propagation
  - WorkflowResult.to_dict: serialisation completeness
  - Governance flags: approval_required, execution_performed, draft_approved
  - Soft-unsupported date fields reported but not blocking
  - Hard-unsupported fields block draft creation
  - save_draft note propagated through MCP tool wrapper (see test_agent_workflow_tools)

All tests are in-memory. No disk I/O, no quant engine, no LLM.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.agent.workflow.research_workflow import (
    run_draft_from_spec,
    run_plan_from_spec,
    run_resolve_spec,
)
from src.agent.workflow.workflow_result import WorkflowResult

# ---------------------------------------------------------------------------
# Canonical test specs
# ---------------------------------------------------------------------------

# The canonical preset requires end_date to fully resolve.
_CANONICAL_DICT = {"preset": "canonical_multi_asset_etf", "end_date": "2024-12-31"}
_CANONICAL_SPEC = '{"preset": "canonical_multi_asset_etf", "end_date": "2024-12-31"}'

_MODEL_OVERRIDE_DICT = {
    "preset": "canonical_multi_asset_etf",
    "model": "LassoRegression",
    "end_date": "2024-12-31",
}

_DATE_OVERRIDE_DICT = {
    "preset": "canonical_multi_asset_etf",
    "start_date": "2015-01-01",
    "end_date": "2024-12-31",
}

_HARD_UNSUPPORTED_DICT = {
    "universe": "some_custom_universe",
    "model": "LassoRegression",
}

_INCOMPLETE_DICT = {
    "preset": "custom_ticker_template",  # unknown preset → invalid
}

_BAD_JSON = "not-json-at-all"


# ---------------------------------------------------------------------------
# WorkflowResult serialisation
# ---------------------------------------------------------------------------


class TestWorkflowResultToDict:
    def test_to_dict_contains_all_required_keys(self):
        wr = WorkflowResult(status="resolved", summary="ok")
        d = wr.to_dict()
        required_keys = {
            "status", "summary", "resolution", "plan", "draft",
            "warnings", "errors", "missing_fields", "defaulted_fields",
            "assumptions", "requires_confirmation",
            "unsupported_fields", "hard_unsupported_fields",
            "approval_required", "execution_performed", "draft_approved",
            "mapped_changes", "validation_errors", "proposed_name", "base_experiment",
        }
        assert required_keys.issubset(d.keys()), (
            f"Missing keys: {required_keys - set(d.keys())}"
        )

    def test_governance_defaults(self):
        wr = WorkflowResult(status="draft_created", summary="ok")
        d = wr.to_dict()
        assert d["approval_required"] is True
        assert d["execution_performed"] is False
        assert d["draft_approved"] is False

    def test_status_propagated(self):
        for status in ("resolved", "planned", "draft_created", "needs_user_input", "invalid"):
            wr = WorkflowResult(status=status, summary="s")
            assert wr.to_dict()["status"] == status


# ---------------------------------------------------------------------------
# run_resolve_spec
# ---------------------------------------------------------------------------


class TestRunResolveSpec:
    def test_canonical_preset_resolves(self):
        result = run_resolve_spec(_CANONICAL_DICT)
        assert result.status == "resolved"
        assert result.resolution is not None
        assert result.resolution["status"] == "resolved"

    def test_resolved_result_has_provenance(self):
        result = run_resolve_spec(_CANONICAL_DICT)
        assert isinstance(result.assumptions, list)
        assert isinstance(result.defaulted_fields, dict)
        assert isinstance(result.warnings, list)

    def test_resolved_spec_in_resolution(self):
        result = run_resolve_spec(_CANONICAL_DICT)
        assert result.resolution["resolved_spec"] is not None
        rs = result.resolution["resolved_spec"]
        assert "model" in rs  # preset provides model

    def test_invalid_preset_returns_invalid(self):
        result = run_resolve_spec({"preset": "nonexistent_preset_xyz"})
        assert result.status == "invalid"
        assert result.errors

    def test_empty_spec_returns_needs_user_input_or_resolves_with_defaults(self):
        # An empty spec may resolve (all defaults) or need input depending on
        # registry default_policy. Either is valid; just not "error".
        result = run_resolve_spec({})
        assert result.status in ("resolved", "needs_user_input", "invalid")
        assert result.status != "error"

    def test_error_status_on_non_dict_input(self):
        result = run_resolve_spec("not a dict")  # type: ignore[arg-type]
        assert result.status == "error"

    def test_date_fields_in_user_specified(self):
        result = run_resolve_spec(_DATE_OVERRIDE_DICT)
        # Dates are user-specified; spec should resolve (canonical preset fills rest)
        assert result.status == "resolved"
        rs = result.resolution["resolved_spec"]
        assert rs["start_date"] == "2015-01-01"
        assert rs["end_date"] == "2024-12-31"

    def test_model_mode_recommended_and_strict_both_work(self):
        for mode in ("recommended", "strict", "demo"):
            result = run_resolve_spec(_CANONICAL_DICT, mode=mode)
            assert result.status in ("resolved", "needs_user_input", "invalid")

    def test_unknown_mode_falls_back_to_recommended(self):
        result = run_resolve_spec(_CANONICAL_DICT, mode="bogus_mode")
        assert result.status in ("resolved", "needs_user_input", "invalid")
        assert result.status != "error"


# ---------------------------------------------------------------------------
# run_plan_from_spec
# ---------------------------------------------------------------------------


class TestRunPlanFromSpec:
    def test_canonical_preset_produces_plan(self):
        result = run_plan_from_spec(_CANONICAL_DICT)
        assert result.status in ("planned", "needs_review")
        assert result.plan is not None

    def test_plan_contains_key_fields(self):
        result = run_plan_from_spec(_CANONICAL_DICT)
        plan = result.plan
        assert plan is not None
        for key in ("plan_id", "name", "hypothesis", "experiment_type",
                    "baseline", "research_objective", "success_criteria",
                    "required_diagnostics"):
            assert key in plan, f"plan missing key: {key}"

    def test_model_override_produces_model_comparison_plan(self):
        result = run_plan_from_spec(_MODEL_OVERRIDE_DICT)
        assert result.status in ("planned", "needs_review")
        assert result.plan is not None
        assert result.plan["experiment_type"] == "model_comparison"

    def test_invalid_preset_propagates_invalid(self):
        result = run_plan_from_spec({"preset": "no_such_preset"})
        assert result.status == "invalid"
        assert result.plan is None or result.plan == {}

    def test_needs_user_input_propagates(self):
        # A spec that fails to resolve completely shouldn't produce a plan
        result = run_plan_from_spec({"preset": "no_such_preset"})
        assert result.status in ("needs_user_input", "invalid")

    def test_plan_carries_resolution_data(self):
        result = run_plan_from_spec(_CANONICAL_DICT)
        assert result.resolution is not None
        assert "resolved_spec" in result.resolution

    def test_warnings_are_accumulated(self):
        result = run_plan_from_spec(_CANONICAL_DICT)
        assert isinstance(result.warnings, list)

    def test_no_execution_performed(self):
        result = run_plan_from_spec(_CANONICAL_DICT)
        assert result.execution_performed is False

    def test_date_fields_present_in_plan(self):
        result = run_plan_from_spec(_DATE_OVERRIDE_DICT)
        assert result.status in ("planned", "needs_review")
        # start_date/end_date are soft-unsupported for drafts but don't affect planning
        assert result.plan is not None


# ---------------------------------------------------------------------------
# run_draft_from_spec
# ---------------------------------------------------------------------------


class TestRunDraftFromSpec:
    def test_canonical_preset_creates_draft_or_needs_base(self):
        # Without a real configs/ directory, draft may reach needs_user_input
        # because the base config YAML may not exist in the test environment.
        result = run_draft_from_spec(_CANONICAL_DICT)
        assert result.status in (
            "draft_created", "needs_user_input", "validation_failed",
            "planned", "needs_review",
        )
        assert result.execution_performed is False
        assert result.draft_approved is False

    def test_hard_unsupported_fields_block_draft(self):
        # Use the preset's own universe explicitly → it's valid (resolver succeeds)
        # but user-specified → triggers unsupported_plan in create_draft_from_plan.
        spec = {
            "preset": "canonical_multi_asset_etf",
            "end_date": "2024-12-31",
            "universe": "canonical_multi_asset_etfs",  # valid but user-specified
        }
        result = run_draft_from_spec(spec)
        assert result.status == "unsupported_plan"
        assert result.hard_unsupported_fields
        assert "universe" in result.hard_unsupported_fields
        assert result.draft is None

    def test_tickers_hard_unsupported(self):
        # tickers must also resolve — use minimal list alongside preset
        spec = {
            "preset": "canonical_multi_asset_etf",
            "end_date": "2024-12-31",
            "tickers": ["SPY", "AGG"],
        }
        result = run_draft_from_spec(spec)
        assert result.status == "unsupported_plan"
        assert "tickers" in result.hard_unsupported_fields

    def test_frequency_hard_unsupported(self):
        # Use the preset's frequency explicitly
        spec = {
            "preset": "canonical_multi_asset_etf",
            "end_date": "2024-12-31",
            "frequency": "monthly",  # valid but user-specified
        }
        result = run_draft_from_spec(spec)
        assert result.status == "unsupported_plan"
        assert "frequency" in result.hard_unsupported_fields

    def test_soft_unsupported_dates_do_not_block(self):
        # start_date/end_date are soft-unsupported — they should appear in
        # unsupported_fields but should NOT cause status=unsupported_plan.
        result = run_draft_from_spec(_DATE_OVERRIDE_DICT)
        # Status can be draft_created, needs_user_input, validation_failed, etc.
        # but NOT unsupported_plan
        assert result.status != "unsupported_plan", (
            "start_date/end_date must NOT block draft creation (soft-unsupported)"
        )
        # If the spec progressed far enough for unsupported_fields to be populated:
        if result.status in ("draft_created", "validation_failed", "needs_user_input"):
            # These fields may or may not appear depending on resolver behaviour
            pass  # Governance requirement: they must not BLOCK

    def test_governance_flags_on_draft_created(self):
        # Patch create_draft_from_plan to return a successful draft
        from unittest.mock import MagicMock
        from src.agent.drafting.draft_result import PlanDraftResult

        mock_draft = MagicMock()
        mock_draft.draft_id = "d1"
        mock_draft.draft_hash = "abc"
        mock_draft.base_experiment = "canonical_ml_multi_asset"
        mock_draft.proposed_name = "canonical_ml_multi_asset_v2"
        mock_draft.changes = []
        mock_draft.generated_at = "2026-01-01T00:00:00+00:00"
        mock_draft.approved = False

        mock_val = MagicMock()
        mock_val.is_valid = True
        mock_val.errors = []

        mock_result = PlanDraftResult(
            status="draft_created",
            plan=None,
            draft=mock_draft,
            base_experiment="canonical_ml_multi_asset",
            unsupported_fields=[],
            warnings=[],
            validation_result=mock_val,
            summary="Draft 'canonical_ml_multi_asset_v2' created.",
        )

        with patch(
            "src.agent.workflow.research_workflow.create_draft_from_plan",
            return_value=mock_result,
        ):
            result = run_draft_from_spec(_CANONICAL_DICT)
            if result.status == "draft_created":
                assert result.approval_required is True
                assert result.execution_performed is False
                assert result.draft_approved is False

    def test_invalid_preset_propagates_to_draft_chain(self):
        result = run_draft_from_spec({"preset": "nonexistent_xyz"})
        assert result.status == "invalid"
        assert result.draft is None

    def test_error_on_bad_input_type(self):
        result = run_draft_from_spec(42)  # type: ignore[arg-type]
        assert result.status == "error"

    def test_resolution_and_plan_present_in_draft_result(self):
        result = run_draft_from_spec(_CANONICAL_DICT)
        # Resolution is always populated
        assert result.resolution is not None

    def test_base_experiment_override(self):
        # base_experiment is forwarded to create_draft_from_plan via plan.baseline
        from unittest.mock import MagicMock
        from src.agent.drafting.draft_result import PlanDraftResult

        captured_plan_baseline: list[Any] = []

        def _capture_draft(plan, **kwargs):
            captured_plan_baseline.append(plan.baseline)
            return PlanDraftResult(
                status="needs_base_experiment",
                plan=plan,
                errors=["no base config"],
                summary="no base",
            )

        with patch(
            "src.agent.workflow.research_workflow.create_draft_from_plan",
            side_effect=_capture_draft,
        ):
            run_draft_from_spec(
                _CANONICAL_DICT,
                base_experiment="my_custom_config",
            )
            if captured_plan_baseline:
                assert captured_plan_baseline[0] == "my_custom_config"


# ---------------------------------------------------------------------------
# Governance: hard vs soft unsupported fields
# ---------------------------------------------------------------------------


class TestUnsupportedFieldGovernance:
    _HARD_FIELDS = ["universe", "tickers", "frequency", "data_source", "asset_class"]
    _SOFT_FIELDS = ["start_date", "end_date"]

    # Map each hard field to a value that keeps the spec resolvable
    _HARD_FIELD_VALID_VALUES: dict[str, Any] = {
        "universe": "canonical_multi_asset_etfs",  # preset's own universe, user-specified
        "tickers": ["SPY", "AGG"],
        "frequency": "monthly",  # preset's own frequency, user-specified
        "data_source": "default",
        "asset_class": "multi_asset",
    }

    @pytest.mark.parametrize("hard_field", _HARD_FIELDS)
    def test_hard_field_blocks_draft(self, hard_field: str):
        value = self._HARD_FIELD_VALID_VALUES.get(hard_field, "some_value")
        spec = {
            "preset": "canonical_multi_asset_etf",
            "end_date": "2024-12-31",  # required to pass resolver
            hard_field: value,
        }
        result = run_draft_from_spec(spec)
        assert result.status == "unsupported_plan", (
            f"Hard-unsupported field '{hard_field}' must block draft creation "
            f"(got status={result.status!r})"
        )
        assert hard_field in result.hard_unsupported_fields

    def test_soft_start_date_does_not_block_draft(self):
        # start_date is soft-unsupported — does not block (end_date resolves the spec)
        spec = {"preset": "canonical_multi_asset_etf", "start_date": "2015-01-01", "end_date": "2024-12-31"}
        result = run_draft_from_spec(spec)
        assert result.status != "unsupported_plan", (
            "start_date must NOT block draft creation (soft-unsupported)"
        )

    def test_both_dates_do_not_block(self):
        spec = {
            "preset": "canonical_multi_asset_etf",
            "start_date": "2015-01-01",
            "end_date": "2024-12-31",
        }
        result = run_draft_from_spec(spec)
        assert result.status != "unsupported_plan"

    def test_mixed_hard_and_soft_still_blocks(self):
        spec = {
            "preset": "canonical_multi_asset_etf",
            "end_date": "2024-12-31",
            "universe": "canonical_multi_asset_etfs",  # valid, user-specified
            "start_date": "2015-01-01",
        }
        result = run_draft_from_spec(spec)
        assert result.status == "unsupported_plan"
        assert "universe" in result.hard_unsupported_fields


# ---------------------------------------------------------------------------
# WorkflowResult package-level import
# ---------------------------------------------------------------------------


class TestPackageInit:
    def test_package_exports(self):
        from src.agent.workflow import (
            WorkflowResult,
            run_draft_from_spec,
            run_plan_from_spec,
            run_resolve_spec,
        )
        assert callable(run_resolve_spec)
        assert callable(run_plan_from_spec)
        assert callable(run_draft_from_spec)
        assert WorkflowResult is not None
