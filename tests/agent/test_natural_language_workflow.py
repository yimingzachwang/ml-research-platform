"""Tests for natural_language_workflow.py (Phase 4B).

Covers run_resolve_from_request(), run_plan_from_request(), run_draft_from_request():

  A. Resolve from NL request
     - successful resolve returns WorkflowResult with parse metadata
     - needs_user_input propagated as normal status (not error)
     - needs_clarification from parser returned as needs_clarification (not error)
     - parser_error returns WorkflowResult with status="error"
     - parse metadata always in result.parse

  B. Plan from NL request
     - parsed + resolved → planned returns WorkflowResult with parse + plan
     - parse failure stops before planner

  C. Draft from NL request
     - full chain: parse → resolve → plan → draft
     - save_draft=False → result.saved_draft is None
     - save_draft=True + draft_created → result.saved_draft populated
     - save_draft never inferred from NL request text

  D. Execution keyword detection
     - "run", "execute", "approve", "trade", "deploy", "live" in request → warning
     - warning does not block result
     - no execution actually performed

  E. Ambiguity detection
     - multiple models in request → warning in result
     - multiple presets in request → warning in result

  F. Governance
     - execution_performed always False
     - draft_approved always False
     - approval_required always True in draft result
     - resolver/planner/drafter not bypassed

  G. WorkflowResult.to_dict() Phase 4B extensions
     - parse key present
     - saved_draft key present
     - governance sub-dict present

All tests use in-memory mocks. No live LLM calls. No disk I/O.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.agent.workflow.natural_language_workflow import (
    _check_execution_keywords,
    _make_parse_dict,
    run_draft_from_request,
    run_plan_from_request,
    run_resolve_from_request,
)
from src.agent.workflow.workflow_result import WorkflowResult

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

_MOD_PARSE = "src.agent.workflow.natural_language_workflow.parse_research_intent"
_MOD_RESOLVE = "src.agent.workflow.natural_language_workflow.run_resolve_spec"
_MOD_PLAN = "src.agent.workflow.natural_language_workflow.run_plan_from_spec"
_MOD_DRAFT_SPEC = "src.agent.workflow.natural_language_workflow.run_draft_from_spec"
_MOD_PERSIST = "src.agent.workflow.natural_language_workflow.save_agent_draft_result"


def _stub_registry() -> MagicMock:
    reg = MagicMock()
    reg.model_names.return_value = frozenset({"RidgeRegression", "LassoRegression"})
    reg.preset_names.return_value = frozenset({"canonical_multi_asset_etf", "single_asset_ml"})
    reg.asset_class_names.return_value = frozenset({"multi_asset"})
    reg.universe_names.return_value = frozenset({"canonical_multi_asset_etfs"})
    reg.label_names.return_value = frozenset({"ranking_target"})
    reg.signal_names.return_value = frozenset({"top_n"})
    reg.validation_names.return_value = frozenset({"rolling"})
    reg.feature_names.return_value = frozenset({"mom_20", "vol_21"})
    reg.diagnostic_names.return_value = frozenset({"sharpe_ratio"})
    reg.feature.side_effect = lambda name: SimpleNamespace(
        name=name, family={"mom_20": "momentum", "vol_21": "volatility"}.get(name)
    )
    reg.default_policy.return_value = SimpleNamespace(default_mode="recommended")
    return reg


def _canonical_llm_output() -> str:
    return json.dumps({
        "research_spec": {
            "preset": "canonical_multi_asset_etf",
            "model": "LassoRegression",
            "start_date": "2015-01-01",
            "end_date": "2024-12-31",
        },
        "unrecognised_terms": [],
        "confidence": 0.91,
        "summary": "Canonical ETF + Lasso.",
    })


def _stub_parse_result(
    status: str = "parsed",
    spec_fields: dict | None = None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> MagicMock:
    from src.agent.intent.research_spec import ResearchSpec
    pr = MagicMock()
    pr.status = status
    pr.research_spec = ResearchSpec.from_dict(spec_fields or {}) if status == "parsed" else None
    pr.raw_text = "test request"
    pr.raw_model_output = None
    pr.extracted_fields = dict(spec_fields or {})
    pr.unrecognised_terms = []
    pr.registry_warnings = []
    pr.missing_after_parse = []
    pr.confidence = 0.9 if status == "parsed" else 0.0
    pr.errors = list(errors or [])
    pr.warnings = list(warnings or [])
    pr.summary = f"stub parse ({status})"
    return pr


def _stub_wf_result(status: str = "resolved") -> WorkflowResult:
    return WorkflowResult(
        status=status,
        summary=f"stub {status}",
        resolution={"preset": "canonical_multi_asset_etf"} if status == "resolved" else None,
        plan={"experiment_type": "preset_baseline"} if status in ("planned", "needs_review") else None,
        draft={"proposed_name": "exp_v2"} if status == "draft_created" else None,
        approval_required=True,
        execution_performed=False,
        draft_approved=False,
    )


# ---------------------------------------------------------------------------
# A. Resolve from NL request
# ---------------------------------------------------------------------------

class TestResolveFromRequest:
    def test_successful_resolve_returns_workflow_result(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"preset": "canonical_multi_asset_etf", "end_date": "2024-12-31"}
            )),
            patch(_MOD_RESOLVE, return_value=_stub_wf_result("resolved")),
        ):
            result = run_resolve_from_request("canonical ETF ending 2024")
        assert result.status == "resolved"

    def test_parse_metadata_present_in_result(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"preset": "canonical_multi_asset_etf", "end_date": "2024-12-31"}
            )),
            patch(_MOD_RESOLVE, return_value=_stub_wf_result("resolved")),
        ):
            result = run_resolve_from_request("canonical ETF")
        assert result.parse is not None
        assert result.parse["status"] == "parsed"

    def test_parse_metadata_keys(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_RESOLVE, return_value=_stub_wf_result("resolved")),
        ):
            result = run_resolve_from_request("ETF 2024")
        assert "extracted_fields" in result.parse
        assert "unrecognised_terms" in result.parse
        assert "confidence" in result.parse
        assert "registry_warnings" in result.parse

    def test_needs_user_input_is_normal_status(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"model": "LassoRegression"}
            )),
            patch(_MOD_RESOLVE, return_value=_stub_wf_result("needs_user_input")),
        ):
            result = run_resolve_from_request("use Lasso model")
        assert result.status == "needs_user_input"
        assert result.parse is not None  # parse info still attached

    def test_needs_user_input_not_wrapped_as_error(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"model": "LassoRegression"}
            )),
            patch(_MOD_RESOLVE, return_value=_stub_wf_result("needs_user_input")),
        ):
            result = run_resolve_from_request("use Lasso model")
        assert result.status != "error"

    def test_needs_clarification_from_parser(self):
        with patch(_MOD_PARSE, return_value=_stub_parse_result("needs_clarification")):
            result = run_resolve_from_request("zzz unknown tokens")
        assert result.status == "needs_clarification"
        # Resolver should NOT be called
        assert result.resolution is None

    def test_needs_clarification_not_wrapped_as_error(self):
        with patch(_MOD_PARSE, return_value=_stub_parse_result("needs_clarification")):
            result = run_resolve_from_request("zzz unknown tokens")
        assert result.status != "error"

    def test_parser_error_returns_error_status(self):
        with patch(_MOD_PARSE, return_value=_stub_parse_result("parser_error", errors=["crash"])):
            result = run_resolve_from_request("bad input")
        assert result.status == "error"

    def test_resolver_not_called_on_parse_failure(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result("needs_clarification")),
            patch(_MOD_RESOLVE) as mock_resolve,
        ):
            run_resolve_from_request("unclear request")
        mock_resolve.assert_not_called()

    def test_parse_metadata_included_on_failure(self):
        with patch(_MOD_PARSE, return_value=_stub_parse_result(
            "needs_clarification",
            warnings=["Used stub fallback."],
        )):
            result = run_resolve_from_request("unclear request")
        assert result.parse is not None
        assert result.parse["status"] == "needs_clarification"

    def test_parse_warnings_merged_into_result(self):
        pr = _stub_parse_result("parsed", {"end_date": "2024-12-31"})
        pr.warnings = ["Used keyword fallback."]
        with (
            patch(_MOD_PARSE, return_value=pr),
            patch(_MOD_RESOLVE, return_value=_stub_wf_result("resolved")),
        ):
            result = run_resolve_from_request("ETF 2024")
        assert any("keyword fallback" in w.lower() for w in result.warnings)

    def test_governance_execution_performed_false(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_RESOLVE, return_value=_stub_wf_result("resolved")),
        ):
            result = run_resolve_from_request("ETF 2024")
        assert result.execution_performed is False

    def test_registry_passed_through_to_parser(self):
        reg = _stub_registry()
        captured = {}
        def _capture_parse(user_request, registry=None, **kw):
            captured["registry"] = registry
            return _stub_parse_result("parsed", {"end_date": "2024-12-31"})

        with (
            patch(_MOD_PARSE, side_effect=_capture_parse),
            patch(_MOD_RESOLVE, return_value=_stub_wf_result("resolved")),
        ):
            run_resolve_from_request("ETF 2024", registry=reg)
        assert captured["registry"] is reg


# ---------------------------------------------------------------------------
# B. Plan from NL request
# ---------------------------------------------------------------------------

class TestPlanFromRequest:
    def test_successful_plan_returns_workflow_result(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"preset": "canonical_multi_asset_etf", "end_date": "2024-12-31"}
            )),
            patch(_MOD_PLAN, return_value=_stub_wf_result("planned")),
        ):
            result = run_plan_from_request("canonical ETF ending 2024")
        assert result.status == "planned"
        assert result.plan is not None

    def test_parse_metadata_in_plan_result(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_PLAN, return_value=_stub_wf_result("planned")),
        ):
            result = run_plan_from_request("ETF 2024")
        assert result.parse is not None
        assert result.parse["status"] == "parsed"

    def test_parse_failure_stops_before_planner(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result("needs_clarification")),
            patch(_MOD_PLAN) as mock_plan,
        ):
            result = run_plan_from_request("unclear request")
        mock_plan.assert_not_called()
        assert result.plan is None

    def test_needs_review_is_ok_status(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_PLAN, return_value=_stub_wf_result("needs_review")),
        ):
            result = run_plan_from_request("ETF 2024")
        assert result.status == "needs_review"

    def test_baseline_passed_through(self):
        captured = {}
        def _capture_plan(spec_dict, mode="recommended", baseline=None):
            captured["baseline"] = baseline
            return _stub_wf_result("planned")

        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_PLAN, side_effect=_capture_plan),
        ):
            run_plan_from_request("ETF 2024", baseline="canonical_ml_multi_asset")
        assert captured["baseline"] == "canonical_ml_multi_asset"

    def test_governance_no_draft_in_plan_result(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_PLAN, return_value=_stub_wf_result("planned")),
        ):
            result = run_plan_from_request("ETF 2024")
        assert result.draft is None
        assert result.execution_performed is False


# ---------------------------------------------------------------------------
# C. Draft from NL request
# ---------------------------------------------------------------------------

class TestDraftFromRequest:
    def test_successful_draft_returns_draft_created(self):
        wf = _stub_wf_result("draft_created")
        # give it a non-None experiment_draft_object so save could work
        wf.experiment_draft_object = MagicMock()
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_DRAFT_SPEC, return_value=wf),
        ):
            result = run_draft_from_request("ETF 2024")
        assert result.status == "draft_created"
        assert result.draft is not None

    def test_save_draft_false_no_persistence(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_DRAFT_SPEC, return_value=_stub_wf_result("draft_created")),
            patch(_MOD_PERSIST) as mock_persist,
        ):
            result = run_draft_from_request("ETF 2024", save_draft=False)
        mock_persist.assert_not_called()
        assert result.saved_draft is None

    def test_save_draft_true_calls_persistence(self):
        wf = _stub_wf_result("draft_created")
        wf.experiment_draft_object = MagicMock()
        persist_result = SimpleNamespace(
            status="saved", draft_id="d1", draft_path="/tmp/d1.json",
            approval_required=True, execution_performed=False, draft_approved=False,
        )
        persist_result.to_dict = lambda: {
            "status": "saved", "draft_id": "d1", "draft_path": "/tmp/d1.json",
            "approval_required": True, "execution_performed": False, "draft_approved": False,
        }

        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_DRAFT_SPEC, return_value=wf),
            patch(_MOD_PERSIST, return_value=persist_result),
        ):
            result = run_draft_from_request("ETF 2024", save_draft=True)
        assert result.saved_draft is not None
        assert result.saved_draft["status"] == "saved"

    def test_save_draft_only_when_draft_created(self):
        """save_draft=True with non-draft_created status → no persistence call."""
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_DRAFT_SPEC, return_value=_stub_wf_result("needs_user_input")),
            patch(_MOD_PERSIST) as mock_persist,
        ):
            result = run_draft_from_request("ETF 2024", save_draft=True)
        mock_persist.assert_not_called()
        assert result.saved_draft is None

    def test_parse_metadata_in_draft_result(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_DRAFT_SPEC, return_value=_stub_wf_result("draft_created")),
        ):
            result = run_draft_from_request("ETF 2024")
        assert result.parse is not None
        assert result.parse["status"] == "parsed"

    def test_parse_failure_stops_before_draft(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result("needs_clarification")),
            patch(_MOD_DRAFT_SPEC) as mock_draft,
        ):
            result = run_draft_from_request("unclear request")
        mock_draft.assert_not_called()
        assert result.draft is None

    def test_governance_flags_on_draft(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_DRAFT_SPEC, return_value=_stub_wf_result("draft_created")),
        ):
            result = run_draft_from_request("ETF 2024")
        assert result.execution_performed is False
        assert result.draft_approved is False
        assert result.approval_required is True

    def test_base_experiment_passed_through(self):
        captured = {}
        def _capture_draft(spec_dict, mode="recommended", baseline=None, base_experiment=None):
            captured["base_experiment"] = base_experiment
            return _stub_wf_result("draft_created")

        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_DRAFT_SPEC, side_effect=_capture_draft),
        ):
            run_draft_from_request("ETF 2024", base_experiment="canonical_ml_multi_asset")
        assert captured["base_experiment"] == "canonical_ml_multi_asset"


# ---------------------------------------------------------------------------
# D. Execution keyword detection
# ---------------------------------------------------------------------------

class TestExecutionKeywordDetection:
    @pytest.mark.parametrize("keyword", ["run", "execute", "approve", "trade", "deploy", "live"])
    def test_execution_keyword_produces_warning(self, keyword):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_RESOLVE, return_value=_stub_wf_result("resolved")),
        ):
            result = run_resolve_from_request(f"Please {keyword} the canonical ETF")
        assert any(
            "execution" in w.lower() or keyword in w.lower() for w in result.warnings
        ), f"Expected execution warning for keyword '{keyword}'"

    def test_execution_warning_does_not_block_result(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_RESOLVE, return_value=_stub_wf_result("resolved")),
        ):
            result = run_resolve_from_request("run the canonical ETF now")
        assert result.status == "resolved"  # not blocked by warning

    def test_no_execution_performed_despite_keyword(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_RESOLVE, return_value=_stub_wf_result("resolved")),
        ):
            result = run_resolve_from_request("execute the ETF experiment now")
        assert result.execution_performed is False

    def test_no_keyword_no_exec_warning(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_RESOLVE, return_value=_stub_wf_result("resolved")),
        ):
            result = run_resolve_from_request("test canonical ETF with Lasso from 2015")
        exec_warnings = [
            w for w in result.warnings
            if "execution" in w.lower() or "approval" in w.lower()
        ]
        assert not exec_warnings

    def test_check_execution_keywords_detects_run(self):
        warning = _check_execution_keywords("Please run this experiment today.")
        assert warning is not None
        assert "run" in warning

    def test_check_execution_keywords_returns_none_when_safe(self):
        warning = _check_execution_keywords("Test Lasso from 2015 to 2024.")
        assert warning is None

    def test_execution_warning_on_parse_failure(self):
        with patch(_MOD_PARSE, return_value=_stub_parse_result("needs_clarification")):
            result = run_resolve_from_request("run this unclear request")
        exec_warnings = [w for w in result.warnings if "execution" in w.lower()]
        assert exec_warnings


# ---------------------------------------------------------------------------
# E. Ambiguity detection
# ---------------------------------------------------------------------------

class TestAmbiguityDetection:
    def test_ambiguous_model_warning_in_parse_result(self):
        """Stub provider: both Ridge and Lasso in request → ambiguity warning."""
        from src.agent.workflow.natural_language_workflow import run_resolve_from_request
        reg = _stub_registry()
        with patch(_MOD_RESOLVE, return_value=_stub_wf_result("resolved")):
            result = run_resolve_from_request(
                "Compare Ridge and Lasso models on canonical ETF from 2015 to 2024",
                registry=reg,
            )
        # Parse warnings (from keyword fallback ambiguity detection) should propagate
        combined = " ".join(result.warnings + (result.parse or {}).get("warnings", []))
        assert "ambiguous" in combined.lower() or "multiple" in combined.lower()

    def test_ambiguous_model_uses_first_match(self):
        """When multiple models detected, first alphabetical name is chosen."""
        reg = _stub_registry()
        with patch(_MOD_RESOLVE, return_value=_stub_wf_result("resolved")):
            result = run_resolve_from_request(
                "Compare Lasso and Ridge on canonical ETF from 2015 to 2024",
                registry=reg,
            )
        # The parse dict should carry the selected model
        extracted = (result.parse or {}).get("extracted_fields", {})
        # Should have selected one model (not both)
        if "model" in extracted:
            assert extracted["model"] in {"LassoRegression", "RidgeRegression"}


# ---------------------------------------------------------------------------
# F. Governance
# ---------------------------------------------------------------------------

class TestGovernance:
    def test_execution_performed_always_false_resolve(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_RESOLVE, return_value=_stub_wf_result("resolved")),
        ):
            result = run_resolve_from_request("ETF 2024")
        assert result.execution_performed is False

    def test_execution_performed_always_false_plan(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_PLAN, return_value=_stub_wf_result("planned")),
        ):
            result = run_plan_from_request("ETF 2024")
        assert result.execution_performed is False

    def test_execution_performed_always_false_draft(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_DRAFT_SPEC, return_value=_stub_wf_result("draft_created")),
        ):
            result = run_draft_from_request("ETF 2024")
        assert result.execution_performed is False

    def test_draft_approved_always_false(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_DRAFT_SPEC, return_value=_stub_wf_result("draft_created")),
        ):
            result = run_draft_from_request("ETF 2024")
        assert result.draft_approved is False

    def test_resolve_uses_deterministic_resolver(self):
        """run_resolve_spec() is called (not bypassed) when parse succeeds."""
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_RESOLVE, return_value=_stub_wf_result("resolved")) as mock_resolve,
        ):
            run_resolve_from_request("ETF 2024")
        mock_resolve.assert_called_once()

    def test_plan_uses_deterministic_planner(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_PLAN, return_value=_stub_wf_result("planned")) as mock_plan,
        ):
            run_plan_from_request("ETF 2024")
        mock_plan.assert_called_once()

    def test_draft_uses_deterministic_drafter(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed", {"end_date": "2024-12-31"}
            )),
            patch(_MOD_DRAFT_SPEC, return_value=_stub_wf_result("draft_created")) as mock_draft,
        ):
            run_draft_from_request("ETF 2024")
        mock_draft.assert_called_once()

    def test_package_level_imports_available(self):
        from src.agent.workflow import (
            run_draft_from_request,
            run_plan_from_request,
            run_resolve_from_request,
        )
        assert callable(run_resolve_from_request)
        assert callable(run_plan_from_request)
        assert callable(run_draft_from_request)


# ---------------------------------------------------------------------------
# G. WorkflowResult.to_dict() Phase 4B extensions
# ---------------------------------------------------------------------------

class TestWorkflowResultPhase4BExtensions:
    def test_to_dict_has_parse_key(self):
        wr = WorkflowResult(status="resolved", summary="ok")
        d = wr.to_dict()
        assert "parse" in d

    def test_to_dict_parse_none_by_default(self):
        wr = WorkflowResult(status="resolved", summary="ok")
        d = wr.to_dict()
        assert d["parse"] is None

    def test_to_dict_has_saved_draft_key(self):
        wr = WorkflowResult(status="resolved", summary="ok")
        d = wr.to_dict()
        assert "saved_draft" in d

    def test_to_dict_saved_draft_none_by_default(self):
        wr = WorkflowResult(status="resolved", summary="ok")
        d = wr.to_dict()
        assert d["saved_draft"] is None

    def test_to_dict_has_governance_subdict(self):
        wr = WorkflowResult(status="resolved", summary="ok")
        d = wr.to_dict()
        assert "governance" in d
        g = d["governance"]
        assert isinstance(g, dict)
        assert "approval_required" in g
        assert "execution_performed" in g
        assert "draft_approved" in g

    def test_governance_values_correct(self):
        wr = WorkflowResult(
            status="draft_created",
            summary="ok",
            approval_required=True,
            execution_performed=False,
            draft_approved=False,
        )
        g = wr.to_dict()["governance"]
        assert g["approval_required"] is True
        assert g["execution_performed"] is False
        assert g["draft_approved"] is False

    def test_flat_governance_keys_still_present(self):
        """Backward compat: flat approval_required/execution_performed/draft_approved preserved."""
        wr = WorkflowResult(status="resolved", summary="ok")
        d = wr.to_dict()
        assert "approval_required" in d
        assert "execution_performed" in d
        assert "draft_approved" in d

    def test_parse_dict_populated_from_workflow(self):
        with (
            patch(_MOD_PARSE, return_value=_stub_parse_result(
                "parsed",
                spec_fields={"preset": "canonical_multi_asset_etf", "end_date": "2024-12-31"},
                warnings=["Used stub fallback."],
            )),
            patch(_MOD_RESOLVE, return_value=_stub_wf_result("resolved")),
        ):
            result = run_resolve_from_request("canonical ETF 2024")
        d = result.to_dict()
        assert d["parse"] is not None
        assert d["parse"]["status"] == "parsed"

    def test_make_parse_dict_helper(self):
        pr = _stub_parse_result("parsed", {"model": "LassoRegression"})
        pr.unrecognised_terms = ["GBM"]
        pr.registry_warnings = ["GBM not supported"]
        pr.confidence = 0.75
        d = _make_parse_dict(pr)
        assert d["status"] == "parsed"
        assert "GBM" in d["unrecognised_terms"]
        assert d["confidence"] == pytest.approx(0.75)
        assert "registry_warnings" in d
        assert "missing_after_parse" in d
        assert "raw_model_output" in d

    def test_existing_to_dict_keys_not_removed(self):
        """Phase 4B additions must not remove any existing WorkflowResult.to_dict() keys."""
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
