"""Tests for Phase-3C agent workflow MCP tools.

Covers:
  list_research_agent_options
  resolve_research_spec (tool)
  create_experiment_plan_from_spec (tool)
  create_validated_draft_from_spec (tool)

All agent workflow function calls are mocked — no disk I/O, no real resolver,
no quant engine, no LLM.  These tests verify:
  * the MCP envelope contract (ok/stage/display/data/next_suggested_action)
  * governance flags (approval_required / execution_performed / draft_approved)
  * JSON parse error handling
  * unsupported field reporting
  * save_draft persistence bridge (Phase 3D)
  * registry listing completeness

MCP envelope contract:
  {"ok": bool, "stage": str, "display": str, "data": dict, "next_suggested_action": str}
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.mcp import zeto_server as zeto

_CONTRACT_KEYS = {"ok", "stage", "display", "data", "next_suggested_action"}
_MOD_WORKFLOW = "src.mcp.zeto_server._agent_workflow"
_MOD_REGISTRY = "src.mcp.zeto_server._load_registry"

_CANONICAL_SPEC = json.dumps({"preset": "canonical_multi_asset_etf", "end_date": "2024-12-31"})
_MODEL_SPEC = json.dumps({
    "preset": "canonical_multi_asset_etf",
    "end_date": "2024-12-31",
    "model": "LassoRegression",
})
_DATE_SPEC = json.dumps({
    "preset": "canonical_multi_asset_etf",
    "start_date": "2015-01-01",
    "end_date": "2024-12-31",
})
_HARD_UNSUPPORTED_SPEC = json.dumps({
    "preset": "canonical_multi_asset_etf",
    "end_date": "2024-12-31",
    "universe": "canonical_multi_asset_etfs",
})


# ---------------------------------------------------------------------------
# Shared stub builders
# ---------------------------------------------------------------------------

def _make_wf_result(
    status: str = "resolved",
    unsupported: list[str] | None = None,
    hard_unsupported: list[str] | None = None,
    **overrides,
) -> MagicMock:
    """Build a WorkflowResult-like MagicMock with to_dict()."""
    d = {
        "status": status,
        "summary": f"stub {status}",
        "resolution": {"status": status, "resolved_spec": {}, "missing_fields": []},
        "plan": {"plan_id": "p1", "name": "stub", "experiment_type": "preset_baseline"},
        "draft": {"draft_id": "d1", "proposed_name": "exp_v2", "changes": []},
        "warnings": [],
        "errors": [],
        "missing_fields": [],
        "defaulted_fields": {},
        "assumptions": ["used recommended defaults"],
        "requires_confirmation": False,
        "unsupported_fields": unsupported or [],
        "hard_unsupported_fields": hard_unsupported or [],
        "approval_required": True,
        "execution_performed": False,
        "draft_approved": False,
        "mapped_changes": [{"section": "model", "field": "type", "current_value": "RidgeRegression", "proposed_value": "LassoRegression", "rationale": "user specified"}],
        "validation_errors": [],
        "proposed_name": "canonical_ml_multi_asset_v2",
        "base_experiment": "canonical_ml_multi_asset",
    }
    d.update(overrides)

    m = MagicMock()
    m.status = status
    for k, v in d.items():
        setattr(m, k, v)
    m.to_dict.return_value = d
    return m


def _stub_registry() -> MagicMock:
    reg = MagicMock()
    reg.model_names.return_value = frozenset({"RidgeRegression", "LassoRegression"})
    reg.feature_names.return_value = frozenset({"mom_20", "vol_21"})
    reg.signal_names.return_value = frozenset({"top_n", "long_short"})
    reg.label_names.return_value = frozenset({"ranking_target"})
    reg.universe_names.return_value = frozenset({"canonical_multi_asset_etfs"})
    reg.asset_class_names.return_value = frozenset({"multi_asset", "equity_index"})
    reg.validation_names.return_value = frozenset({"rolling"})
    reg.weighting_scheme_names.return_value = frozenset({"equal_weight"})
    reg.prediction_normalization_names.return_value = frozenset({"none"})
    reg.diagnostic_names.return_value = frozenset({"sharpe_ratio", "split_metrics"})
    reg.preset_names.return_value = frozenset({"canonical_multi_asset_etf"})
    reg.changeable_paths.return_value = frozenset({"model.type", "model.params.alpha"})
    # Dict access for _models, _features, etc.
    reg._models = {
        "RidgeRegression": SimpleNamespace(
            name="RidgeRegression", display_name="Ridge", description="L2",
            supported=True, default_params={"alpha": 0.5}, tunable_params={"alpha": [0.1, 1.0]},
        )
    }
    reg._features = {
        "mom_20": SimpleNamespace(
            name="mom_20", display_name="Momentum 20", family="momentum",
            description="20-day momentum", required_params=frozenset({"lookback"}),
            default_params={"lookback": 20},
        )
    }
    reg._signals = {
        "top_n": SimpleNamespace(
            name="top_n", display_name="Top-N", description="long top-N",
            required_params=frozenset({"n"}), default_params={"n": 3},
            requires_multi_asset=True,
        )
    }
    reg._labels = {
        "ranking_target": SimpleNamespace(
            name="ranking_target", display_name="Ranking Target", description="cross-sec rank",
            required_params=frozenset({"horizon"}), default_params={"horizon": 21},
        )
    }
    reg._universes = {
        "canonical_multi_asset_etfs": SimpleNamespace(
            name="canonical_multi_asset_etfs", display_name="Multi-Asset ETFs",
            asset_class="multi_asset", frequency="monthly", tickers=("SPY", "AGG"),
            description="canonical universe",
        )
    }
    reg._asset_classes = {
        "multi_asset": SimpleNamespace(
            name="multi_asset", display_name="Multi-Asset", description="mixed",
        )
    }
    reg._validations = {
        "rolling": SimpleNamespace(
            name="rolling", display_name="Rolling WF", description="walk-forward",
            required_params=frozenset(), default_params={"train_months": 48},
            minimum_data_guidance="5 years",
        )
    }
    reg._weighting_schemes = {
        "equal_weight": SimpleNamespace(
            name="equal_weight", display_name="Equal Weight", description="1/N",
            required_params=frozenset(), default_params={},
        )
    }
    reg._prediction_normalizations = {
        "none": SimpleNamespace(name="none", display_name="None", description="raw"),
    }
    reg._diagnostics = {
        "sharpe_ratio": SimpleNamespace(
            name="sharpe_ratio", display_name="Sharpe Ratio", description="OOS SR",
            required_for_complete_artefact=True,
        )
    }
    reg._presets = {
        "canonical_multi_asset_etf": SimpleNamespace(
            name="canonical_multi_asset_etf", display_name="Canonical Multi-Asset",
            description="main preset", asset_class="multi_asset",
            universe="canonical_multi_asset_etfs", frequency="monthly",
            benchmark="SPY", notes="",
        )
    }
    reg._default_policy = SimpleNamespace(default_mode="recommended", field_policies={})
    reg.change_path_entries.return_value = (
        SimpleNamespace(section="model", field="type", dotted_path="model.type", description="model type"),
    )
    return reg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_contract(out: dict) -> None:
    assert _CONTRACT_KEYS.issubset(out.keys()), (
        f"Missing contract keys: {_CONTRACT_KEYS - set(out.keys())}"
    )
    assert isinstance(out["ok"], bool)
    assert isinstance(out["stage"], str) and out["stage"]
    assert isinstance(out["display"], str)
    assert isinstance(out["data"], dict)
    assert isinstance(out["next_suggested_action"], str)


# ---------------------------------------------------------------------------
# list_research_agent_options
# ---------------------------------------------------------------------------


class TestListResearchAgentOptions:
    def test_overview_contract(self):
        with patch(_MOD_REGISTRY, return_value=_stub_registry()):
            out = zeto._mcp_list_research_agent_options()
        _assert_contract(out)
        assert out["ok"] is True

    def test_overview_contains_categories(self):
        with patch(_MOD_REGISTRY, return_value=_stub_registry()):
            out = zeto._mcp_list_research_agent_options()
        cats = out["data"]["categories"]
        assert "models" in cats
        assert "presets" in cats
        assert "change_paths" in cats
        assert "RidgeRegression" in cats["models"]

    def test_overview_has_counts(self):
        with patch(_MOD_REGISTRY, return_value=_stub_registry()):
            out = zeto._mcp_list_research_agent_options()
        assert "counts" in out["data"]
        assert out["data"]["counts"]["models"] >= 1

    def test_field_models_returns_model_list(self):
        with patch(_MOD_REGISTRY, return_value=_stub_registry()):
            out = zeto._mcp_list_research_agent_options(field="models")
        _assert_contract(out)
        assert out["ok"] is True
        opts = out["data"]["options"]
        assert isinstance(opts, list)
        assert opts[0]["name"] == "RidgeRegression"

    def test_field_presets_returns_preset_list(self):
        with patch(_MOD_REGISTRY, return_value=_stub_registry()):
            out = zeto._mcp_list_research_agent_options(field="presets")
        _assert_contract(out)
        opts = out["data"]["options"]
        assert any(p["name"] == "canonical_multi_asset_etf" for p in opts)

    def test_field_change_paths_returns_dotted_paths(self):
        with patch(_MOD_REGISTRY, return_value=_stub_registry()):
            out = zeto._mcp_list_research_agent_options(field="change_paths")
        _assert_contract(out)
        opts = out["data"]["options"]
        assert any(p["dotted_path"] == "model.type" for p in opts)

    def test_field_universes_filtered_by_asset_class(self):
        with patch(_MOD_REGISTRY, return_value=_stub_registry()):
            out = zeto._mcp_list_research_agent_options(
                field="universes", asset_class="multi_asset"
            )
        _assert_contract(out)
        opts = out["data"]["options"]
        assert all(u["asset_class"] == "multi_asset" for u in opts)

    def test_field_portfolio_construction_nested(self):
        with patch(_MOD_REGISTRY, return_value=_stub_registry()):
            out = zeto._mcp_list_research_agent_options(field="portfolio_construction")
        _assert_contract(out)
        assert "weighting_schemes" in out["data"]["options"]
        assert "prediction_normalizations" in out["data"]["options"]

    def test_field_defaults(self):
        with patch(_MOD_REGISTRY, return_value=_stub_registry()):
            out = zeto._mcp_list_research_agent_options(field="defaults")
        _assert_contract(out)
        assert "default_mode" in out["data"]["options"]

    def test_invalid_field_returns_ok_false(self):
        with patch(_MOD_REGISTRY, return_value=_stub_registry()):
            out = zeto._mcp_list_research_agent_options(field="nonexistent_xyz")
        _assert_contract(out)
        assert out["ok"] is False
        assert "supported_fields" in out["data"]

    def test_registry_load_error_returns_ok_false(self):
        with patch(_MOD_REGISTRY, side_effect=FileNotFoundError("registry missing")):
            out = zeto._mcp_list_research_agent_options()
        _assert_contract(out)
        assert out["ok"] is False


# ---------------------------------------------------------------------------
# resolve_research_spec (MCP tool)
# ---------------------------------------------------------------------------


class TestMcpResolveResearchSpec:
    def test_resolved_contract(self):
        with patch(f"{_MOD_WORKFLOW}.run_resolve_spec", return_value=_make_wf_result("resolved")):
            out = zeto._mcp_resolve_research_spec(_CANONICAL_SPEC)
        _assert_contract(out)
        assert out["ok"] is True
        assert "resolved" in out["stage"]

    def test_needs_user_input_ok_false(self):
        r = _make_wf_result("needs_user_input", missing_fields=["model"])
        with patch(f"{_MOD_WORKFLOW}.run_resolve_spec", return_value=r):
            out = zeto._mcp_resolve_research_spec(_CANONICAL_SPEC)
        _assert_contract(out)
        assert out["ok"] is False

    def test_invalid_ok_false(self):
        r = _make_wf_result("invalid", errors=["incompatible: logistic + ranking_target"])
        with patch(f"{_MOD_WORKFLOW}.run_resolve_spec", return_value=r):
            out = zeto._mcp_resolve_research_spec(_CANONICAL_SPEC)
        _assert_contract(out)
        assert out["ok"] is False

    def test_bad_json_input_returns_error_envelope(self):
        out = zeto._mcp_resolve_research_spec("not-valid-json{{")
        _assert_contract(out)
        assert out["ok"] is False
        assert "parse_error" in out["data"]

    def test_result_data_in_payload(self):
        wfr = _make_wf_result("resolved")
        with patch(f"{_MOD_WORKFLOW}.run_resolve_spec", return_value=wfr):
            out = zeto._mcp_resolve_research_spec(_CANONICAL_SPEC)
        d = out["data"]
        assert "status" in d
        assert "resolution" in d

    def test_next_action_on_resolved(self):
        with patch(f"{_MOD_WORKFLOW}.run_resolve_spec", return_value=_make_wf_result("resolved")):
            out = zeto._mcp_resolve_research_spec(_CANONICAL_SPEC)
        assert out["next_suggested_action"] == "create_experiment_plan_from_spec"

    def test_next_action_on_failure(self):
        r = _make_wf_result("needs_user_input")
        with patch(f"{_MOD_WORKFLOW}.run_resolve_spec", return_value=r):
            out = zeto._mcp_resolve_research_spec(_CANONICAL_SPEC)
        assert out["next_suggested_action"] == "list_research_agent_options"


# ---------------------------------------------------------------------------
# create_experiment_plan_from_spec (MCP tool)
# ---------------------------------------------------------------------------


class TestMcpCreateExperimentPlanFromSpec:
    def test_planned_contract(self):
        with patch(f"{_MOD_WORKFLOW}.run_plan_from_spec", return_value=_make_wf_result("planned")):
            out = zeto._mcp_create_experiment_plan_from_spec(_CANONICAL_SPEC)
        _assert_contract(out)
        assert out["ok"] is True
        assert "planned" in out["stage"]

    def test_needs_review_ok_true(self):
        with patch(f"{_MOD_WORKFLOW}.run_plan_from_spec", return_value=_make_wf_result("needs_review")):
            out = zeto._mcp_create_experiment_plan_from_spec(_CANONICAL_SPEC)
        _assert_contract(out)
        assert out["ok"] is True

    def test_invalid_ok_false(self):
        r = _make_wf_result("invalid", errors=["spec is invalid"])
        with patch(f"{_MOD_WORKFLOW}.run_plan_from_spec", return_value=r):
            out = zeto._mcp_create_experiment_plan_from_spec(_CANONICAL_SPEC)
        _assert_contract(out)
        assert out["ok"] is False

    def test_bad_json_input(self):
        out = zeto._mcp_create_experiment_plan_from_spec("{bad json}")
        _assert_contract(out)
        assert out["ok"] is False

    def test_plan_data_present(self):
        with patch(f"{_MOD_WORKFLOW}.run_plan_from_spec", return_value=_make_wf_result("planned")):
            out = zeto._mcp_create_experiment_plan_from_spec(_CANONICAL_SPEC)
        assert "plan" in out["data"]

    def test_next_action_on_planned(self):
        with patch(f"{_MOD_WORKFLOW}.run_plan_from_spec", return_value=_make_wf_result("planned")):
            out = zeto._mcp_create_experiment_plan_from_spec(_CANONICAL_SPEC)
        assert out["next_suggested_action"] == "create_validated_draft_from_spec"

    def test_baseline_forwarded(self):
        captured = {}

        def _capture(spec_dict, mode, baseline):
            captured["baseline"] = baseline
            return _make_wf_result("planned")

        with patch(f"{_MOD_WORKFLOW}.run_plan_from_spec", side_effect=_capture):
            zeto._mcp_create_experiment_plan_from_spec(_CANONICAL_SPEC, baseline="my_baseline")
        assert captured.get("baseline") == "my_baseline"

    def test_no_execution(self):
        with patch(f"{_MOD_WORKFLOW}.run_plan_from_spec", return_value=_make_wf_result("planned")):
            out = zeto._mcp_create_experiment_plan_from_spec(_CANONICAL_SPEC)
        assert out["data"]["execution_performed"] is False


# ---------------------------------------------------------------------------
# create_validated_draft_from_spec (MCP tool)
# ---------------------------------------------------------------------------


class TestMcpCreateValidatedDraftFromSpec:
    def test_draft_created_contract(self):
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC)
        _assert_contract(out)
        assert out["ok"] is True
        assert "draft_created" in out["stage"]

    def test_governance_flags_in_data(self):
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC)
        d = out["data"]
        assert d["approval_required"] is True
        assert d["execution_performed"] is False
        assert d["draft_approved"] is False

    def test_hard_unsupported_ok_false(self):
        r = _make_wf_result(
            "unsupported_plan",
            hard_unsupported=["universe"],
            unsupported=["universe"],
            errors=["universe is hard-unsupported"],
        )
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=r):
            out = zeto._mcp_create_validated_draft_from_spec(_HARD_UNSUPPORTED_SPEC)
        _assert_contract(out)
        assert out["ok"] is False
        assert "unsupported_plan" in out["stage"]

    def test_soft_unsupported_dates_in_display(self):
        r = _make_wf_result("draft_created", unsupported=["start_date", "end_date"])
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=r):
            out = zeto._mcp_create_validated_draft_from_spec(_DATE_SPEC)
        _assert_contract(out)
        # Soft-unsupported does not block
        assert out["ok"] is True
        assert out["data"]["unsupported_fields"] == ["start_date", "end_date"]

    def test_validation_failed_ok_false(self):
        r = _make_wf_result("validation_failed", errors=["model type invalid"])
        r.status = "validation_failed"
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=r):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC)
        _assert_contract(out)
        assert out["ok"] is False

    def test_bad_json_input(self):
        out = zeto._mcp_create_validated_draft_from_spec("not_json!!")
        _assert_contract(out)
        assert out["ok"] is False
        assert "parse_error" in out["data"]

    def test_mapped_changes_in_data(self):
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")):
            out = zeto._mcp_create_validated_draft_from_spec(_MODEL_SPEC)
        assert "mapped_changes" in out["data"]
        assert isinstance(out["data"]["mapped_changes"], list)

    def test_save_draft_true_calls_persistence(self):
        _MOD_PERSIST = "src.mcp.zeto_server._save_agent_draft"
        mock_persist = MagicMock()
        mock_persist.status = "saved"
        mock_persist.draft_id = "d001"
        mock_persist.draft_path = "/tmp/draft_d001.json"
        mock_persist.summary = "Draft saved."
        mock_persist.to_dict.return_value = {
            "status": "saved", "draft_id": "d001",
            "draft_path": "/tmp/draft_d001.json",
            "approval_required": True, "execution_performed": False,
            "draft_approved": False, "warnings": [], "errors": [], "summary": "saved",
        }
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")), \
             patch(_MOD_PERSIST, return_value=mock_persist):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=True)
        _assert_contract(out)
        assert "saved_draft" in out["data"]
        assert out["data"]["saved_draft"]["status"] == "saved"

    def test_save_draft_false_no_saved_draft_key(self):
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC, save_draft=False)
        _assert_contract(out)
        assert "saved_draft" not in out["data"]
        assert "save_draft_note" not in out["data"]

    def test_next_action_on_draft_created(self):
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=_make_wf_result("draft_created")):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC)
        assert out["next_suggested_action"] == "approve_experiment_draft"

    def test_base_experiment_forwarded(self):
        captured = {}

        def _capture(spec_dict, mode, baseline, base_experiment, **kw):
            captured["base_experiment"] = base_experiment
            return _make_wf_result("draft_created")

        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", side_effect=_capture):
            zeto._mcp_create_validated_draft_from_spec(
                _CANONICAL_SPEC, base_experiment="canonical_ml_multi_asset"
            )
        assert captured.get("base_experiment") == "canonical_ml_multi_asset"

    def test_no_execution_on_any_status(self):
        for status in ("draft_created", "unsupported_plan", "validation_failed", "needs_user_input", "invalid"):
            r = _make_wf_result(status)
            with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=r):
                out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC)
            assert out["data"].get("execution_performed") is False, (
                f"execution_performed must be False for status={status}"
            )

    def test_draft_never_approved(self):
        r = _make_wf_result("draft_created")
        with patch(f"{_MOD_WORKFLOW}.run_draft_from_spec", return_value=r):
            out = zeto._mcp_create_validated_draft_from_spec(_CANONICAL_SPEC)
        assert out["data"]["draft_approved"] is False


# ---------------------------------------------------------------------------
# Overall governance: all new tools in registered tool set
# ---------------------------------------------------------------------------


class TestPhase3CToolRegistration:
    _EXPECTED_NEW_TOOLS = {
        "list_research_agent_options",
        "resolve_research_spec",
        "create_experiment_plan_from_spec",
        "create_validated_draft_from_spec",
    }

    def test_new_tools_are_registered(self):
        names = {t.name for t in zeto.mcp._tool_manager.list_tools()}
        assert self._EXPECTED_NEW_TOOLS.issubset(names), (
            f"Missing tools: {self._EXPECTED_NEW_TOOLS - names}"
        )

    def test_total_tool_count(self):
        names = {t.name for t in zeto.mcp._tool_manager.list_tools()}
        assert len(names) == 51  # 36 original + 4 Phase-3C + 1 Phase-4A + 3 Phase-4B + 2 Phase-5C + 2 Phase-5E + 1 Phase-5F + 1 Phase-5G.2 + 1 Workflow Trace get_workflow_trace

    def test_new_tools_have_descriptions(self):
        for t in zeto.mcp._tool_manager.list_tools():
            if t.name in self._EXPECTED_NEW_TOOLS:
                assert t.description and len(t.description.strip()) > 20, (
                    f"{t.name} has missing/short description"
                )
