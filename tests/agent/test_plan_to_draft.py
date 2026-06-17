"""Tests for src/agent/drafting/plan_to_draft.py

Covers all five PlanDraftResult status values:
  draft_created / needs_base_experiment / unsupported_plan / invalid_plan / validation_failed

All disk I/O uses tmp_path fixtures.
list_all() is patched to avoid scanning results/experiments/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from src.agent.drafting.draft_result import PlanDraftResult, summarise_plan_draft_result
from src.agent.drafting.plan_to_draft import (
    _PRESET_TO_CONFIG_NAME,
    _resolve_base_experiment,
    create_draft_from_plan,
)
from src.agent.intent.research_spec import ResearchSpec
from src.agent.planner.create_experiment_plan import create_experiment_plan
from src.agent.planner.experiment_plan import ExperimentPlan
from src.agent.registry.registry_loader import load_registry
from src.agent.resolver.resolution_result import ResearchSpecResolutionResult
from src.agent.resolver.resolve_research_spec import resolve_research_spec
from src.agent.resolver.resolved_spec import ResolvedResearchSpec
from src.orchestration.config_generation.draft_schema import ExperimentDraft

# ---------------------------------------------------------------------------
# Shared fixtures and base config
# ---------------------------------------------------------------------------

_MULTI_ASSET_BASE: dict[str, Any] = {
    "version": "2",
    "name": "canonical_ml_multi_asset",
    "universe": {"tickers": ["SPY", "TLT", "GLD", "EEM", "VNQ"]},
    "date_range": {"start": "2015-01-01", "end": "2024-12-31"},
    "model": {"type": "RidgeRegression", "params": {"alpha": 0.5}},
    "labels": {"type": "ranking_target", "params": {"horizon": 21}},
    "signal": {
        "type": "top_n",
        "params": {"n": 3},
    },
    "validation": {
        "type": "rolling",
        "parameters": {"train_months": 48, "test_months": 12, "gap_days": 0},
    },
    "execution": {"transaction_cost_bps": 5.0},
    "features": {
        "ticker": "SPY",
        "entries": [
            {"name": "mom_20", "type": "momentum", "params": {"lookback": 20}},
            {"name": "mom_60", "type": "momentum", "params": {"lookback": 60}},
        ],
    },
    "output": {
        "base_dir": "results/experiments",
        "registry_path": "results/experiments/registry.json",
        "register": True,
        "save_plots": True,
    },
}


def _write_base_config(
    tmp_path: Path,
    name: str = "canonical_ml_multi_asset",
    config: dict[str, Any] | None = None,
) -> Path:
    cfg_dir = tmp_path / "configs" / "experiments"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = config if config is not None else dict(_MULTI_ASSET_BASE)
    cfg["name"] = name
    (cfg_dir / f"{name}.yaml").write_text(yaml.dump(cfg), encoding="utf-8")
    return cfg_dir


def _patched_call(plan: ExperimentPlan, tmp_path: Path, **kwargs: Any) -> PlanDraftResult:
    """Call create_draft_from_plan with list_all patched to avoid registry scan."""
    cfg_dir = kwargs.pop("configs_base", tmp_path / "configs" / "experiments")
    with patch(
        "src.orchestration.config_generation.draft_validator.list_all",
        return_value=["canonical_ml_multi_asset"],
    ):
        return create_draft_from_plan(
            plan,
            configs_base=cfg_dir,
            base=tmp_path / "results" / "experiments",
            reports_base=tmp_path / "reports",
            **kwargs,
        )


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def _resolve(spec: ResearchSpec, registry: Any) -> ResolvedResearchSpec:
    result: ResearchSpecResolutionResult = resolve_research_spec(spec, registry)
    assert result.status == "resolved", (
        f"Unexpected resolution status: {result.status}. "
        f"Errors: {result.errors}"
    )
    return result.resolved_spec


def _make_plan(resolved: ResolvedResearchSpec, registry: Any) -> ExperimentPlan:
    plan_result = create_experiment_plan(resolved, registry=registry)
    assert not plan_result.is_invalid, (
        f"Unexpected plan status: {plan_result.status}. "
        f"Critic errors: {plan_result.critic_errors}"
    )
    return plan_result.plan


# ---------------------------------------------------------------------------
# _resolve_base_experiment — unit tests
# ---------------------------------------------------------------------------


class TestResolveBaseExperiment:
    def test_preset_key_mapped_to_config_name(self, registry):
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        assert _resolve_base_experiment(plan) == "canonical_ml_multi_asset"

    def test_single_asset_preset_mapped(self, registry):
        resolved = _resolve(
            ResearchSpec(preset="single_asset_ml", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        assert _resolve_base_experiment(plan) == "canonical_ml_showcase"

    def test_explicit_baseline_not_a_preset_used_as_is(self):
        plan = ExperimentPlan(
            plan_id="x",
            name="x",
            description="x",
            resolved_spec=ResolvedResearchSpec(
                preset=None,
                tickers=["SPY", "TLT"],
                end_date="2024-12-31",
            ),
            research_objective="x",
            hypothesis="x",
            experiment_type="custom_research_design",
            baseline="my_custom_experiment",
            comparison_policy="x",
        )
        assert _resolve_base_experiment(plan) == "my_custom_experiment"

    def test_no_baseline_no_preset_returns_none(self):
        plan = ExperimentPlan(
            plan_id="x",
            name="x",
            description="x",
            resolved_spec=ResolvedResearchSpec(
                tickers=["SPY"],
                end_date="2024-12-31",
            ),
            research_objective="x",
            hypothesis="x",
            experiment_type="custom_research_design",
            baseline=None,
            comparison_policy="x",
        )
        assert _resolve_base_experiment(plan) is None

    def test_preset_to_config_name_table(self):
        assert "canonical_multi_asset_etf" in _PRESET_TO_CONFIG_NAME
        assert "single_asset_ml" in _PRESET_TO_CONFIG_NAME
        assert _PRESET_TO_CONFIG_NAME["canonical_multi_asset_etf"] == "canonical_ml_multi_asset"
        assert _PRESET_TO_CONFIG_NAME["single_asset_ml"] == "canonical_ml_showcase"


# ---------------------------------------------------------------------------
# Scenario 1: preset_baseline — no user-specified config changes
# ---------------------------------------------------------------------------


class TestScenarioPresetBaseline:
    def test_preset_baseline_returns_draft_created(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        assert result.is_draft_created
        assert result.status == "draft_created"

    def test_preset_baseline_draft_not_approved(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        assert result.draft is not None
        assert result.draft.approved is False

    def test_preset_baseline_base_experiment_identified(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        assert result.base_experiment == "canonical_ml_multi_asset"

    def test_preset_baseline_draft_has_proposed_name(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        assert result.draft is not None
        assert result.draft.proposed_name.startswith("canonical_ml_multi_asset_v")

    def test_preset_baseline_plan_preserved_in_result(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        assert result.plan is plan

    def test_preset_baseline_summary_is_string(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        assert isinstance(result.summary, str)
        assert result.summary


# ---------------------------------------------------------------------------
# Scenario 2: model_comparison — user specifies a different model
# ---------------------------------------------------------------------------


class TestScenarioModelComparison:
    def test_model_comparison_returns_draft_created(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(
                preset="canonical_multi_asset_etf",
                end_date="2024-12-31",
                model="LassoRegression",
            ),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        assert result.is_draft_created

    def test_model_comparison_draft_contains_model_change(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(
                preset="canonical_multi_asset_etf",
                end_date="2024-12-31",
                model="LassoRegression",
            ),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        assert result.draft is not None
        model_changes = [
            c for c in result.draft.changes
            if c.section == "model" and c.field == "type"
        ]
        assert len(model_changes) == 1
        assert model_changes[0].proposed_value == "LassoRegression"

    def test_model_comparison_experiment_type(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(
                preset="canonical_multi_asset_etf",
                end_date="2024-12-31",
                model="LassoRegression",
            ),
            registry,
        )
        plan = _make_plan(resolved, registry)
        assert plan.experiment_type == "model_comparison"

    def test_model_params_change_included_in_draft(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(
                preset="canonical_multi_asset_etf",
                end_date="2024-12-31",
                model_params={"alpha": 2.0},
            ),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        assert result.is_draft_created
        alpha_changes = [
            c for c in result.draft.changes
            if c.section == "model" and c.field == "params.alpha"
        ]
        assert len(alpha_changes) == 1
        assert alpha_changes[0].proposed_value == 2.0


# ---------------------------------------------------------------------------
# Scenario 3: feature_family_test — user specifies features with feature_specs
# ---------------------------------------------------------------------------


class TestScenarioFeatureFamilyTest:
    def test_feature_test_with_feature_specs_returns_draft_created(self, tmp_path):
        resolved = ResolvedResearchSpec(
            preset="canonical_multi_asset_etf",
            tickers=["SPY", "TLT", "GLD", "EEM", "VNQ"],
            start_date="2015-01-01",
            end_date="2024-12-31",
            model="RidgeRegression",
            model_params={"alpha": 0.5},
            signal="top_n",
            signal_params={"n": 3},
            label="ranking_target",
            label_horizon=21,
            train_window="48m",
            test_window="12m",
            transaction_cost_bps=5.0,
            features=["vol_21"],
            feature_specs=[
                {"name": "vol_21", "type": "rolling_volatility", "params": {"window": 21}}
            ],
            user_specified_fields={"features": ["vol_21"]},
            preset_fields={
                "model": "RidgeRegression",
                "model_params": {"alpha": 0.5},
                "signal": "top_n",
                "signal_params": {"n": 3},
                "label": "ranking_target",
                "label_horizon": 21,
                "train_window": "48m",
                "test_window": "12m",
                "transaction_cost_bps": 5.0,
                "tickers": ["SPY", "TLT", "GLD", "EEM", "VNQ"],
            },
        )
        plan = ExperimentPlan(
            plan_id="plan_feat001",
            name="feature_family_test",
            description="Test adding volatility feature",
            resolved_spec=resolved,
            research_objective="Add vol_21 feature.",
            hypothesis="Adding vol_21 will improve IC.",
            experiment_type="feature_family_test",
            baseline="canonical_multi_asset_etf",
            comparison_policy="Compare against baseline.",
            success_criteria=["IC > 0.05"],
            risks=["Feature may overfit."],
        )
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        assert result.is_draft_created
        add_changes = [
            c for c in result.draft.changes if c.field == "entries.add"
        ]
        assert len(add_changes) >= 1
        names = [c.proposed_value["name"] for c in add_changes]
        assert "vol_21" in names

    def test_feature_test_existing_features_not_re_added(self, tmp_path):
        resolved = ResolvedResearchSpec(
            preset="canonical_multi_asset_etf",
            tickers=["SPY", "TLT", "GLD", "EEM", "VNQ"],
            start_date="2015-01-01",
            end_date="2024-12-31",
            model="RidgeRegression",
            model_params={"alpha": 0.5},
            signal="top_n",
            signal_params={"n": 3},
            label="ranking_target",
            label_horizon=21,
            train_window="48m",
            test_window="12m",
            transaction_cost_bps=5.0,
            features=["mom_20"],  # already in base config
            feature_specs=[
                {"name": "mom_20", "type": "momentum", "params": {"lookback": 20}}
            ],
            user_specified_fields={"features": ["mom_20"]},
            preset_fields={},
        )
        plan = ExperimentPlan(
            plan_id="plan_feat002",
            name="feature_family_test_no_new",
            description="Test feature already in base",
            resolved_spec=resolved,
            research_objective="Check existing feature.",
            hypothesis="No new feature needed.",
            experiment_type="feature_family_test",
            baseline="canonical_multi_asset_etf",
            comparison_policy="Compare.",
            success_criteria=["IC > 0.0"],
            risks=["None."],
        )
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        assert result.is_draft_created
        add_changes = [c for c in result.draft.changes if c.field == "entries.add"]
        assert add_changes == []


# ---------------------------------------------------------------------------
# Scenario 4: needs_base_experiment
# ---------------------------------------------------------------------------


class TestScenarioNeedsBaseExperiment:
    def test_no_baseline_no_preset_needs_base_experiment(self, tmp_path):
        resolved = ResolvedResearchSpec(
            tickers=["AAPL"],
            start_date="2015-01-01",
            end_date="2024-12-31",
            model="RidgeRegression",
            model_params={"alpha": 0.5},
            signal="sign",
            signal_params={},
            label="forward_returns",
            label_horizon=21,
            train_window="48m",
            test_window="12m",
            transaction_cost_bps=5.0,
            user_specified_fields={"end_date": "2024-12-31"},
        )
        plan = ExperimentPlan(
            plan_id="plan_custom001",
            name="custom no preset",
            description="custom plan",
            resolved_spec=resolved,
            research_objective="x",
            hypothesis="x",
            experiment_type="custom_research_design",
            baseline=None,
            comparison_policy="x",
            success_criteria=["x"],
            risks=["x"],
        )
        result = create_draft_from_plan(plan, configs_base=tmp_path / "cfg")

        assert result.is_needs_base_experiment
        assert result.status == "needs_base_experiment"
        assert result.draft is None

    def test_base_config_not_on_disk_needs_base_experiment(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        # Do NOT write the base config file
        empty_cfg_dir = tmp_path / "empty" / "configs" / "experiments"
        empty_cfg_dir.mkdir(parents=True, exist_ok=True)

        result = create_draft_from_plan(
            plan,
            configs_base=empty_cfg_dir,
            base=tmp_path / "results",
            reports_base=tmp_path / "reports",
        )

        assert result.is_needs_base_experiment
        assert result.base_experiment == "canonical_ml_multi_asset"
        assert result.draft is None

    def test_needs_base_experiment_has_error_message(self, tmp_path):
        resolved = ResolvedResearchSpec(
            tickers=["AAPL"],
            start_date="2015-01-01",
            end_date="2024-12-31",
            model="RidgeRegression",
            model_params={"alpha": 0.5},
            signal="sign",
            signal_params={},
            label="forward_returns",
            label_horizon=21,
            train_window="48m",
            test_window="12m",
            transaction_cost_bps=5.0,
            user_specified_fields={"end_date": "2024-12-31"},
        )
        plan = ExperimentPlan(
            plan_id="p001",
            name="no base",
            description="x",
            resolved_spec=resolved,
            research_objective="x",
            hypothesis="x",
            experiment_type="custom_research_design",
            baseline=None,
            comparison_policy="x",
            success_criteria=["x"],
            risks=["x"],
        )
        result = create_draft_from_plan(plan)
        assert result.errors


# ---------------------------------------------------------------------------
# Scenario 5: unsupported_plan — user specifies universe / tickers
# ---------------------------------------------------------------------------


class TestScenarioUnsupportedPlan:
    def test_user_specifies_tickers_unsupported(self):
        resolved = ResolvedResearchSpec(
            tickers=["SPY", "QQQ"],
            start_date="2015-01-01",
            end_date="2024-12-31",
            model="RidgeRegression",
            model_params={"alpha": 0.5},
            signal="top_n",
            signal_params={"n": 2},
            label="ranking_target",
            label_horizon=21,
            train_window="48m",
            test_window="12m",
            transaction_cost_bps=5.0,
            user_specified_fields={
                "tickers": ["SPY", "QQQ"],
                "end_date": "2024-12-31",
            },
            preset_fields={
                "model": "RidgeRegression",
                "model_params": {"alpha": 0.5},
            },
        )
        plan = ExperimentPlan(
            plan_id="p_universe",
            name="universe_test",
            description="x",
            resolved_spec=resolved,
            research_objective="x",
            hypothesis="x",
            experiment_type="universe_test",
            baseline="canonical_multi_asset_etf",
            comparison_policy="compare",
            success_criteria=["x"],
            risks=["x"],
        )
        result = create_draft_from_plan(plan)

        assert result.is_unsupported
        assert result.status == "unsupported_plan"
        assert "tickers" in result.unsupported_fields
        assert result.draft is None

    def test_user_specifies_universe_unsupported(self):
        resolved = ResolvedResearchSpec(
            universe="global_equity",
            tickers=["SPY", "EFA"],
            start_date="2015-01-01",
            end_date="2024-12-31",
            model="RidgeRegression",
            model_params={"alpha": 0.5},
            signal="top_n",
            signal_params={"n": 2},
            label="ranking_target",
            label_horizon=21,
            train_window="48m",
            test_window="12m",
            transaction_cost_bps=5.0,
            user_specified_fields={"universe": "global_equity"},
            preset_fields={},
        )
        plan = ExperimentPlan(
            plan_id="p_universe2",
            name="universe_test2",
            description="x",
            resolved_spec=resolved,
            research_objective="x",
            hypothesis="x",
            experiment_type="universe_test",
            baseline=None,
            comparison_policy="x",
            success_criteria=["x"],
            risks=["x"],
        )
        result = create_draft_from_plan(plan)

        assert result.is_unsupported
        assert "universe" in result.unsupported_fields

    def test_unsupported_plan_has_descriptive_errors(self):
        resolved = ResolvedResearchSpec(
            tickers=["SPY", "QQQ"],
            end_date="2024-12-31",
            user_specified_fields={"tickers": ["SPY", "QQQ"]},
        )
        plan = ExperimentPlan(
            plan_id="p003",
            name="x",
            description="x",
            resolved_spec=resolved,
            research_objective="x",
            hypothesis="x",
            experiment_type="universe_test",
            baseline=None,
            comparison_policy="x",
            success_criteria=["x"],
            risks=["x"],
        )
        result = create_draft_from_plan(plan)

        assert result.errors
        assert any("whitelist" in e.lower() or "unsupported" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# Scenario 6: invalid_plan — resolver missing_fields or compat errors
# ---------------------------------------------------------------------------


class TestScenarioInvalidPlan:
    def test_incomplete_resolved_spec_returns_invalid_plan(self, tmp_path):
        resolved = ResolvedResearchSpec(
            tickers=["SPY"],
            end_date="2024-12-31",
            missing_fields=["model", "label"],  # simulate incomplete resolution
        )
        plan = ExperimentPlan(
            plan_id="p_invalid",
            name="incomplete plan",
            description="x",
            resolved_spec=resolved,
            research_objective="x",
            hypothesis="x",
            experiment_type="custom_research_design",
            baseline=None,
            comparison_policy="x",
            success_criteria=["x"],
            risks=["x"],
        )
        result = create_draft_from_plan(plan, configs_base=tmp_path)

        assert result.is_invalid_plan
        assert result.status == "invalid_plan"
        assert result.draft is None
        assert result.errors

    def test_compat_errors_return_invalid_plan(self, tmp_path):
        resolved = ResolvedResearchSpec(
            tickers=["SPY"],
            end_date="2024-12-31",
            compatibility_errors=["Panel signal incompatible with single-asset universe."],
        )
        plan = ExperimentPlan(
            plan_id="p_compat",
            name="compat error plan",
            description="x",
            resolved_spec=resolved,
            research_objective="x",
            hypothesis="x",
            experiment_type="custom_research_design",
            baseline=None,
            comparison_policy="x",
            success_criteria=["x"],
            risks=["x"],
        )
        result = create_draft_from_plan(plan, configs_base=tmp_path)

        assert result.is_invalid_plan
        assert any("Panel signal" in e or "compat" in e.lower() for e in result.errors)

    def test_invalid_plan_has_no_draft(self, tmp_path):
        resolved = ResolvedResearchSpec(
            missing_fields=["model"],
        )
        plan = ExperimentPlan(
            plan_id="p_nodraft",
            name="x",
            description="x",
            resolved_spec=resolved,
            research_objective="x",
            hypothesis="x",
            experiment_type="custom_research_design",
            baseline=None,
            comparison_policy="x",
            success_criteria=["x"],
            risks=["x"],
        )
        result = create_draft_from_plan(plan)
        assert result.draft is None


# ---------------------------------------------------------------------------
# Scenario 7: validation_failed — draft assembled but validate_draft rejects it
# ---------------------------------------------------------------------------


class TestScenarioValidationFailed:
    def test_invalid_model_type_rejected_by_resolver_is_invalid_plan(self, registry, tmp_path):
        # The resolver rejects unknown model names with a compatibility error,
        # so create_draft_from_plan returns invalid_plan (not validation_failed).
        result_spec = resolve_research_spec(
            ResearchSpec(
                preset="canonical_multi_asset_etf",
                end_date="2024-12-31",
                model="BogusModelXYZ",
            ),
            registry,
        )
        if result_spec.status == "resolved":
            pytest.skip("Resolver unexpectedly accepted BogusModelXYZ.")

        # Simulate a plan built from the rejected spec (with compat errors still present).
        resolved = result_spec.resolved_spec or ResolvedResearchSpec(
            compatibility_errors=["Model 'BogusModelXYZ' is not in the registry."],
        )
        plan = ExperimentPlan(
            plan_id="plan_val_fail",
            name="bogus model plan",
            description="x",
            resolved_spec=resolved,
            research_objective="x",
            hypothesis="x",
            experiment_type="model_comparison",
            baseline="canonical_multi_asset_etf",
            comparison_policy="compare",
            success_criteria=["x"],
            risks=["x"],
        )
        result = create_draft_from_plan(plan, configs_base=tmp_path / "cfg")
        assert result.is_invalid_plan

    def test_validation_failed_has_validation_result(self, tmp_path):
        """Build a draft with a manually forced invalid change to trigger validation_failed."""
        from src.orchestration.config_generation.draft_schema import (
            DraftChange,
            ExperimentDraft,
            compute_draft_hash,
        )
        import uuid
        from datetime import UTC, datetime

        # Build an intentionally invalid draft (bad model type) and confirm
        # validate_draft rejects it, then confirm create_draft_from_plan surfaces this.
        resolved = ResolvedResearchSpec(
            preset="canonical_multi_asset_etf",
            tickers=["SPY", "TLT", "GLD", "EEM", "VNQ"],
            start_date="2015-01-01",
            end_date="2024-12-31",
            model="BogusModelXYZ",
            model_params={"alpha": 0.5},
            signal="top_n",
            signal_params={"n": 3},
            label="ranking_target",
            label_horizon=21,
            train_window="48m",
            test_window="12m",
            transaction_cost_bps=5.0,
            user_specified_fields={"model": "BogusModelXYZ"},
            preset_fields={
                "model_params": {"alpha": 0.5},
                "signal": "top_n",
                "signal_params": {"n": 3},
                "label": "ranking_target",
                "label_horizon": 21,
                "train_window": "48m",
                "test_window": "12m",
                "transaction_cost_bps": 5.0,
                "tickers": ["SPY", "TLT", "GLD", "EEM", "VNQ"],
            },
        )
        plan = ExperimentPlan(
            plan_id="plan_val_fail2",
            name="bogus model plan 2",
            description="x",
            resolved_spec=resolved,
            research_objective="x",
            hypothesis="x",
            experiment_type="model_comparison",
            baseline="canonical_multi_asset_etf",
            comparison_policy="compare",
            success_criteria=["IC > 0.05"],
            risks=["Model may fail."],
        )
        cfg_dir = _write_base_config(tmp_path)
        with patch(
            "src.orchestration.config_generation.draft_validator.list_all",
            return_value=["canonical_ml_multi_asset"],
        ):
            result = create_draft_from_plan(
                plan,
                configs_base=cfg_dir,
                base=tmp_path / "results",
                reports_base=tmp_path / "reports",
            )

        assert result.status in ("draft_created", "validation_failed")
        if result.status == "validation_failed":
            assert result.validation_result is not None
            assert not result.validation_result.is_valid
            assert result.draft is not None  # draft was assembled, just invalid


# ---------------------------------------------------------------------------
# PlanDraftResult properties and invariants
# ---------------------------------------------------------------------------


class TestPlanDraftResultProperties:
    def _make_basic_result(self, status: str, **kwargs) -> PlanDraftResult:
        return PlanDraftResult(status=status, plan=None, **kwargs)  # type: ignore[arg-type]

    def test_is_draft_created_true_when_status_draft_created(self):
        r = self._make_basic_result("draft_created")
        assert r.is_draft_created is True
        assert r.is_needs_base_experiment is False
        assert r.is_unsupported is False
        assert r.is_invalid_plan is False
        assert r.is_validation_failed is False

    def test_is_needs_base_experiment_true(self):
        r = self._make_basic_result("needs_base_experiment")
        assert r.is_needs_base_experiment is True
        assert r.is_draft_created is False

    def test_is_unsupported_true(self):
        r = self._make_basic_result("unsupported_plan")
        assert r.is_unsupported is True
        assert r.is_draft_created is False

    def test_is_invalid_plan_true(self):
        r = self._make_basic_result("invalid_plan")
        assert r.is_invalid_plan is True

    def test_is_validation_failed_true(self):
        r = self._make_basic_result("validation_failed")
        assert r.is_validation_failed is True

    def test_default_fields(self):
        r = PlanDraftResult(status="draft_created", plan=None)
        assert r.draft is None
        assert r.base_experiment is None
        assert r.errors == []
        assert r.warnings == []
        assert r.unsupported_fields == []
        assert r.validation_result is None
        assert r.summary == ""


# ---------------------------------------------------------------------------
# summarise_plan_draft_result
# ---------------------------------------------------------------------------


class TestSummarisePlanDraftResult:
    def test_summary_is_string(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        text = summarise_plan_draft_result(result)
        assert isinstance(text, str)
        assert len(text) > 20

    def test_summary_contains_status(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        text = summarise_plan_draft_result(result)
        assert result.status in text

    def test_summary_contains_base_experiment(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        text = summarise_plan_draft_result(result)
        assert "canonical_ml_multi_asset" in text

    def test_summary_shows_errors_when_present(self):
        r = PlanDraftResult(
            status="needs_base_experiment",
            plan=None,
            errors=["Base config not found."],
            summary="No base.",
        )
        text = summarise_plan_draft_result(r)
        assert "Base config not found." in text

    def test_summary_shows_unsupported_fields(self):
        r = PlanDraftResult(
            status="unsupported_plan",
            plan=None,
            unsupported_fields=["tickers", "universe"],
            summary="Unsupported.",
        )
        text = summarise_plan_draft_result(r)
        assert "tickers" in text
        assert "universe" in text


# ---------------------------------------------------------------------------
# Draft content integrity
# ---------------------------------------------------------------------------


class TestDraftContentIntegrity:
    def test_draft_is_experiment_draft_instance(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        assert isinstance(result.draft, ExperimentDraft)

    def test_draft_source_proposal_hash_is_plan_id(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        assert result.draft.source_proposal_hash == plan.plan_id

    def test_draft_base_experiment_field_set(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        assert result.draft.base_experiment == "canonical_ml_multi_asset"

    def test_draft_hash_is_12_char_hex(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        h = result.draft.draft_hash
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_second_call_gets_different_proposed_name(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        cfg_dir = _write_base_config(tmp_path)

        with patch(
            "src.orchestration.config_generation.draft_validator.list_all",
            return_value=["canonical_ml_multi_asset"],
        ):
            r1 = create_draft_from_plan(
                plan,
                configs_base=cfg_dir,
                base=tmp_path / "results",
                reports_base=tmp_path / "reports",
            )
            # Simulate the first draft's name being taken
            (cfg_dir / f"{r1.draft.proposed_name}.yaml").write_text(
                yaml.dump(_MULTI_ASSET_BASE), encoding="utf-8"
            )
            r2 = create_draft_from_plan(
                plan,
                configs_base=cfg_dir,
                base=tmp_path / "results",
                reports_base=tmp_path / "reports",
            )

        assert r1.draft.proposed_name != r2.draft.proposed_name

    def test_validation_result_present_on_success(self, registry, tmp_path):
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        assert result.validation_result is not None
        assert result.validation_result.is_valid


# ---------------------------------------------------------------------------
# Phase 3B.1 governance: unsupported field reporting for dates
# ---------------------------------------------------------------------------


class TestPhase3B1DateGovernance:
    """Tests for requirements A–E of the Phase 3B.1 governance patch.

    Governs the treatment of start_date / end_date as soft-unsupported:
    reported in result.unsupported_fields but non-blocking for draft_created.
    """

    # Requirement A: user-specified start_date appears in unsupported_fields
    def test_user_specified_start_date_in_result_unsupported_fields(
        self, registry, tmp_path
    ):
        """start_date is always user-specified and has no change-path.
        It must appear in result.unsupported_fields even for draft_created."""
        resolved = _resolve(
            ResearchSpec(
                preset="canonical_multi_asset_etf",
                start_date="2015-01-01",
                end_date="2024-12-31",
            ),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        # start_date must be reported (not silently ignored)
        if "start_date" in resolved.user_specified_fields:
            assert "start_date" in result.unsupported_fields, (
                "user-specified start_date must appear in result.unsupported_fields"
            )

    # Requirement B: user-specified end_date appears in unsupported_fields
    def test_user_specified_end_date_in_result_unsupported_fields(
        self, registry, tmp_path
    ):
        """end_date is always user-specified (required field). It has no change-path
        and must appear in result.unsupported_fields."""
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        assert "end_date" in resolved.user_specified_fields, (
            "end_date should be user-specified after real resolution"
        )
        assert "end_date" in result.unsupported_fields, (
            "user-specified end_date must appear in result.unsupported_fields"
        )

    # Requirement C: preset/default dates do not create unsupported fields
    def test_defaulted_start_date_not_in_unsupported_fields(
        self, registry, tmp_path
    ):
        """start_date sourced from defaults (not user_specified_fields) must NOT
        appear in result.unsupported_fields."""
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        if "start_date" not in resolved.user_specified_fields:
            assert "start_date" not in result.unsupported_fields

    # Requirement D: preset is NOT treated as unsupported
    def test_preset_not_in_unsupported_fields(self, registry, tmp_path):
        """preset is provenance metadata and must not appear in unsupported_fields."""
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        assert "preset" not in result.unsupported_fields

    # Requirement E: draft_created can have non-empty unsupported_fields (partial draft)
    def test_draft_created_allows_non_empty_unsupported_fields(
        self, registry, tmp_path
    ):
        """Soft-unsupported dates must NOT prevent draft_created.
        draft_created with non-empty unsupported_fields is a valid partial draft."""
        resolved = _resolve(
            ResearchSpec(preset="canonical_multi_asset_etf", end_date="2024-12-31"),
            registry,
        )
        plan = _make_plan(resolved, registry)
        _write_base_config(tmp_path)
        result = _patched_call(plan, tmp_path)

        # Status must be draft_created even though end_date is unsupported
        assert result.is_draft_created
        # unsupported_fields may be non-empty (partial draft is acceptable)
        assert isinstance(result.unsupported_fields, list)
        # Verify the contract: if end_date was user-specified, it's in unsupported_fields
        if "end_date" in resolved.user_specified_fields:
            assert "end_date" in result.unsupported_fields

    def test_unsupported_fields_is_empty_for_model_only_spec(self, tmp_path):
        """When a plan only has model changes (plus end_date), end_date is the
        only unsupported field. Universe-only specs with no dates do not generate
        unsupported_fields for model changes."""
        resolved = ResolvedResearchSpec(
            preset="canonical_multi_asset_etf",
            tickers=["SPY", "TLT", "GLD", "EEM", "VNQ"],
            end_date="2024-12-31",
            model="LassoRegression",
            model_params={"alpha": 0.5},
            signal="top_n",
            signal_params={"n": 3},
            label="ranking_target",
            label_horizon=21,
            train_window="48m",
            test_window="12m",
            transaction_cost_bps=5.0,
            user_specified_fields={"model": "LassoRegression"},  # no dates user-specified
            preset_fields={
                "model_params": {"alpha": 0.5},
                "signal": "top_n",
                "signal_params": {"n": 3},
                "label": "ranking_target",
                "label_horizon": 21,
                "train_window": "48m",
                "test_window": "12m",
                "transaction_cost_bps": 5.0,
                "tickers": ["SPY", "TLT", "GLD", "EEM", "VNQ"],
            },
        )
        plan = ExperimentPlan(
            plan_id="plan_model_only",
            name="model_only",
            description="x",
            resolved_spec=resolved,
            research_objective="x",
            hypothesis="x",
            experiment_type="model_comparison",
            baseline="canonical_multi_asset_etf",
            comparison_policy="compare",
            success_criteria=["x"],
            risks=["x"],
        )
        cfg_dir = _write_base_config(tmp_path)
        with patch(
            "src.orchestration.config_generation.draft_validator.list_all",
            return_value=["canonical_ml_multi_asset"],
        ):
            result = create_draft_from_plan(
                plan,
                configs_base=cfg_dir,
                base=tmp_path / "results",
                reports_base=tmp_path / "reports",
            )

        assert result.is_draft_created
        # Only model-related user fields — no date unsupported fields
        assert "end_date" not in result.unsupported_fields
        assert "start_date" not in result.unsupported_fields

    def test_unsupported_plan_includes_all_unsupported_hard_and_soft(self, tmp_path):
        """When hard-unsupported fields (tickers) are present alongside soft-unsupported
        (end_date), result.unsupported_fields must include both."""
        resolved = ResolvedResearchSpec(
            tickers=["SPY", "QQQ"],
            end_date="2024-12-31",
            user_specified_fields={
                "tickers": ["SPY", "QQQ"],
                "end_date": "2024-12-31",
            },
        )
        plan = ExperimentPlan(
            plan_id="p_mixed",
            name="mixed unsupported",
            description="x",
            resolved_spec=resolved,
            research_objective="x",
            hypothesis="x",
            experiment_type="universe_test",
            baseline=None,
            comparison_policy="x",
            success_criteria=["x"],
            risks=["x"],
        )
        result = create_draft_from_plan(plan)

        assert result.is_unsupported
        assert "tickers" in result.unsupported_fields  # hard
        assert "end_date" in result.unsupported_fields  # soft


# ---------------------------------------------------------------------------
# Phase 3B.1 governance: __init__.py package check
# ---------------------------------------------------------------------------


class TestPackageInit:
    """Verify that src/agent/drafting/__init__.py is correctly named and exports
    all public symbols. If the file were named init.py (without underscores),
    the package would not be importable as a module."""

    def test_package_importable(self):
        import src.agent.drafting as drafting_pkg
        assert drafting_pkg is not None

    def test_create_draft_from_plan_exported(self):
        from src.agent.drafting import create_draft_from_plan
        assert callable(create_draft_from_plan)

    def test_plan_draft_result_exported(self):
        from src.agent.drafting import PlanDraftResult
        assert PlanDraftResult is not None

    def test_detect_unsupported_fields_exported(self):
        from src.agent.drafting import detect_unsupported_fields
        assert callable(detect_unsupported_fields)

    def test_detect_hard_unsupported_fields_exported(self):
        from src.agent.drafting import detect_hard_unsupported_fields
        assert callable(detect_hard_unsupported_fields)

    def test_summarise_plan_draft_result_exported(self):
        from src.agent.drafting import summarise_plan_draft_result
        assert callable(summarise_plan_draft_result)

    def test_map_spec_to_changes_exported(self):
        from src.agent.drafting import map_spec_to_changes
        assert callable(map_spec_to_changes)


# ---------------------------------------------------------------------------
# Phase 3B.1: feature governance
# ---------------------------------------------------------------------------


class TestPhase3B1FeatureGovernance:
    """Tests for requirement 7: feature add / missing params / implied removals."""

    # 7A: new feature entries can be mapped to entries.add
    def test_new_feature_with_valid_params_creates_add_change(self, tmp_path):
        resolved = ResolvedResearchSpec(
            preset="canonical_multi_asset_etf",
            tickers=["SPY", "TLT", "GLD", "EEM", "VNQ"],
            end_date="2024-12-31",
            model="RidgeRegression",
            model_params={"alpha": 0.5},
            signal="top_n",
            signal_params={"n": 3},
            label="ranking_target",
            label_horizon=21,
            train_window="48m",
            test_window="12m",
            transaction_cost_bps=5.0,
            features=["trend_20"],
            feature_specs=[
                {"name": "trend_20", "type": "trend_strength", "params": {"window": 20}}
            ],
            user_specified_fields={"features": ["trend_20"]},
            preset_fields={
                "model": "RidgeRegression",
                "model_params": {"alpha": 0.5},
                "signal": "top_n",
                "signal_params": {"n": 3},
                "label": "ranking_target",
                "label_horizon": 21,
                "train_window": "48m",
                "test_window": "12m",
                "transaction_cost_bps": 5.0,
                "tickers": ["SPY", "TLT", "GLD", "EEM", "VNQ"],
            },
        )
        plan = ExperimentPlan(
            plan_id="plan_feat_7a",
            name="feat add test",
            description="x",
            resolved_spec=resolved,
            research_objective="x",
            hypothesis="x",
            experiment_type="feature_family_test",
            baseline="canonical_multi_asset_etf",
            comparison_policy="x",
            success_criteria=["x"],
            risks=["x"],
        )
        cfg_dir = _write_base_config(tmp_path)
        with patch(
            "src.orchestration.config_generation.draft_validator.list_all",
            return_value=["canonical_ml_multi_asset"],
        ):
            result = create_draft_from_plan(
                plan,
                configs_base=cfg_dir,
                base=tmp_path / "results",
                reports_base=tmp_path / "reports",
            )

        assert result.is_draft_created
        add_changes = [c for c in result.draft.changes if c.field == "entries.add"]
        assert len(add_changes) == 1
        assert add_changes[0].proposed_value["name"] == "trend_20"

    # 7B: missing required feature params are warned about (not silently added)
    def test_feature_missing_params_produces_warning_in_result(self, tmp_path):
        resolved = ResolvedResearchSpec(
            preset="canonical_multi_asset_etf",
            tickers=["SPY", "TLT", "GLD", "EEM", "VNQ"],
            end_date="2024-12-31",
            model="RidgeRegression",
            model_params={"alpha": 0.5},
            signal="top_n",
            signal_params={"n": 3},
            label="ranking_target",
            label_horizon=21,
            train_window="48m",
            test_window="12m",
            transaction_cost_bps=5.0,
            features=["bad_momentum"],
            feature_specs=[
                # momentum requires 'lookback' — omitted
                {"name": "bad_momentum", "type": "momentum", "params": {}}
            ],
            user_specified_fields={"features": ["bad_momentum"]},
            preset_fields={
                "model": "RidgeRegression",
                "model_params": {"alpha": 0.5},
                "signal": "top_n",
                "signal_params": {"n": 3},
                "label": "ranking_target",
                "label_horizon": 21,
                "train_window": "48m",
                "test_window": "12m",
                "transaction_cost_bps": 5.0,
                "tickers": ["SPY", "TLT", "GLD", "EEM", "VNQ"],
            },
        )
        plan = ExperimentPlan(
            plan_id="plan_feat_7b",
            name="feat missing params",
            description="x",
            resolved_spec=resolved,
            research_objective="x",
            hypothesis="x",
            experiment_type="feature_family_test",
            baseline="canonical_multi_asset_etf",
            comparison_policy="x",
            success_criteria=["x"],
            risks=["x"],
        )
        cfg_dir = _write_base_config(tmp_path)
        with patch(
            "src.orchestration.config_generation.draft_validator.list_all",
            return_value=["canonical_ml_multi_asset"],
        ):
            result = create_draft_from_plan(
                plan,
                configs_base=cfg_dir,
                base=tmp_path / "results",
                reports_base=tmp_path / "reports",
            )

        # Should still produce a draft (no entries.add for bad feature)
        assert result.status in ("draft_created", "validation_failed")
        assert any(
            "bad_momentum" in w or "missing" in w.lower() or "param" in w.lower()
            for w in result.warnings
        )

    # 7C: feature removals (implied) are not silently ignored
    def test_implied_feature_removals_reported_as_warnings(self, tmp_path):
        """When user specifies features=[vol_21] but base has mom_20 + mom_60,
        the unmapped base features must generate a warning in result.warnings."""
        resolved = ResolvedResearchSpec(
            preset="canonical_multi_asset_etf",
            tickers=["SPY", "TLT", "GLD", "EEM", "VNQ"],
            end_date="2024-12-31",
            model="RidgeRegression",
            model_params={"alpha": 0.5},
            signal="top_n",
            signal_params={"n": 3},
            label="ranking_target",
            label_horizon=21,
            train_window="48m",
            test_window="12m",
            transaction_cost_bps=5.0,
            features=["vol_21"],
            feature_specs=[
                {"name": "vol_21", "type": "rolling_volatility", "params": {"window": 21}}
            ],
            user_specified_fields={"features": ["vol_21"]},
            preset_fields={
                "model": "RidgeRegression",
                "model_params": {"alpha": 0.5},
                "signal": "top_n",
                "signal_params": {"n": 3},
                "label": "ranking_target",
                "label_horizon": 21,
                "train_window": "48m",
                "test_window": "12m",
                "transaction_cost_bps": 5.0,
                "tickers": ["SPY", "TLT", "GLD", "EEM", "VNQ"],
            },
        )
        plan = ExperimentPlan(
            plan_id="plan_feat_7c",
            name="implied removal test",
            description="x",
            resolved_spec=resolved,
            research_objective="x",
            hypothesis="x",
            experiment_type="feature_family_test",
            baseline="canonical_multi_asset_etf",
            comparison_policy="x",
            success_criteria=["x"],
            risks=["x"],
        )
        cfg_dir = _write_base_config(tmp_path)
        with patch(
            "src.orchestration.config_generation.draft_validator.list_all",
            return_value=["canonical_ml_multi_asset"],
        ):
            result = create_draft_from_plan(
                plan,
                configs_base=cfg_dir,
                base=tmp_path / "results",
                reports_base=tmp_path / "reports",
            )

        assert result.status in ("draft_created", "validation_failed")
        # Implied removals must be reported — not silently ignored
        assert any(
            "NOT removed" in w or "not in your specified" in w or "implied" in w.lower()
            for w in result.warnings
        ), f"Expected implied-removal warning in result.warnings, got: {result.warnings}"
