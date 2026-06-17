"""Hardening tests for propose_research_plan.

Verifies: granular already-tested detection by keyword, correct field paths
from _VALID_CHANGE_PATHS, feature_replacement (not signal_change) for feature
substitution suggestions, and no side-effect governance violations.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import src.orchestration.api.research_api as api
from src.orchestration.analysis.intervention_classifier import INTERVENTION_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MCP_MOD = "src.orchestration.api.research_api"


def _write_experiment(
    exp_base: Path,
    name: str,
    *,
    sharpe: float = 0.5,
    mean_oos: float = -0.3,
) -> None:
    d = exp_base / name
    (d / "diagnostics").mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"experiment_name": name}))
    (d / "metrics.json").write_text(json.dumps({
        "sharpe_ratio": sharpe,
        "mean_oos_sharpe": mean_oos,
        "max_drawdown": -0.35,
        "max_drawdown_pct": -35.0,
    }))
    (d / "diagnostics" / "split_metrics.json").write_text(json.dumps({
        "summary": {
            "n_splits": 6,
            "mean_sharpe": mean_oos,
            "std_sharpe": 0.4,
            "hit_rate_positive_sharpe": 0.5,
            "mean_annualized_return": 0.03,
            "worst_max_drawdown": -0.35,
        },
        "splits": [{"sharpe_ratio": mean_oos + i * 0.05} for i in range(6)],
    }))


def _write_comparison_evidence(
    comp_base: Path,
    base_name: str,
    cand_name: str,
    *,
    tested_change: str = "",
    delta_sharpe: float | None = None,
    delta_oos: float | None = None,
    intervention_type: str | None = None,
    failure_modes_cand: list[str] | None = None,
) -> None:
    from src.orchestration.utils.filesystem import comparison_evidence_json_path
    path = comparison_evidence_json_path(base_name, cand_name, comp_base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "artefact_type": "comparison_evidence",
        "base_experiment_name": base_name,
        "candidate_experiment_name": cand_name,
        "tested_change": tested_change,
        "intervention_type": intervention_type,
        "metric_deltas": {
            "delta_sharpe": delta_sharpe,
            "delta_mean_oos_sharpe": delta_oos,
        },
        "failure_modes_base": ["poor_oos_consistency"],
        "failure_modes_candidate": failure_modes_cand or [],
        "conclusion": f"Comparison {base_name} -> {cand_name}.",
    }))


# ---------------------------------------------------------------------------
# 1. Alpha increase recognised as already_tested via keyword match
# ---------------------------------------------------------------------------


def test_alpha_increase_already_tested_via_keyword(tmp_path):
    """Regularisation step must be already_tested when evidence contains 'alpha'."""
    exp_base = tmp_path / "results" / "experiments"
    comp_base = tmp_path / "results" / "comparisons"
    _write_experiment(exp_base, "ml_v11", sharpe=0.5, mean_oos=-0.3)
    _write_comparison_evidence(
        comp_base, "ml_v11", "ml_v11_alpha",
        tested_change="model.params.alpha 0.5 -> 2.0",
        delta_sharpe=-0.02,
        delta_oos=0.05,
        failure_modes_cand=["poor_oos_consistency"],
    )

    result = api.propose_research_plan(
        research_goal="Improve OOS stability",
        experiment_prefix="ml",
        experiments_base=exp_base,
        comparisons_base=comp_base,
    )
    assert result["status"] == "ok"
    plan = result["plan"]
    reg_steps = [s for s in plan if s["intervention_type"] == "regularisation_change"]
    assert reg_steps, "regularisation_change suggestion should appear in plan"
    assert reg_steps[0]["status"] == "already_tested", (
        "alpha increase must be recognised as already_tested by keyword match"
    )
    assert reg_steps[0]["supporting_evidence_from"], "evidence_from must not be empty"
    assert "ml_v11_alpha" in reg_steps[0]["supporting_evidence_from"]


def test_alpha_increase_already_tested_via_stored_type(tmp_path):
    """Already-tested detection also works when intervention_type is stored directly."""
    exp_base = tmp_path / "results" / "experiments"
    comp_base = tmp_path / "results" / "comparisons"
    _write_experiment(exp_base, "ml_v11", sharpe=0.5, mean_oos=-0.3)
    _write_comparison_evidence(
        comp_base, "ml_v11", "ml_v11_reg",
        tested_change="Set regularisation parameter",
        intervention_type="regularisation_change",
        delta_sharpe=0.01,
        delta_oos=0.03,
    )

    result = api.propose_research_plan(
        research_goal="Reduce OOS variance",
        experiment_prefix="ml",
        experiments_base=exp_base,
        comparisons_base=comp_base,
    )
    reg_steps = [s for s in result["plan"] if s["intervention_type"] == "regularisation_change"]
    assert reg_steps[0]["status"] == "already_tested"


# ---------------------------------------------------------------------------
# 2. risk_adjusted_momentum_20 recognised as already_tested
# ---------------------------------------------------------------------------


def test_risk_adjusted_momentum_feature_addition_already_tested(tmp_path):
    """Feature addition of risk_adjusted_momentum must be recognised as already_tested."""
    exp_base = tmp_path / "results" / "experiments"
    comp_base = tmp_path / "results" / "comparisons"
    _write_experiment(exp_base, "ml_v12", sharpe=0.51, mean_oos=-0.5)
    _write_comparison_evidence(
        comp_base, "ml_v12", "ml_v12_feat",
        tested_change="Added risk_adjusted_momentum_20",
        delta_sharpe=0.009,
        delta_oos=-0.207,
        failure_modes_cand=["poor_oos_consistency"],
    )

    result = api.propose_research_plan(
        research_goal="Improve Sharpe",
        experiment_prefix="ml",
        experiments_base=exp_base,
        comparisons_base=comp_base,
    )
    feat_steps = [s for s in result["plan"] if s["intervention_type"] == "feature_addition"]
    assert feat_steps, "feature_addition suggestion should appear in plan"
    assert feat_steps[0]["status"] == "already_tested", (
        "risk_adjusted_momentum_20 must be recognised as already_tested"
    )
    assert "ml_v12_feat" in feat_steps[0]["supporting_evidence_from"]


# ---------------------------------------------------------------------------
# 3. No invented field paths (train_period_years, n_splits)
# ---------------------------------------------------------------------------


def test_no_invented_validation_fields_in_suggested_change(tmp_path):
    """suggested_change must not contain invented fields like 'train_period_years' or 'n_splits'."""
    exp_base = tmp_path / "results" / "experiments"
    _write_experiment(exp_base, "ml_v1", sharpe=0.5, mean_oos=-0.3)

    result = api.propose_research_plan(
        research_goal="Improve OOS",
        experiment_prefix="ml",
        experiments_base=exp_base,
    )
    for step in result["plan"]:
        sc = step.get("suggested_change", "")
        assert "train_period_years" not in sc, (
            f"Step {step['step']} invents 'train_period_years' — must use real config field"
        )
        assert "n_splits" not in sc, (
            f"Step {step['step']} invents 'n_splits' as a change target — use validation.parameters.*"
        )


def test_validation_step_uses_real_field_path(tmp_path):
    """Validation suggestion must reference 'train_months' (real field) or flag needs_config_inspection."""
    exp_base = tmp_path / "results" / "experiments"
    _write_experiment(exp_base, "ml_v1", sharpe=0.5, mean_oos=-0.3)

    result = api.propose_research_plan(
        research_goal="Improve OOS",
        experiment_prefix="ml",
        experiments_base=exp_base,
    )
    for step in result["plan"]:
        if step["intervention_type"] == "validation_change":
            sc = step.get("suggested_change", "")
            fp = step.get("field_path", "")
            assert "train_months" in sc or step.get("needs_config_inspection"), (
                "Validation step must reference 'train_months' in suggested_change "
                "or set needs_config_inspection=True"
            )
            if fp:
                assert "train_months" in fp or "test_months" in fp or "gap_days" in fp, (
                    f"Validation field_path '{fp}' is not a recognised validation parameter"
                )


# ---------------------------------------------------------------------------
# 4. All field_path values are from the allowed changeable surface
# ---------------------------------------------------------------------------


def test_all_field_paths_are_from_valid_change_paths(tmp_path):
    """Every non-None field_path in plan steps must be in _VALID_CHANGE_PATHS."""
    from src.orchestration.config_generation.draft_validator import split_field_path, change_path_allowed

    exp_base = tmp_path / "results" / "experiments"
    _write_experiment(exp_base, "ml_v1", sharpe=0.5, mean_oos=-0.3)

    result = api.propose_research_plan(
        research_goal="Improve anything",
        experiment_prefix="ml",
        experiments_base=exp_base,
    )
    for step in result["plan"]:
        fp = step.get("field_path")
        if fp is not None:
            section, field = split_field_path(fp)
            assert change_path_allowed(section, field), (
                f"Step field_path '{fp}' is not in _VALID_CHANGE_PATHS — "
                "use None + needs_config_inspection for experiment-specific paths"
            )


def test_none_field_path_steps_have_needs_config_inspection(tmp_path):
    """Steps with field_path=None must set needs_config_inspection=True."""
    exp_base = tmp_path / "results" / "experiments"
    _write_experiment(exp_base, "ml_v1", sharpe=0.5, mean_oos=-0.3)

    result = api.propose_research_plan(
        research_goal="Explore features",
        experiment_prefix="ml",
        experiments_base=exp_base,
    )
    for step in result["plan"]:
        if "field_path" not in step:
            assert step.get("needs_config_inspection") is True, (
                f"Step {step['step']} has no field_path but missing needs_config_inspection"
            )
            assert step.get("inspect_with") == "inspect_experiment_config"


# ---------------------------------------------------------------------------
# 5. Feature replacement classified correctly (not signal_change)
# ---------------------------------------------------------------------------


def test_plan_suggestions_do_not_use_signal_change_for_feature_ops():
    """_PLAN_SUGGESTIONS must not classify feature replacement as signal_change."""
    from src.orchestration.api.research_api import _PLAN_SUGGESTIONS

    for suggestion in _PLAN_SUGGESTIONS:
        sc = suggestion.get("suggested_change", "")
        itype = suggestion["intervention_type"]
        if "replace" in sc.lower() and "feature" in sc.lower():
            assert itype != "signal_change", (
                f"Suggestion '{suggestion['question']}' replaces a feature "
                f"but is classified as signal_change — must be feature_replacement"
            )


def test_catastrophic_split_suggestion_is_feature_replacement():
    """The catastrophic_split suggestion must use intervention_type='feature_replacement'."""
    from src.orchestration.api.research_api import _PLAN_SUGGESTIONS

    catas = [
        s for s in _PLAN_SUGGESTIONS
        if "catastrophic_split" in s.get("trigger_failure_modes", set())
    ]
    assert catas, "There must be at least one catastrophic_split suggestion"
    for s in catas:
        assert s["intervention_type"] == "feature_replacement", (
            f"Catastrophic-split suggestion uses '{s['intervention_type']}' "
            "instead of 'feature_replacement'"
        )


def test_all_suggestion_intervention_types_are_valid():
    """All intervention_type values in _PLAN_SUGGESTIONS must be in INTERVENTION_TYPES."""
    from src.orchestration.api.research_api import _PLAN_SUGGESTIONS

    for suggestion in _PLAN_SUGGESTIONS:
        assert suggestion["intervention_type"] in INTERVENTION_TYPES, (
            f"Suggestion '{suggestion['question']}' has invalid "
            f"intervention_type='{suggestion['intervention_type']}'"
        )


# ---------------------------------------------------------------------------
# 6. Plan returns compact governed envelope
# ---------------------------------------------------------------------------


def test_plan_returns_required_envelope_fields(tmp_path):
    """propose_research_plan must return all required top-level fields."""
    exp_base = tmp_path / "results" / "experiments"
    _write_experiment(exp_base, "ml_v1", sharpe=0.5, mean_oos=-0.3)

    result = api.propose_research_plan(
        research_goal="Test",
        experiment_prefix="ml",
        experiments_base=exp_base,
    )
    assert result["status"] == "ok"
    assert "plan" in result
    assert "advisory_note" in result
    assert "advisory" in result["advisory_note"].lower()
    assert "experiment_count" in result
    assert "common_failure_modes" in result


def test_plan_steps_have_required_fields(tmp_path):
    """Every plan step must carry the required compact fields."""
    exp_base = tmp_path / "results" / "experiments"
    _write_experiment(exp_base, "ml_v1", sharpe=0.5, mean_oos=-0.3)

    result = api.propose_research_plan(
        research_goal="Test steps",
        experiment_prefix="ml",
        experiments_base=exp_base,
    )
    for step in result["plan"]:
        assert "step" in step
        assert "question" in step
        assert "intervention_type" in step
        assert step["status"] in ("candidate", "already_tested")
        assert "suggested_change" in step
        assert "supporting_evidence_from" in step
        assert isinstance(step["supporting_evidence_from"], list)


# ---------------------------------------------------------------------------
# 7. No approval/render/execute side effects
# ---------------------------------------------------------------------------


def test_propose_plan_no_approval_render_execute_calls(tmp_path):
    """propose_research_plan must never call approve, render, or execute."""
    exp_base = tmp_path / "results" / "experiments"
    _write_experiment(exp_base, "ml_v1", sharpe=0.5, mean_oos=-0.3)

    with patch(f"{_MCP_MOD}.approve_experiment_draft") as m_approve, \
         patch(f"{_MCP_MOD}.render_draft_to_yaml") as m_render, \
         patch(f"{_MCP_MOD}.execute_approved_config") as m_execute:
        api.propose_research_plan(
            research_goal="No side effects",
            experiment_prefix="ml",
            experiments_base=exp_base,
        )
    m_approve.assert_not_called()
    m_render.assert_not_called()
    m_execute.assert_not_called()


# ---------------------------------------------------------------------------
# 8. Evidence summary includes prior experiment name
# ---------------------------------------------------------------------------


def test_already_tested_step_includes_prior_experiment_name(tmp_path):
    """already_tested steps must name the prior experiment in supporting_evidence_from."""
    exp_base = tmp_path / "results" / "experiments"
    comp_base = tmp_path / "results" / "comparisons"
    _write_experiment(exp_base, "ml_v11", sharpe=0.5, mean_oos=-0.3)
    _write_comparison_evidence(
        comp_base, "ml_v11", "ml_v11_b",
        tested_change="alpha -> 2.0 regularization increase",
        delta_sharpe=-0.01,
        delta_oos=0.04,
    )

    result = api.propose_research_plan(
        research_goal="Should detect alpha already tested",
        experiment_prefix="ml",
        experiments_base=exp_base,
        comparisons_base=comp_base,
    )
    tested = [s for s in result["plan"] if s["status"] == "already_tested"]
    assert tested, "At least one step must be already_tested"
    for step in tested:
        assert step["supporting_evidence_from"], (
            f"already_tested step '{step['question']}' has no supporting_evidence_from"
        )


def test_evidence_summary_contains_delta_metrics(tmp_path):
    """Evidence summary for already_tested steps must include metric deltas."""
    exp_base = tmp_path / "results" / "experiments"
    comp_base = tmp_path / "results" / "comparisons"
    _write_experiment(exp_base, "ml_v1", sharpe=0.5, mean_oos=-0.3)
    _write_comparison_evidence(
        comp_base, "ml_v1", "ml_v1_alpha",
        tested_change="model.params.alpha from 0.5 to 2.0",
        delta_sharpe=-0.02,
        delta_oos=0.06,
    )

    result = api.propose_research_plan(
        research_goal="Check deltas in evidence",
        experiment_prefix="ml",
        experiments_base=exp_base,
        comparisons_base=comp_base,
    )
    reg = next(
        (s for s in result["plan"]
         if s["intervention_type"] == "regularisation_change" and s["status"] == "already_tested"),
        None,
    )
    assert reg is not None
    evidence_text = reg.get("evidence", "")
    assert "Sharpe" in evidence_text, (
        "Evidence summary must include Sharpe delta when available"
    )


# ---------------------------------------------------------------------------
# 9. Freshly-absent evidence keeps suggestions as candidate
# ---------------------------------------------------------------------------


def test_suggestions_are_candidate_with_no_comparison_evidence(tmp_path):
    """With no comparison evidence, all steps should be 'candidate', not 'already_tested'."""
    exp_base = tmp_path / "results" / "experiments"
    _write_experiment(exp_base, "ml_v1", sharpe=0.5, mean_oos=-0.3)

    result = api.propose_research_plan(
        research_goal="Fresh start",
        experiment_prefix="ml",
        experiments_base=exp_base,
    )
    for step in result["plan"]:
        assert step["status"] == "candidate", (
            f"With no evidence, step '{step['question']}' must be 'candidate'"
        )


# ---------------------------------------------------------------------------
# 10. Keyword match is case-insensitive
# ---------------------------------------------------------------------------


def test_keyword_match_is_case_insensitive(tmp_path):
    """tested_change text matching must be case-insensitive."""
    exp_base = tmp_path / "results" / "experiments"
    comp_base = tmp_path / "results" / "comparisons"
    _write_experiment(exp_base, "ml_v1", sharpe=0.5, mean_oos=-0.3)
    _write_comparison_evidence(
        comp_base, "ml_v1", "ml_v1_reg",
        tested_change="Increased ALPHA from 0.5 to 2.0 (Regularisation)",
        delta_sharpe=-0.01,
        delta_oos=0.02,
    )

    result = api.propose_research_plan(
        research_goal="Case check",
        experiment_prefix="ml",
        experiments_base=exp_base,
        comparisons_base=comp_base,
    )
    reg_steps = [s for s in result["plan"] if s["intervention_type"] == "regularisation_change"]
    assert reg_steps[0]["status"] == "already_tested", (
        "Keyword matching must be case-insensitive — 'ALPHA' and 'Regularisation' must match"
    )


# ---------------------------------------------------------------------------
# 11. model_change keyword precision — "changed model" false-positive regression
# ---------------------------------------------------------------------------


def _alpha_evidence_record() -> dict:
    """Canonical alpha-increase evidence record used across precision tests."""
    return {
        "base_experiment_name": "ml_v11",
        "candidate_experiment_name": "ml_v11_alpha",
        "tested_change": "Changed model.params.alpha from 0.5 to 2.0.",
        "intervention_type": "regularisation_change",
        "metric_deltas": {"delta_sharpe": -0.02, "delta_mean_oos_sharpe": 0.05},
        "failure_modes_base": ["poor_oos_consistency"],
        "failure_modes_candidate": ["poor_oos_consistency"],
    }


def test_alpha_evidence_marks_regularisation_change_already_tested():
    """'Changed model.params.alpha' must mark regularisation_change as already_tested."""
    from src.orchestration.api.research_api import _check_step_already_tested, _PLAN_SUGGESTIONS

    reg_suggestion = next(s for s in _PLAN_SUGGESTIONS if s["intervention_type"] == "regularisation_change")
    is_tested, evidence_from, summary = _check_step_already_tested(reg_suggestion, [_alpha_evidence_record()])
    assert is_tested, "alpha evidence must satisfy regularisation_change"
    assert "ml_v11_alpha" in evidence_from


def test_alpha_evidence_does_not_mark_model_change_already_tested():
    """'Changed model.params.alpha' must NOT satisfy model_change — it is a param change."""
    from src.orchestration.api.research_api import _check_step_already_tested, _PLAN_SUGGESTIONS

    model_suggestion = next(s for s in _PLAN_SUGGESTIONS if s["intervention_type"] == "model_change")
    is_tested, _, _ = _check_step_already_tested(model_suggestion, [_alpha_evidence_record()])
    assert not is_tested, (
        "'Changed model.params.alpha' must not satisfy model_change — "
        "model.params.* changes are regularisation/parameter changes, not model type changes"
    )


def test_model_type_evidence_marks_model_change_already_tested():
    """Evidence with 'model.type changed to ElasticNetRegression' must mark model_change."""
    from src.orchestration.api.research_api import _check_step_already_tested, _PLAN_SUGGESTIONS

    model_suggestion = next(s for s in _PLAN_SUGGESTIONS if s["intervention_type"] == "model_change")
    ev = {
        "base_experiment_name": "ml_v11",
        "candidate_experiment_name": "ml_v11_enet",
        "tested_change": "model.type changed to ElasticNetRegression",
        "intervention_type": None,
        "metric_deltas": {"delta_sharpe": 0.01, "delta_mean_oos_sharpe": 0.03},
    }
    is_tested, evidence_from, _ = _check_step_already_tested(model_suggestion, [ev])
    assert is_tested, "'model.type changed to ElasticNetRegression' must satisfy model_change"
    assert "ml_v11_enet" in evidence_from


def test_ridge_to_elasticnet_marks_model_change():
    """'RidgeRegression -> ElasticNetRegression' must mark model_change."""
    from src.orchestration.api.research_api import _check_step_already_tested, _PLAN_SUGGESTIONS

    model_suggestion = next(s for s in _PLAN_SUGGESTIONS if s["intervention_type"] == "model_change")
    ev = {
        "base_experiment_name": "ml_v11",
        "candidate_experiment_name": "ml_v11_switch",
        "tested_change": "RidgeRegression -> ElasticNetRegression",
        "intervention_type": None,
        "metric_deltas": {"delta_sharpe": 0.03, "delta_mean_oos_sharpe": 0.01},
    }
    is_tested, _, _ = _check_step_already_tested(model_suggestion, [ev])
    assert is_tested, "'RidgeRegression -> ElasticNetRegression' must be recognised as model_change"


def test_ridge_to_lasso_marks_model_change():
    """'RidgeRegression -> LassoRegression' must mark model_change."""
    from src.orchestration.api.research_api import _check_step_already_tested, _PLAN_SUGGESTIONS

    model_suggestion = next(s for s in _PLAN_SUGGESTIONS if s["intervention_type"] == "model_change")
    ev = {
        "base_experiment_name": "ml_v11",
        "candidate_experiment_name": "ml_v11_lasso",
        "tested_change": "RidgeRegression -> LassoRegression",
        "intervention_type": None,
        "metric_deltas": {},
    }
    is_tested, _, _ = _check_step_already_tested(model_suggestion, [ev])
    assert is_tested, "'RidgeRegression -> LassoRegression' must be recognised as model_change"


def test_stored_model_change_type_marks_model_change():
    """Stored intervention_type='model_change' must mark model_change regardless of tested_change text."""
    from src.orchestration.api.research_api import _check_step_already_tested, _PLAN_SUGGESTIONS

    model_suggestion = next(s for s in _PLAN_SUGGESTIONS if s["intervention_type"] == "model_change")
    ev = {
        "base_experiment_name": "ml_v11",
        "candidate_experiment_name": "ml_v11_m",
        "tested_change": "some opaque description with no obvious model keywords",
        "intervention_type": "model_change",
        "metric_deltas": {},
    }
    is_tested, evidence_from, _ = _check_step_already_tested(model_suggestion, [ev])
    assert is_tested, "stored intervention_type='model_change' must always satisfy model_change matching"


# ---------------------------------------------------------------------------
# 12. Full plan — alpha evidence → reg=already_tested, model=candidate
# ---------------------------------------------------------------------------


def test_propose_plan_alpha_evidence_reg_already_tested_model_candidate(tmp_path):
    """Success criterion: after alpha comparison, plan shows reg=already_tested, model=candidate."""
    exp_base = tmp_path / "results" / "experiments"
    comp_base = tmp_path / "results" / "comparisons"
    _write_experiment(exp_base, "ml_v11", sharpe=0.5, mean_oos=-0.3)
    _write_comparison_evidence(
        comp_base, "ml_v11", "ml_v11_alpha",
        tested_change="Changed model.params.alpha from 0.5 to 2.0.",
        delta_sharpe=-0.02,
        delta_oos=0.05,
        failure_modes_cand=["poor_oos_consistency"],
    )

    result = api.propose_research_plan(
        research_goal="Improve OOS stability",
        experiment_prefix="ml",
        experiments_base=exp_base,
        comparisons_base=comp_base,
    )
    assert result["status"] == "ok"
    statuses = {s["intervention_type"]: s["status"] for s in result["plan"]}

    assert statuses.get("regularisation_change") == "already_tested", (
        "regularisation_change must be already_tested after 'Changed model.params.alpha' evidence"
    )
    assert statuses.get("model_change") == "candidate", (
        "model_change must remain candidate — alpha param change is not a model type switch"
    )


def test_propose_plan_alpha_evidence_no_side_effects(tmp_path):
    """propose_research_plan with alpha evidence must not call approve/render/execute."""
    exp_base = tmp_path / "results" / "experiments"
    comp_base = tmp_path / "results" / "comparisons"
    _write_experiment(exp_base, "ml_v11", sharpe=0.5, mean_oos=-0.3)
    _write_comparison_evidence(
        comp_base, "ml_v11", "ml_v11_alpha",
        tested_change="Changed model.params.alpha from 0.5 to 2.0.",
        delta_sharpe=-0.02,
        delta_oos=0.05,
    )

    with patch(f"{_MCP_MOD}.approve_experiment_draft") as m_a, \
         patch(f"{_MCP_MOD}.render_draft_to_yaml") as m_r, \
         patch(f"{_MCP_MOD}.execute_approved_config") as m_e:
        api.propose_research_plan(
            research_goal="No side effects",
            experiment_prefix="ml",
            experiments_base=exp_base,
            comparisons_base=comp_base,
        )
    m_a.assert_not_called()
    m_r.assert_not_called()
    m_e.assert_not_called()
