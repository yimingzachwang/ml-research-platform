"""Experiment planner — converts a ResolvedResearchSpec into an ExperimentPlan.

Fully deterministic. No LLM calls. No YAML generation. No experiment execution.

Flow:
  1. Guard: resolved spec must be complete and valid
  2. Infer experiment type from provenance metadata
  3. Select or record baseline
  4. Generate research framing (objective, hypothesis, name, description)
  5. Generate planned changes and controlled variables
  6. Generate success criteria (multi-asset vs single-asset)
  7. Select required diagnostics (filtered against registry)
  8. Generate expected evidence
  9. Generate risks and failure modes
  10. Run plan critic
  11. Return ExperimentPlanResult
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from src.agent.planner.experiment_plan import ExperimentPlan
from src.agent.planner.plan_critic import critique_experiment_plan
from src.agent.planner.plan_result import ExperimentPlanResult
from src.agent.registry.registry_schema import ResearchConfigRegistry
from src.agent.resolver.resolved_spec import ResolvedResearchSpec

# ---------------------------------------------------------------------------
# Experiment type constants
# ---------------------------------------------------------------------------

PRESET_BASELINE = "preset_baseline"
MODEL_COMPARISON = "model_comparison"
FEATURE_FAMILY_TEST = "feature_family_test"
UNIVERSE_TEST = "universe_test"
VALIDATION_TEST = "validation_test"
COST_SENSITIVITY_TEST = "cost_sensitivity_test"
PORTFOLIO_CONSTRUCTION_TEST = "portfolio_construction_test"
CUSTOM_RESEARCH_DESIGN = "custom_research_design"

# Fields that are meaningful to list as controlled variables
_MEANINGFUL_CONTROLLED_FIELDS = frozenset({
    "model", "model_params", "universe", "features", "label", "label_horizon",
    "signal", "signal_params", "validation_protocol", "train_window", "test_window",
    "transaction_cost_bps", "start_date", "end_date", "asset_class", "frequency",
})

# User-provided fields that are non-substantive (not meaningful experiment changes)
_NON_SUBSTANTIVE_USER_FIELDS = frozenset({
    "preset", "end_date", "start_date", "notes", "objective",
})


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_experiment_plan(
    resolved_spec: ResolvedResearchSpec,
    registry: ResearchConfigRegistry | None = None,
    baseline: str | None = None,
) -> ExperimentPlanResult:
    """Convert a ResolvedResearchSpec into an ExperimentPlan.

    Only accepts complete, valid resolved specs (is_complete() and is_valid()
    must both be True). Incomplete or invalid specs return status="invalid".

    Args:
        resolved_spec: A fully-resolved spec from resolve_research_spec().
        registry:      Pre-loaded registry. Loaded from disk if None.
        baseline:      Explicit baseline name override. Inferred if None.

    Returns:
        ExperimentPlanResult with status 'planned', 'needs_review', or 'invalid'.
    """
    # Guard: spec must be complete (all required fields resolved)
    if not resolved_spec.is_complete():
        return ExperimentPlanResult(
            status="invalid",
            plan=None,
            errors=[
                f"Cannot create plan from an incomplete resolved spec. "
                f"Missing fields: {resolved_spec.missing_fields}"
            ],
            summary="invalid — resolved spec is incomplete",
        )

    # Guard: spec must be valid (no compatibility errors)
    if not resolved_spec.is_valid():
        return ExperimentPlanResult(
            status="invalid",
            plan=None,
            errors=[
                f"Cannot create plan from an invalid resolved spec. "
                f"Compatibility errors: {resolved_spec.compatibility_errors}"
            ],
            summary="invalid — resolved spec has compatibility errors",
        )

    if registry is None:
        from src.agent.registry.registry_loader import load_registry
        registry = load_registry()

    warnings_acc: list[str] = []

    # Step 1: Infer experiment type from provenance
    experiment_type = _infer_experiment_type(resolved_spec)

    # Step 2: Select or record baseline
    if baseline is None:
        baseline, baseline_warnings = _select_baseline(resolved_spec, registry)
        warnings_acc.extend(baseline_warnings)

    # Step 3: Generate research framing
    hypothesis = _generate_hypothesis(experiment_type, resolved_spec, baseline, registry)
    research_objective = _generate_objective(resolved_spec, experiment_type, baseline)
    name = _generate_name(resolved_spec, experiment_type)
    description = _generate_description(resolved_spec, experiment_type, baseline)
    comparison_policy = _comparison_policy(experiment_type, baseline)
    plan_id = _generate_plan_id(resolved_spec)

    # Step 4: Planned changes and controlled variables
    planned_changes = _generate_planned_changes(
        resolved_spec, experiment_type, baseline, registry
    )
    controlled_variables = _identify_controlled_variables(resolved_spec, experiment_type)

    # Step 5: Success criteria
    success_criteria, primary_metric, secondary_metrics = _generate_success_criteria(
        resolved_spec
    )

    # Step 6: Required diagnostics
    required_diagnostics, diag_warnings = _select_required_diagnostics(
        resolved_spec, registry
    )
    warnings_acc.extend(diag_warnings)

    # Step 7: Expected evidence
    expected_evidence = _generate_expected_evidence(resolved_spec, experiment_type)

    # Step 8: Risks and failure modes
    risks, failure_modes = _generate_risks(resolved_spec, experiment_type)

    # Carry forward resolver warnings and assumptions
    if resolved_spec.warnings:
        warnings_acc.extend(resolved_spec.warnings)
    assumptions = list(resolved_spec.assumptions)

    # Build the plan
    plan = ExperimentPlan(
        plan_id=plan_id,
        name=name,
        description=description,
        resolved_spec=resolved_spec,
        research_objective=research_objective,
        hypothesis=hypothesis,
        experiment_type=experiment_type,
        baseline=baseline,
        comparison_policy=comparison_policy,
        planned_changes=planned_changes,
        controlled_variables=controlled_variables,
        success_criteria=success_criteria,
        primary_metric=primary_metric,
        secondary_metrics=secondary_metrics,
        required_diagnostics=required_diagnostics,
        expected_evidence=expected_evidence,
        risks=risks,
        failure_modes_to_monitor=failure_modes,
        assumptions=assumptions,
        warnings=warnings_acc,
        requires_confirmation=resolved_spec.requires_confirmation,
        is_executable_ready=False,
        created_at=None,
    )

    # Step 10: Run plan critic
    critic_result = critique_experiment_plan(plan, registry)

    return _make_plan_result(plan, critic_result, warnings_acc)


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def _make_plan_result(
    plan: ExperimentPlan,
    critic_result: Any,
    warnings_acc: list[str],
) -> ExperimentPlanResult:
    """Translate critic output into an ExperimentPlanResult with appropriate status."""
    if not critic_result.is_valid:
        return ExperimentPlanResult(
            status="invalid",
            plan=plan,
            errors=[],
            warnings=warnings_acc,
            critic_errors=critic_result.error_messages(),
            critic_warnings=critic_result.warning_messages(),
            summary=f"invalid — critic errors: {critic_result.error_messages()}",
        )

    all_warnings = warnings_acc + critic_result.warning_messages()
    if all_warnings or plan.requires_confirmation:
        n = len(all_warnings)
        confirmation_note = " (requires confirmation)" if plan.requires_confirmation else ""
        return ExperimentPlanResult(
            status="needs_review",
            plan=plan,
            errors=[],
            warnings=warnings_acc,
            critic_errors=[],
            critic_warnings=critic_result.warning_messages(),
            summary=f"needs_review — {n} warning(s){confirmation_note}",
        )

    return ExperimentPlanResult(
        status="planned",
        plan=plan,
        errors=[],
        warnings=[],
        critic_errors=[],
        critic_warnings=[],
        summary="planned",
    )


# ---------------------------------------------------------------------------
# Experiment type inference
# ---------------------------------------------------------------------------

def _infer_experiment_type(spec: ResolvedResearchSpec) -> str:
    """Deterministically infer experiment type from provenance metadata.

    Reads spec.user_specified_fields to identify what the user changed.
    Priority order: model > features > universe > validation > cost > preset_baseline.
    """
    user_fields = set(spec.user_specified_fields.keys())

    # Model override → model comparison
    if "model" in user_fields:
        return MODEL_COMPARISON

    # Feature change → feature family test
    if "features" in user_fields or "feature_families" in user_fields:
        return FEATURE_FAMILY_TEST

    # Universe change → universe test
    if "universe" in user_fields:
        return UNIVERSE_TEST

    # Validation change → validation test
    if any(f in user_fields for f in ("validation_protocol", "train_window", "test_window")):
        return VALIDATION_TEST

    # Transaction cost change → cost sensitivity test
    if "transaction_cost_bps" in user_fields:
        return COST_SENSITIVITY_TEST

    # Pure preset (user only set non-substantive fields like preset/end_date/start_date)
    substantive_user_fields = user_fields - _NON_SUBSTANTIVE_USER_FIELDS
    if not substantive_user_fields and spec.preset is not None:
        return PRESET_BASELINE

    return CUSTOM_RESEARCH_DESIGN


# ---------------------------------------------------------------------------
# Baseline selection
# ---------------------------------------------------------------------------

def _select_baseline(
    spec: ResolvedResearchSpec,
    registry: ResearchConfigRegistry,
) -> tuple[str | None, list[str]]:
    """Select or infer the most appropriate baseline for comparison.

    Rules (in priority order):
      1. If spec came from a preset, that preset is the baseline.
      2. If universe matches canonical_multi_asset_etfs and that preset exists,
         use canonical_multi_asset_etf as inferred baseline (with a warning).
      3. Otherwise, baseline is None with a warning.
    """
    warnings: list[str] = []

    if spec.preset is not None:
        return spec.preset, warnings

    if spec.universe == "canonical_multi_asset_etfs":
        if "canonical_multi_asset_etf" in registry.preset_names():
            warnings.append(
                "No preset specified; inferred baseline as 'canonical_multi_asset_etf' "
                "because the universe matches. Verify this is the correct reference."
            )
            return "canonical_multi_asset_etf", warnings

    warnings.append(
        "No valid baseline could be inferred from the resolved spec. "
        "Define a comparison target manually before interpreting results."
    )
    return None, warnings


# ---------------------------------------------------------------------------
# Hypothesis generation
# ---------------------------------------------------------------------------

def _generate_hypothesis(
    experiment_type: str,
    spec: ResolvedResearchSpec,
    baseline: str | None,
    registry: ResearchConfigRegistry | None,
) -> str:
    """Generate a template-based, deterministic hypothesis."""
    if experiment_type == PRESET_BASELINE:
        return (
            "This run establishes a reproducible baseline for the selected "
            "research configuration."
        )

    if experiment_type == MODEL_COMPARISON:
        selected_model = spec.model or "the selected model"
        baseline_model = _get_baseline_model(baseline, registry)
        return (
            f"Replacing {baseline_model} with {selected_model} may improve predictive "
            "ranking quality, but may increase overfitting risk due to different "
            "regularisation properties."
        )

    if experiment_type == FEATURE_FAMILY_TEST:
        return (
            "The selected feature set may improve out-of-sample ranking stability if "
            "the added signals provide persistent cross-sectional information that is "
            "not captured by the baseline feature configuration."
        )

    if experiment_type == UNIVERSE_TEST:
        return (
            "Changing the universe may alter signal stability and realised spread "
            "because asset relationships and regime exposure differ across universes."
        )

    if experiment_type == VALIDATION_TEST:
        return (
            "Modifying the validation protocol may affect the reliability of "
            "performance estimates and should be compared against the baseline protocol "
            "to confirm that the change does not introduce look-ahead bias."
        )

    if experiment_type == COST_SENSITIVITY_TEST:
        return (
            "The strategy remains useful only if performance is robust after the "
            "selected transaction cost assumption. High turnover relative to signal "
            "strength will erode net returns."
        )

    if experiment_type == PORTFOLIO_CONSTRUCTION_TEST:
        return (
            "The portfolio construction method may affect turnover, concentration, "
            "and the realised return distribution across walk-forward periods."
        )

    # CUSTOM_RESEARCH_DESIGN
    return (
        "The experiment explores a research design that differs from standard presets. "
        "Results should be compared against a defined baseline to establish whether "
        "the design provides a genuine edge."
    )


def _get_baseline_model(
    baseline: str | None,
    registry: ResearchConfigRegistry | None,
) -> str:
    """Look up the model type used in a baseline preset."""
    if baseline is None or registry is None:
        return "the baseline model"
    try:
        preset = registry.preset(baseline)
        if isinstance(preset.model, dict):
            return preset.model.get("type", "the baseline model")
    except KeyError:
        pass
    return "the baseline model"


# ---------------------------------------------------------------------------
# Research objective
# ---------------------------------------------------------------------------

def _generate_objective(
    spec: ResolvedResearchSpec,
    experiment_type: str,
    baseline: str | None,
) -> str:
    if experiment_type == PRESET_BASELINE:
        preset_label = f"'{spec.preset}'" if spec.preset else "the selected"
        return (
            f"Establish a reproducible performance baseline for the "
            f"{preset_label} research configuration."
        )

    if experiment_type == MODEL_COMPARISON:
        model = spec.model or "the selected model"
        baseline_label = f"'{baseline}'" if baseline else "the baseline"
        return (
            f"Evaluate whether '{model}' produces better risk-adjusted performance "
            f"than the {baseline_label} baseline model."
        )

    if experiment_type == FEATURE_FAMILY_TEST:
        return (
            "Determine whether the selected feature set improves out-of-sample "
            "performance compared to the baseline feature configuration."
        )

    if experiment_type == UNIVERSE_TEST:
        universe_label = f"'{spec.universe}'" if spec.universe else "the selected universe"
        return (
            f"Evaluate the impact of using {universe_label} on signal stability "
            "and strategy performance."
        )

    if experiment_type == VALIDATION_TEST:
        return (
            "Assess whether the modified validation protocol produces more reliable "
            "out-of-sample performance estimates than the baseline protocol."
        )

    if experiment_type == COST_SENSITIVITY_TEST:
        tc = spec.transaction_cost_bps
        tc_label = f"{tc} bps" if tc is not None else "the stated"
        return (
            f"Determine whether the strategy generates sufficient alpha to remain "
            f"profitable at {tc_label} transaction cost."
        )

    if experiment_type == PORTFOLIO_CONSTRUCTION_TEST:
        return (
            "Assess the impact of the portfolio construction method on strategy "
            "turnover, concentration, and realised return distribution."
        )

    return (
        "Explore a research configuration that differs from standard presets and "
        "establish initial performance characteristics."
    )


# ---------------------------------------------------------------------------
# Name and description
# ---------------------------------------------------------------------------

def _generate_name(spec: ResolvedResearchSpec, experiment_type: str) -> str:
    parts: list[str] = []
    if spec.preset:
        parts.append(spec.preset)
    if experiment_type == MODEL_COMPARISON and spec.model:
        parts.append(spec.model)
    elif experiment_type == UNIVERSE_TEST and spec.universe:
        parts.append(spec.universe)
    if experiment_type not in (PRESET_BASELINE, CUSTOM_RESEARCH_DESIGN):
        parts.append(experiment_type)
    if not parts:
        if spec.universe:
            parts.append(spec.universe)
        if spec.model:
            parts.append(spec.model)
        parts.append(experiment_type)
    return " / ".join(parts) if parts else "custom_experiment"


def _generate_description(
    spec: ResolvedResearchSpec,
    experiment_type: str,
    baseline: str | None,
) -> str:
    detail_parts: list[str] = []
    if spec.universe:
        detail_parts.append(f"Universe: {spec.universe}")
    if spec.model:
        detail_parts.append(f"Model: {spec.model}")
    if spec.start_date and spec.end_date:
        detail_parts.append(f"Period: {spec.start_date} to {spec.end_date}")
    return (
        f"Experiment type: {experiment_type}. "
        + (f"Baseline: {baseline}. " if baseline else "No baseline. ")
        + " | ".join(detail_parts)
    )


# ---------------------------------------------------------------------------
# Plan ID (deterministic hash)
# ---------------------------------------------------------------------------

def _generate_plan_id(spec: ResolvedResearchSpec) -> str:
    """Generate a short deterministic ID from key spec fields."""
    key_parts = [
        spec.universe or "",
        spec.model or "",
        spec.label or "",
        spec.signal or "",
        spec.start_date or "",
        spec.end_date or "",
        ",".join(sorted(spec.features or [])),
    ]
    content = "|".join(key_parts)
    return "plan_" + hashlib.sha256(content.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Comparison policy
# ---------------------------------------------------------------------------

def _comparison_policy(experiment_type: str, baseline: str | None) -> str:
    if experiment_type == PRESET_BASELINE:
        return "No comparison required — this is the baseline run itself."
    if baseline:
        return (
            f"Compare against '{baseline}' baseline on OOS Sharpe ratio, "
            "mean fold Sharpe, and maximum drawdown. "
            "A result is considered materially better only if improvement exceeds "
            "expected sampling noise across walk-forward folds."
        )
    return (
        "No baseline has been inferred. Define a comparison target manually "
        "before interpreting results."
    )


# ---------------------------------------------------------------------------
# Planned changes
# ---------------------------------------------------------------------------

def _generate_planned_changes(
    spec: ResolvedResearchSpec,
    experiment_type: str,
    baseline: str | None,
    registry: ResearchConfigRegistry | None,
) -> list[str]:
    """Build a list of user-driven changes from the provenance record."""
    user_fields = spec.user_specified_fields
    changes: list[str] = []

    if experiment_type == PRESET_BASELINE:
        preset_label = f"'{spec.preset}'" if spec.preset else "the selected preset"
        changes.append(
            f"Running {preset_label} as-is to establish a reproducible baseline."
        )
        return changes

    if "model" in user_fields:
        baseline_model = _get_baseline_model(baseline, registry)
        selected_model = spec.model or "unknown"
        if baseline_model != "the baseline model" and baseline_model != selected_model:
            changes.append(
                f"Model changed from '{baseline_model}' (baseline) to "
                f"'{selected_model}' (user specified)."
            )
        else:
            changes.append(f"Model set to '{selected_model}' (user specified).")

    if "features" in user_fields:
        n = len(spec.features or [])
        changes.append(
            f"Feature set specified by user: {spec.features} ({n} feature(s))."
        )

    if "feature_families" in user_fields:
        changes.append(
            f"Feature families specified by user: {spec.feature_families}."
        )

    if "universe" in user_fields:
        changes.append(f"Universe changed to '{spec.universe}' (user specified).")

    if "validation_protocol" in user_fields:
        changes.append(
            f"Validation protocol changed to '{spec.validation_protocol}' (user specified)."
        )

    if "train_window" in user_fields or "test_window" in user_fields:
        changes.append(
            f"Walk-forward windows changed: "
            f"train={spec.train_window}, test={spec.test_window}."
        )

    if "transaction_cost_bps" in user_fields:
        changes.append(
            f"Transaction cost set to {spec.transaction_cost_bps} bps (user specified)."
        )

    if "label" in user_fields:
        changes.append(f"Label changed to '{spec.label}' (user specified).")

    if "signal" in user_fields:
        changes.append(f"Signal changed to '{spec.signal}' (user specified).")

    if not changes:
        changes.append(
            "No explicit user-specified changes identified; "
            "experiment uses resolved defaults."
        )

    return changes


# ---------------------------------------------------------------------------
# Controlled variables
# ---------------------------------------------------------------------------

def _identify_controlled_variables(
    spec: ResolvedResearchSpec,
    experiment_type: str,
) -> list[str]:
    """Identify fields held constant from preset or defaults (not user-changed)."""
    user_fields = set(spec.user_specified_fields.keys())
    held_constant = (
        set(spec.preset_fields.keys()) | set(spec.defaulted_fields.keys())
    ) - user_fields
    held_constant &= _MEANINGFUL_CONTROLLED_FIELDS

    controlled: list[str] = []
    for field_name in sorted(held_constant):
        value = getattr(spec, field_name, None)
        if value is not None:
            source = spec.source_of(field_name)
            controlled.append(f"{field_name}: {value!r} (from {source})")
    return controlled


# ---------------------------------------------------------------------------
# Success criteria
# ---------------------------------------------------------------------------

def _generate_success_criteria(
    spec: ResolvedResearchSpec,
) -> tuple[list[str], str | None, list[str]]:
    """Return (criteria, primary_metric, secondary_metrics).

    Criteria are strategy-type-appropriate. Multi-asset strategies receive
    cross-sectional evaluation criteria. Single-asset strategies receive
    directional accuracy criteria.
    """
    is_multi_asset = spec.signal in ("top_n", "long_short", "normalize")

    if is_multi_asset:
        criteria = [
            "OOS Sharpe ratio is positive after transaction costs",
            "Rolling information coefficient (IC) is consistently positive "
            "across walk-forward folds",
            "Rank monotonicity: top quantile outperforms bottom quantile "
            "in realised returns",
            "Top-minus-bottom realised spread is positive after transaction costs",
            "Turnover is within acceptable bounds for the stated transaction "
            "cost assumption",
            "Maximum drawdown does not exceed an acceptable threshold",
        ]
        primary_metric = "OOS Sharpe ratio"
        secondary_metrics = [
            "Rolling IC stability",
            "Rank monotonicity (top/bottom spread)",
            "Turnover-adjusted return",
            "Maximum drawdown",
        ]
    else:
        criteria = [
            "OOS Sharpe ratio is positive after transaction costs",
            "Hit rate exceeds 0.5 over the out-of-sample period",
            "Maximum drawdown does not exceed an acceptable threshold",
            "Turnover is within acceptable bounds for the stated transaction "
            "cost assumption",
            "Benchmark-relative return is positive over the evaluation period",
        ]
        primary_metric = "OOS Sharpe ratio"
        secondary_metrics = [
            "Hit rate (directional accuracy)",
            "Maximum drawdown",
            "Benchmark-relative return",
            "Turnover",
        ]

    return criteria, primary_metric, secondary_metrics


# ---------------------------------------------------------------------------
# Required diagnostics
# ---------------------------------------------------------------------------

def _select_required_diagnostics(
    spec: ResolvedResearchSpec,
    registry: ResearchConfigRegistry | None,
) -> tuple[list[str], list[str]]:
    """Return (recommended_diagnostics, warnings).

    Only recommends diagnostics present in the registry. Records a warning
    for any desired diagnostic that is absent.
    """
    warnings: list[str] = []

    # Desired diagnostics in priority order
    desired = [
        "split_metrics",       # Required for artefact completeness
        "sharpe_ratio",        # Core OOS performance metric
        "max_drawdown",        # Risk metric
        "feature_importance",  # Feature analysis
        "mean_oos_sharpe",     # Fold-averaged performance
        "annual_return",       # Annualised return
    ]

    if registry is not None:
        registry_diags = registry.diagnostic_names()
        validated = [d for d in desired if d in registry_diags]
        for d in desired:
            if d not in registry_diags:
                warnings.append(
                    f"Desired diagnostic '{d}' is not in the current registry "
                    "and was excluded from the plan."
                )
    else:
        validated = list(desired)

    return validated, warnings


# ---------------------------------------------------------------------------
# Expected evidence
# ---------------------------------------------------------------------------

def _generate_expected_evidence(
    spec: ResolvedResearchSpec,
    experiment_type: str,
) -> list[str]:
    evidence = [
        "Walk-forward OOS performance metrics per fold (split_metrics.json)",
        "Feature importance or coefficient magnitudes across folds",
        "OOS Sharpe ratio and maximum drawdown (metrics.json)",
    ]

    if experiment_type == MODEL_COMPARISON:
        evidence.append(
            "Side-by-side Sharpe ratio and drawdown comparison against baseline model"
        )
        evidence.append(
            "Feature importance ranking to compare regularisation effect across models"
        )

    elif experiment_type == FEATURE_FAMILY_TEST:
        evidence.append("Feature importance ranking for the specified feature set")
        evidence.append(
            "Fold-level IC stability to identify regime sensitivity of the feature set"
        )

    elif experiment_type == UNIVERSE_TEST:
        evidence.append(
            "Asset-level contribution to portfolio performance across walk-forward folds"
        )
        evidence.append(
            "Comparison of Sharpe ratio and volatility against the baseline universe"
        )

    elif experiment_type == COST_SENSITIVITY_TEST:
        evidence.append("Net Sharpe at stated transaction cost vs gross Sharpe (zero cost)")
        evidence.append("Turnover breakdown by walk-forward period")

    elif experiment_type == VALIDATION_TEST:
        evidence.append("Fold count and distribution of train/test periods")
        evidence.append(
            "Comparison of fold-level Sharpe stability against the baseline protocol"
        )

    return evidence


# ---------------------------------------------------------------------------
# Risks and failure modes
# ---------------------------------------------------------------------------

def _generate_risks(
    spec: ResolvedResearchSpec,
    experiment_type: str,
) -> tuple[list[str], list[str]]:
    """Return (risks, failure_modes_to_monitor)."""
    risks: list[str] = []
    failure_modes: list[str] = []

    # Model risks
    model = spec.model
    if model == "LassoRegression":
        risks.append(
            "Lasso regularisation may aggressively zero out features; "
            "verify that selected features remain stable across folds."
        )
    elif model == "ElasticNetRegression":
        risks.append(
            "ElasticNet combines L1 and L2 penalties; "
            "regularisation strength may dominate prediction variance."
        )
    elif model == "LogisticRegression":
        risks.append(
            "Logistic regression output may not calibrate well to continuous "
            "return targets; use only with binary_direction label."
        )
    elif model == "LinearRegression":
        risks.append(
            "Linear regression without regularisation may overfit when feature "
            "count approaches sample size."
        )
    elif model == "RidgeRegression":
        risks.append(
            "Ridge compression shrinks all coefficients; prediction variance may "
            "be too low for clear cross-sectional discrimination."
        )

    # Feature risks
    features = spec.features or []
    if len(features) > 8:
        risks.append(
            f"High feature count ({len(features)}) may reduce interpretability "
            "and increase overfitting risk in small universes."
        )
    if "rolling_beta" in features:
        risks.append(
            "rolling_beta without an explicit market_ticker parameter defaults "
            "to SPY; verify this is appropriate for the chosen universe."
        )
    if features:
        risks.append(
            "Feature effects may be regime-dependent; validate performance across "
            "bull, bear, and high-volatility market cycles."
        )

    # Strategy risks
    is_multi_asset = spec.signal in ("top_n", "long_short", "normalize")
    if is_multi_asset:
        risks.append(
            "Cross-sectional ranking can fail if prediction dispersion is weak or "
            "regularisation compression is too strong."
        )
        risks.append(
            "High turnover can erase signal value; verify that net-of-cost Sharpe "
            "remains positive at the stated transaction cost assumption."
        )
        failure_modes.append(
            "Rank dispersion collapses: all assets receive similar predictions, "
            "producing random top-N selection."
        )
        failure_modes.append(
            "Rolling IC turns consistently negative: the signal has no persistent "
            "cross-sectional predictive power."
        )
    else:
        risks.append(
            "Single-asset strategy cannot diversify idiosyncratic risk; "
            "drawdowns may be deeper than multi-asset designs."
        )
        failure_modes.append(
            "Hit rate falls below 0.5 OOS: the directional signal has no "
            "persistent edge."
        )

    # Validation risks
    if spec.train_window:
        try:
            train_months = int(spec.train_window.rstrip("m"))
            if train_months < 24:
                risks.append(
                    f"Short train window ({spec.train_window}) may produce unreliable "
                    "coefficient estimates due to insufficient historical variation."
                )
                failure_modes.append(
                    f"Insufficient training data ({spec.train_window}): "
                    "model coefficients are unstable across folds."
                )
        except (ValueError, AttributeError):
            pass

    # Time range risks
    if spec.start_date and spec.end_date:
        try:
            start = datetime.strptime(spec.start_date, "%Y-%m-%d")
            end = datetime.strptime(spec.end_date, "%Y-%m-%d")
            years = (end - start).days / 365.25
            if years < 5:
                risks.append(
                    f"Short evaluation timeframe ({years:.1f} years) limits statistical "
                    "confidence; results may reflect a single market regime."
                )
        except (ValueError, TypeError):
            pass

    # Transaction cost risks
    if spec.transaction_cost_bps is not None and spec.transaction_cost_bps == 0.0:
        risks.append(
            "Zero transaction cost assumption produces optimistic gross-performance "
            "estimates; verify viability at realistic cost levels."
        )
        failure_modes.append(
            "Net-of-cost performance is materially worse than gross performance at "
            "realistic transaction cost assumptions."
        )

    # Universe risks
    if spec.universe == "custom_ticker_list" or (
        spec.universe and not spec.tickers
    ):
        risks.append(
            "Custom or empty ticker universe may have inconsistent data coverage; "
            "verify that all tickers have sufficient history."
        )

    if spec.asset_class in ("multi_asset",):
        risks.append(
            "Mixed-asset universe combines structurally different instruments; "
            "cross-asset correlation regime may shift, affecting signal stability."
        )

    # Universal failure modes
    failure_modes.append(
        "OOS Sharpe ratio is negative or near zero: no genuine edge detected."
    )
    failure_modes.append(
        "Feature importance is concentrated in a single feature: "
        "model is effectively a single-factor design."
    )

    return risks, failure_modes
