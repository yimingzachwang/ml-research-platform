"""Main resolver entry point.

resolve_research_spec() is the single public function for Phase 2.

Flow:
  1. Load registry (if not provided)
  2. Expand preset (if spec.preset is set)
  3. Record user-specified fields
  4. Apply defaults according to mode
  5. Detect remaining missing fields
  6. Collect options for missing fields
  7. If required fields are still missing → return needs_user_input
  8. Build ResolvedResearchSpec
  9. Run compatibility checker
  10. If compatibility errors → return invalid
  11. Return resolved
"""

from __future__ import annotations

from typing import Any

from src.agent.critic.compatibility_checker import check_compatibility
from src.agent.intent.research_spec import ResearchSpec
from src.agent.registry.registry_loader import load_registry
from src.agent.registry.registry_schema import ResearchConfigRegistry
from src.agent.resolver.default_resolver import (
    DefaultResolutionResult,
    apply_defaults,
    expand_preset,
    options_for_field,
)
from src.agent.resolver.missing_fields import detect_missing_fields
from src.agent.resolver.resolution_result import ResearchSpecResolutionResult
from src.agent.resolver.resolved_spec import ResolvedResearchSpec


def resolve_research_spec(
    spec: ResearchSpec,
    registry: ResearchConfigRegistry | None = None,
    mode: str = "recommended",
) -> ResearchSpecResolutionResult:
    """Resolve a partial ResearchSpec into a ResolvedResearchSpec.

    Args:
        spec:     The (possibly incomplete) user intent.
        registry: Pre-loaded registry. Loaded from disk if None.
        mode:     'strict' | 'recommended' | 'demo'.

    Returns:
        ResearchSpecResolutionResult with status 'resolved', 'needs_user_input',
        or 'invalid'.
    """
    if registry is None:
        registry = load_registry()

    user_fields: dict[str, Any] = spec.specified_fields()

    # ------------------------------------------------------------------
    # 1. Expand preset (if provided)
    # ------------------------------------------------------------------
    preset_fields: dict[str, Any] = {}
    feature_specs_from_preset: list[dict[str, Any]] = []

    if spec.preset is not None:
        try:
            working_spec, preset_fields, feature_specs_from_preset = expand_preset(spec, registry)
        except KeyError as exc:
            # Unknown preset — return invalid immediately
            return ResearchSpecResolutionResult(
                input_spec=spec,
                resolved_spec=None,
                status="invalid",
                errors=[str(exc)],
            )
    else:
        working_spec = spec

    # ------------------------------------------------------------------
    # 2. Apply defaults
    # ------------------------------------------------------------------
    default_result: DefaultResolutionResult = apply_defaults(working_spec, registry, mode)
    working_spec = default_result.spec

    # ------------------------------------------------------------------
    # 3. Detect missing required fields
    # ------------------------------------------------------------------
    missing = detect_missing_fields(working_spec, registry=registry)

    # ------------------------------------------------------------------
    # 4. Collect options for missing fields
    # ------------------------------------------------------------------
    field_options: dict[str, list[Any]] = {
        f: options_for_field(f, registry, working_spec)
        for f in missing
    }

    # Collect what would be recommended for still-missing fields
    recommended_defaults: dict[str, Any] = {}
    for mf in missing:
        opts = field_options.get(mf, [])
        if opts:
            recommended_defaults[mf] = opts[0]  # first option = recommended

    # Aggregate all warnings and assumptions
    all_assumptions = list(default_result.assumptions)
    all_warnings = list(default_result.warnings)
    all_defaulted = {**default_result.defaulted_fields}

    # ------------------------------------------------------------------
    # 5. If required fields still missing → needs_user_input
    # ------------------------------------------------------------------
    if missing:
        partial = _build_resolved_spec(
            spec=working_spec,
            user_fields=user_fields,
            preset_fields=preset_fields,
            feature_specs_from_preset=feature_specs_from_preset,
            default_result=default_result,
            missing_fields=missing,
            compat_errors=[],
        )
        return ResearchSpecResolutionResult(
            input_spec=spec,
            resolved_spec=partial,
            status="needs_user_input",
            missing_fields=missing,
            field_options=field_options,
            recommended_defaults=recommended_defaults,
            assumptions=all_assumptions,
            defaulted_fields=all_defaulted,
            warnings=all_warnings,
            errors=[],
            requires_confirmation=default_result.requires_confirmation,
        )

    # ------------------------------------------------------------------
    # 6. Build the (complete) ResolvedResearchSpec for compatibility check
    # ------------------------------------------------------------------
    resolved = _build_resolved_spec(
        spec=working_spec,
        user_fields=user_fields,
        preset_fields=preset_fields,
        feature_specs_from_preset=feature_specs_from_preset,
        default_result=default_result,
        missing_fields=[],
        compat_errors=[],
    )

    # ------------------------------------------------------------------
    # 7. Run compatibility checker
    # ------------------------------------------------------------------
    compat = check_compatibility(resolved, registry)
    all_warnings.extend(compat.warning_messages())
    all_warnings.extend(i.message for i in compat.info)

    if not compat.is_valid:
        compat_errors = compat.error_messages()
        resolved = _build_resolved_spec(
            spec=working_spec,
            user_fields=user_fields,
            preset_fields=preset_fields,
            feature_specs_from_preset=feature_specs_from_preset,
            default_result=default_result,
            missing_fields=[],
            compat_errors=compat_errors,
        )
        resolved.warnings = all_warnings
        return ResearchSpecResolutionResult(
            input_spec=spec,
            resolved_spec=resolved,
            status="invalid",
            missing_fields=[],
            field_options={},
            recommended_defaults={},
            assumptions=all_assumptions,
            defaulted_fields=all_defaulted,
            warnings=all_warnings,
            errors=compat_errors,
            requires_confirmation=default_result.requires_confirmation,
        )

    # ------------------------------------------------------------------
    # 8. Return resolved
    # ------------------------------------------------------------------
    resolved.warnings = all_warnings
    return ResearchSpecResolutionResult(
        input_spec=spec,
        resolved_spec=resolved,
        status="resolved",
        missing_fields=[],
        field_options={},
        recommended_defaults={},
        assumptions=all_assumptions,
        defaulted_fields=all_defaulted,
        warnings=all_warnings,
        errors=[],
        requires_confirmation=default_result.requires_confirmation,
    )


# ---------------------------------------------------------------------------
# Internal builder
# ---------------------------------------------------------------------------

def _build_resolved_spec(
    spec: ResearchSpec,
    user_fields: dict[str, Any],
    preset_fields: dict[str, Any],
    feature_specs_from_preset: list[dict[str, Any]],
    default_result: DefaultResolutionResult,
    missing_fields: list[str],
    compat_errors: list[str],
) -> ResolvedResearchSpec:
    """Construct a ResolvedResearchSpec from a fully-worked ResearchSpec.

    Provenance tracking: each field is tagged as user / preset / default / unset.
    """
    # Determine feature_specs: prefer preset, then default, then derive from names
    feature_specs: list[dict[str, Any]]
    if feature_specs_from_preset:
        feature_specs = feature_specs_from_preset
    elif default_result.feature_specs:
        feature_specs = default_result.feature_specs
    else:
        # Derive from feature names using registry defaults
        # (The registry isn't available here; use empty params as fallback.)
        feature_specs = [
            {"type": ft, "params": {}}
            for ft in (spec.features or [])
        ]

    return ResolvedResearchSpec(
        # Content
        preset=spec.preset,
        objective=spec.objective,
        asset_class=spec.asset_class,
        universe=spec.universe,
        tickers=list(spec.tickers or []),
        start_date=spec.start_date,
        end_date=spec.end_date,
        frequency=spec.frequency,
        data_source=spec.data_source,
        feature_families=list(spec.feature_families or []),
        features=list(spec.features or []),
        feature_specs=feature_specs,
        label=spec.label,
        label_horizon=spec.label_horizon,
        model=spec.model,
        model_params=dict(spec.model_params or {}),
        signal=spec.signal,
        signal_params=dict(spec.signal_params or {}),
        validation_protocol=spec.validation_protocol,
        train_window=spec.train_window,
        test_window=spec.test_window,
        transaction_cost_bps=spec.transaction_cost_bps,
        benchmark=spec.benchmark,
        diagnostics=list(spec.diagnostics or []),
        risk_constraints=dict(spec.risk_constraints or {}),
        notes=spec.notes,
        # Provenance
        assumptions=list(default_result.assumptions),
        defaulted_fields=dict(default_result.defaulted_fields),
        user_specified_fields=dict(user_fields),
        preset_fields=dict(preset_fields),
        missing_fields=list(missing_fields),
        warnings=[],
        compatibility_errors=list(compat_errors),
        requires_confirmation=default_result.requires_confirmation,
    )
