"""Map a ResolvedResearchSpec's user-specified fields to DraftChange objects.

Only user_specified_fields contribute changes. Preset and default fields are
already captured in the base experiment config and do not represent a diff.

Field categories
----------------
Hard unsupported (trigger unsupported_plan status):
  universe, tickers, frequency, data_source, asset_class
  — structural fields absent from _VALID_CHANGE_PATHS; require whitelist extension.

Soft unsupported (reported in result.unsupported_fields, non-blocking):
  start_date, end_date
  — always user-specified (end_date is required), but date_range is not in
    _VALID_CHANGE_PATHS. Reported so users know the date constraint cannot be
    expressed as a draft change. Draft creation continues with a partial draft.

Silently skipped (provenance / context, not draft parameters):
  preset, objective, benchmark, notes, diagnostics, risk_constraints,
  validation_protocol
  — These identify or contextualise the experiment but have no change-path
    equivalent. preset is base-selection metadata, not a config change.
"""

from __future__ import annotations

import re
from typing import Any

from src.agent.resolver.resolved_spec import ResolvedResearchSpec
from src.orchestration.config_generation.draft_schema import DraftChange

# ---------------------------------------------------------------------------
# Field classification
# ---------------------------------------------------------------------------

# HARD unsupported: structural fields with no _VALID_CHANGE_PATHS entry.
# User-specified values trigger status=unsupported_plan (blocking).
_HARD_UNSUPPORTED_SPEC_FIELDS: frozenset[str] = frozenset({
    "universe",
    "tickers",
    "frequency",
    "data_source",
    "asset_class",
})

# SOFT unsupported: user-specified date fields that have no change-path.
# Reported in result.unsupported_fields but do NOT block draft_created.
# date_range is a structural experiment boundary, not a per-run config param.
_SOFT_UNSUPPORTED_SPEC_FIELDS: frozenset[str] = frozenset({
    "start_date",
    "end_date",
})

# Silently skipped: provenance / context fields with no change-path equivalent.
# These must not be treated as "silently ignored" in the governance sense —
# they are explicitly classified here and their behaviour is tested.
_SILENTLY_SKIPPED_SPEC_FIELDS: frozenset[str] = frozenset({
    "preset",           # base-selection metadata, not a config change
    "objective",
    "benchmark",
    "notes",
    "diagnostics",
    "risk_constraints",
    "validation_protocol",  # validation.type is not in _VALID_CHANGE_PATHS
})

# ---------------------------------------------------------------------------
# Direct scalar mappings: spec field → (section, field)
# ---------------------------------------------------------------------------

_SPEC_SCALAR_TO_CHANGE_PATH: dict[str, tuple[str, str]] = {
    "model":                ("model", "type"),
    "label":                ("labels", "type"),
    "label_horizon":        ("labels", "params.horizon"),
    "signal":               ("signal", "type"),
    "transaction_cost_bps": ("execution", "transaction_cost_bps"),
}

# Sub-fields of model_params dict → (section, field)
_MODEL_PARAM_TO_CHANGE_PATH: dict[str, tuple[str, str]] = {
    "alpha":    ("model", "params.alpha"),
    "C":        ("model", "params.C"),
    "l1_ratio": ("model", "params.l1_ratio"),
    "max_iter": ("model", "params.max_iter"),
}

# Sub-fields of signal_params dict → (section, field)
_SIGNAL_PARAM_TO_CHANGE_PATH: dict[str, tuple[str, str]] = {
    "n":         ("signal", "params.n"),
    "n_long":    ("signal", "params.n_long"),
    "n_short":   ("signal", "params.n_short"),
    "threshold": ("signal", "params.threshold"),
}

_WINDOW_MONTHS_RE = re.compile(r"^(\d+)m$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_window_months(val: Any) -> int | None:
    """Parse a window spec like '48m' → 48, or pass through plain ints."""
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        m = _WINDOW_MONTHS_RE.match(val.strip())
        if m:
            return int(m.group(1))
    return None


def _get_current_value(base_config: dict[str, Any], section: str, change_field: str) -> Any:
    """Navigate into base_config[section] using a dotted field path."""
    if change_field in ("entries.add", "entries.remove"):
        return None
    target: Any = base_config.get(section, {})
    for part in change_field.split("."):
        if not isinstance(target, dict):
            return None
        target = target.get(part)
    return target


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_unsupported_fields(resolved_spec: ResolvedResearchSpec) -> list[str]:
    """Return ALL user-specified fields that cannot be represented as DraftChanges.

    Includes both hard-blocking fields (universe, tickers, …) and soft-reported
    fields (start_date, end_date).  Use detect_hard_unsupported_fields() when
    you need only the blocking subset.

    Silently-skipped provenance fields (preset, objective, …) are NOT returned
    here — they are intentionally excluded because they have no change-path
    equivalent and are classified as metadata, not missing draft changes.
    """
    return [
        f for f in resolved_spec.user_specified_fields
        if f in _HARD_UNSUPPORTED_SPEC_FIELDS or f in _SOFT_UNSUPPORTED_SPEC_FIELDS
    ]


def detect_hard_unsupported_fields(resolved_spec: ResolvedResearchSpec) -> list[str]:
    """Return user-specified fields that block draft creation (structural fields).

    These are outside _VALID_CHANGE_PATHS AND represent structural config
    changes (universe, tickers, frequency, data_source, asset_class) that
    require a whitelist extension before a draft can be created.
    """
    return [
        f for f in resolved_spec.user_specified_fields
        if f in _HARD_UNSUPPORTED_SPEC_FIELDS
    ]


def map_spec_to_changes(
    resolved_spec: ResolvedResearchSpec,
    base_config: dict[str, Any],
) -> tuple[list[DraftChange], list[str]]:
    """Translate user-specified fields in the resolved spec into DraftChanges.

    Only ``user_specified_fields`` contribute changes — preset and default
    fields are already in the base config and do not represent a diff.

    Returns ``(changes, warnings)``.  Warnings are non-blocking advisories
    (e.g. unparseable window format, feature names without feature_specs,
    implied feature removals the draft system cannot auto-generate).
    """
    changes: list[DraftChange] = []
    warnings: list[str] = []
    user = resolved_spec.user_specified_fields

    # --- Scalar field mappings ---
    for spec_field, (section, change_field) in _SPEC_SCALAR_TO_CHANGE_PATH.items():
        if spec_field in user:
            value = getattr(resolved_spec, spec_field, None)
            if value is not None:
                current = _get_current_value(base_config, section, change_field)
                if value != current:
                    changes.append(DraftChange(
                        section=section,
                        field=change_field,
                        current_value=current,
                        proposed_value=value,
                        rationale=f"User specified {spec_field}={value!r}.",
                    ))

    # --- model_params (dict of sub-field overrides) ---
    if "model_params" in user and resolved_spec.model_params:
        for param_name, (section, change_field) in _MODEL_PARAM_TO_CHANGE_PATH.items():
            if param_name in resolved_spec.model_params:
                value = resolved_spec.model_params[param_name]
                current = _get_current_value(base_config, section, change_field)
                if value != current:
                    changes.append(DraftChange(
                        section=section,
                        field=change_field,
                        current_value=current,
                        proposed_value=value,
                        rationale=f"User specified model_params.{param_name}={value!r}.",
                    ))

    # --- signal_params (dict of sub-field overrides) ---
    if "signal_params" in user and resolved_spec.signal_params:
        for param_name, (section, change_field) in _SIGNAL_PARAM_TO_CHANGE_PATH.items():
            if param_name in resolved_spec.signal_params:
                value = resolved_spec.signal_params[param_name]
                current = _get_current_value(base_config, section, change_field)
                if value != current:
                    changes.append(DraftChange(
                        section=section,
                        field=change_field,
                        current_value=current,
                        proposed_value=value,
                        rationale=f"User specified signal_params.{param_name}={value!r}.",
                    ))

    # --- Validation windows ---
    if "train_window" in user and resolved_spec.train_window is not None:
        months = _parse_window_months(resolved_spec.train_window)
        if months is not None:
            current = _get_current_value(base_config, "validation", "parameters.train_months")
            if months != current:
                changes.append(DraftChange(
                    section="validation",
                    field="parameters.train_months",
                    current_value=current,
                    proposed_value=months,
                    rationale=(
                        f"User specified train_window={resolved_spec.train_window!r} "
                        f"→ {months} months."
                    ),
                ))
        else:
            warnings.append(
                f"Could not parse train_window={resolved_spec.train_window!r} as months "
                "(expected e.g. '48m'). Skipping train_months change."
            )

    if "test_window" in user and resolved_spec.test_window is not None:
        months = _parse_window_months(resolved_spec.test_window)
        if months is not None:
            current = _get_current_value(base_config, "validation", "parameters.test_months")
            if months != current:
                changes.append(DraftChange(
                    section="validation",
                    field="parameters.test_months",
                    current_value=current,
                    proposed_value=months,
                    rationale=(
                        f"User specified test_window={resolved_spec.test_window!r} "
                        f"→ {months} months."
                    ),
                ))
        else:
            warnings.append(
                f"Could not parse test_window={resolved_spec.test_window!r} as months "
                "(expected e.g. '12m'). Skipping test_months change."
            )

    # --- Features ---
    if "features" in user or "feature_families" in user or "feature_specs" in user:
        feat_changes, feat_warnings = _map_feature_changes(resolved_spec, base_config)
        changes.extend(feat_changes)
        warnings.extend(feat_warnings)

    return changes, warnings


def _map_feature_changes(
    resolved_spec: ResolvedResearchSpec,
    base_config: dict[str, Any],
) -> tuple[list[DraftChange], list[str]]:
    """Generate entries.add changes for features in the spec not in the base config.

    Phase 3B is add-only: new features are added via entries.add.
    Features in the base config but absent from the user spec (implied removals)
    are reported as warnings — they are NOT silently ignored.
    Missing required feature params are also warned about.

    Phase 3C will add entries.remove support for full replacement semantics.
    """
    from src.experiments.ml_config import get_feature_required_params, get_valid_feature_types

    warnings: list[str] = []
    changes: list[DraftChange] = []

    if not resolved_spec.feature_specs:
        if resolved_spec.features:
            warnings.append(
                "Feature names specified but feature_specs is empty — "
                "cannot generate entries.add changes without full feature configs "
                "(name, type, params). No feature changes added to draft."
            )
        return changes, warnings

    base_entries = base_config.get("features", {}).get("entries", [])
    base_names: set[str] = {
        e.get("name")
        for e in base_entries
        if isinstance(e, dict) and e.get("name")
    }

    valid_types = get_valid_feature_types()
    required_params = get_feature_required_params()
    spec_names: set[str] = set()

    for feat_spec in resolved_spec.feature_specs:
        if not isinstance(feat_spec, dict):
            continue
        name = feat_spec.get("name")
        if not name:
            continue
        spec_names.add(name)

        if name in base_names:
            continue  # already in base config, no add needed

        # Pre-validate: check required params before queuing the add change.
        feat_type = feat_spec.get("type")
        if feat_type not in valid_types:
            warnings.append(
                f"Feature {name!r} has unsupported type {feat_type!r}. "
                f"Valid types: {sorted(valid_types)}. Skipping this add."
            )
            continue

        params = feat_spec.get("params") or {}
        missing_p = [
            r for r in required_params.get(feat_type, frozenset())
            if r not in params
        ]
        if missing_p:
            warnings.append(
                f"Feature {name!r} (type={feat_type!r}) is missing required params: "
                f"{missing_p}. Cannot generate entries.add — validate_draft would reject it."
            )
            continue

        changes.append(DraftChange(
            section="features",
            field="entries.add",
            current_value=None,
            proposed_value=dict(feat_spec),
            rationale=f"Add feature {name!r} (type={feat_type!r}).",
        ))

    # Warn about base features not in the user-specified set (implied removals).
    # These are not silently ignored — they are reported so the user knows the
    # draft is partial. entries.remove support is deferred to Phase 3C.
    implied_removals = sorted(base_names - spec_names)
    if implied_removals:
        warnings.append(
            f"Base config has {len(implied_removals)} feature(s) not in your specified "
            f"set: {implied_removals}. These were NOT removed — Phase 3B supports add-only "
            "feature changes. Use entries.remove explicitly in Phase 3C for full replacement."
        )

    return changes, warnings
