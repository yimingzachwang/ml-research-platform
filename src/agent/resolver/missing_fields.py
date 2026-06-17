"""Missing-field detection for ResearchSpec.

The default required fields represent the minimum design decisions needed
to produce a valid experiment config in Phase 3.

features or feature_families: at least one must be set (checked separately
from the simple None-check because either satisfies the requirement).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from src.agent.intent.research_spec import ResearchSpec
from src.agent.registry.registry_schema import ResearchConfigRegistry

# Minimum required fields for a resolved experiment design (Phase 2).
# Note: "features" is NOT in this list — it is handled by the composite
# features-or-feature_families check in detect_missing_fields().
_DEFAULT_REQUIRED_FIELDS: tuple[str, ...] = (
    "asset_class",
    "universe",
    "start_date",
    "end_date",
    "frequency",
    "label",
    "model",
    "signal",
    "validation_protocol",
    "transaction_cost_bps",
)


def detect_missing_fields(
    spec: ResearchSpec,
    required_fields: Iterable[str] | None = None,
    registry: ResearchConfigRegistry | None = None,
) -> list[str]:
    """Return a sorted list of required fields that are not set in spec.

    Args:
        spec:           The ResearchSpec to inspect.
        required_fields: Override the default required field list. If None,
                         uses _DEFAULT_REQUIRED_FIELDS.
        registry:        Unused in Phase 2 (reserved for context-aware checks
                         in Phase 3, e.g., validating ticker availability).

    Returns:
        Sorted list of missing field names. Empty list if spec is complete.
        "features" is added when both spec.features and spec.feature_families
        are absent, regardless of whether "features" is in required_fields.
    """
    rf: tuple[str, ...] = tuple(required_fields) if required_fields is not None else _DEFAULT_REQUIRED_FIELDS

    missing: list[str] = []
    for name in rf:
        val = getattr(spec, name, None)
        if val is None:
            missing.append(name)

    # Composite check: features OR feature_families (at least one required)
    # Only add if "features" is not already in rf (avoid double-counting).
    if "features" not in rf and "feature_families" not in rf:
        has_features = bool(spec.features)
        has_families = bool(spec.feature_families)
        if not has_features and not has_families:
            missing.append("features")

    return sorted(missing)


@dataclass
class MissingFieldReport:
    """Detailed completeness breakdown for a ResearchSpec."""
    missing_required: list[str]
    present_required: list[str]
    has_features: bool        # features or feature_families is populated
    missing_optional: list[str]
    total_required: int
    completeness_pct: float   # present_required / total_required * 100


# Optional fields that are useful but not required for execution.
_OPTIONAL_FIELDS: tuple[str, ...] = (
    "preset",
    "objective",
    "tickers",
    "data_source",
    "feature_families",
    "label_horizon",
    "model_params",
    "signal_params",
    "train_window",
    "test_window",
    "benchmark",
    "diagnostics",
    "risk_constraints",
    "notes",
)


def analyse_spec_completeness(
    spec: ResearchSpec,
    registry: ResearchConfigRegistry,
) -> MissingFieldReport:
    """Produce a rich completeness breakdown for a ResearchSpec.

    Args:
        spec:     The ResearchSpec to analyse.
        registry: Registry instance (used for future field-validity checks).

    Returns:
        MissingFieldReport summarising what is present vs. absent.
    """
    missing_required = detect_missing_fields(spec, registry=registry)
    present_required = [
        f for f in _DEFAULT_REQUIRED_FIELDS if f not in missing_required
    ]
    has_features = bool(spec.features or spec.feature_families)

    missing_optional = [
        f for f in _OPTIONAL_FIELDS
        if getattr(spec, f, None) is None
    ]

    total = len(_DEFAULT_REQUIRED_FIELDS) + 1  # +1 for features composite requirement
    present_count = total - len(missing_required)
    completeness_pct = round(present_count / total * 100, 1) if total else 0.0

    return MissingFieldReport(
        missing_required=missing_required,
        present_required=present_required,
        has_features=has_features,
        missing_optional=missing_optional,
        total_required=total,
        completeness_pct=completeness_pct,
    )
