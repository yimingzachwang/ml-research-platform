"""ResolvedResearchSpec — a ResearchSpec after defaults and preset expansion.

All content fields are Optional because resolution may be partial (status=needs_user_input).
Provenance fields (assumptions, defaulted_fields, …) are always-present collections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResolvedResearchSpec:
    """Research design after deterministic resolution.

    Content fields mirror ResearchSpec but are populated where possible.
    Provenance fields record every decision made during resolution.

    Not executable — Phase 3 will translate this to an MLExperimentSpec.
    """

    # -----------------------------------------------------------------
    # Content — may still be None if field could not be resolved
    # -----------------------------------------------------------------

    preset: str | None = None
    objective: str | None = None

    asset_class: str | None = None
    universe: str | None = None
    tickers: list[str] = field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None
    frequency: str | None = None
    data_source: str | None = None

    feature_families: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    # Full feature configurations (type + params). Populated from preset or registry defaults.
    feature_specs: list[dict[str, Any]] = field(default_factory=list)

    label: str | None = None
    label_horizon: int | None = None

    model: str | None = None
    model_params: dict[str, Any] = field(default_factory=dict)

    signal: str | None = None
    signal_params: dict[str, Any] = field(default_factory=dict)

    validation_protocol: str | None = None
    train_window: str | None = None
    test_window: str | None = None

    transaction_cost_bps: float | None = None
    benchmark: str | None = None

    diagnostics: list[str] = field(default_factory=list)
    risk_constraints: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    # -----------------------------------------------------------------
    # Provenance — always present, even in partial resolution
    # -----------------------------------------------------------------

    # Human-readable descriptions of every assumption made.
    assumptions: list[str] = field(default_factory=list)

    # Fields that were filled from default_policy.yaml (field → value).
    defaulted_fields: dict[str, Any] = field(default_factory=dict)

    # Fields that came explicitly from the user's ResearchSpec input.
    user_specified_fields: dict[str, Any] = field(default_factory=dict)

    # Fields that came from a preset (after preset expansion).
    preset_fields: dict[str, Any] = field(default_factory=dict)

    # Required fields that remain unresolved after defaults + preset expansion.
    missing_fields: list[str] = field(default_factory=list)

    # Non-blocking advisory messages produced during resolution.
    warnings: list[str] = field(default_factory=list)

    # Blocking compatibility errors (non-empty → status=invalid).
    compatibility_errors: list[str] = field(default_factory=list)

    # True when defaults were applied and the user should review before execution.
    requires_confirmation: bool = False

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def is_complete(self) -> bool:
        """True when no required fields are missing."""
        return not self.missing_fields

    def is_valid(self) -> bool:
        """True when there are no blocking compatibility errors."""
        return not self.compatibility_errors

    def source_of(self, field_name: str) -> str:
        """Return the provenance label for a field: 'user', 'preset', 'default', or 'unset'."""
        if field_name in self.user_specified_fields:
            return "user"
        if field_name in self.preset_fields:
            return "preset"
        if field_name in self.defaulted_fields:
            return "default"
        return "unset"

    def to_dict(self) -> dict[str, Any]:
        """Serialise the resolved spec (content + provenance) to a plain dict."""
        return {
            "preset": self.preset,
            "objective": self.objective,
            "asset_class": self.asset_class,
            "universe": self.universe,
            "tickers": list(self.tickers),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "frequency": self.frequency,
            "data_source": self.data_source,
            "feature_families": list(self.feature_families),
            "features": list(self.features),
            "feature_specs": [dict(s) for s in self.feature_specs],
            "label": self.label,
            "label_horizon": self.label_horizon,
            "model": self.model,
            "model_params": dict(self.model_params),
            "signal": self.signal,
            "signal_params": dict(self.signal_params),
            "validation_protocol": self.validation_protocol,
            "train_window": self.train_window,
            "test_window": self.test_window,
            "transaction_cost_bps": self.transaction_cost_bps,
            "benchmark": self.benchmark,
            "diagnostics": list(self.diagnostics),
            "risk_constraints": dict(self.risk_constraints),
            "notes": self.notes,
            # Provenance
            "assumptions": list(self.assumptions),
            "defaulted_fields": dict(self.defaulted_fields),
            "user_specified_fields": dict(self.user_specified_fields),
            "preset_fields": dict(self.preset_fields),
            "missing_fields": list(self.missing_fields),
            "warnings": list(self.warnings),
            "compatibility_errors": list(self.compatibility_errors),
            "requires_confirmation": self.requires_confirmation,
        }
