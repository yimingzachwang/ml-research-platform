"""ResearchSpec — partial / incomplete user research intent.

All fields are optional. This is not an executable experiment config.
It represents a partially-specified research goal that a future resolver,
planner, and critic will complete, validate, and translate into a full
MLExperimentSpec.

Do not inherit from MLExperimentSpec. Do not require fields at construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Iterable


@dataclass
class ResearchSpec:
    """Partial research intent. All fields are optional (default None).

    The resolver (Phase 2) will fill missing fields using default policy.
    The critic (Phase 2) will validate compatibility.
    The planner (Phase 3) will translate this into a full MLExperimentSpec.
    """

    # Identity / framing
    preset: str | None = None
    objective: str | None = None

    # Universe
    asset_class: str | None = None
    universe: str | None = None
    tickers: list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None
    frequency: str | None = None
    data_source: str | None = None

    # Features
    feature_families: list[str] | None = None
    features: list[str] | None = None

    # Label
    label: str | None = None
    label_horizon: int | None = None

    # Model
    model: str | None = None
    model_params: dict[str, Any] | None = None

    # Signal
    signal: str | None = None
    signal_params: dict[str, Any] | None = None

    # Validation
    validation_protocol: str | None = None
    train_window: str | None = None
    test_window: str | None = None

    # Execution
    transaction_cost_bps: float | None = None
    benchmark: str | None = None

    # Output
    diagnostics: list[str] | None = None
    risk_constraints: dict[str, Any] | None = None
    notes: str | None = None

    def specified_fields(self) -> dict[str, Any]:
        """Return only fields that have been set (non-None)."""
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}

    def missing_fields(self, required: Iterable[str]) -> list[str]:
        """Return names from `required` that are not set in this spec."""
        return [name for name in required if getattr(self, name, None) is None]

    def to_dict(self) -> dict[str, Any]:
        """Serialise all fields to a plain dict (None values included)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchSpec:
        """Deserialise from a plain dict. Unknown keys are ignored."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
