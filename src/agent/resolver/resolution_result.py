"""ResearchSpecResolutionResult — the envelope returned by resolve_research_spec()."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.agent.intent.research_spec import ResearchSpec
from src.agent.resolver.resolved_spec import ResolvedResearchSpec

ResolutionStatus = Literal["resolved", "needs_user_input", "invalid"]


@dataclass
class ResearchSpecResolutionResult:
    """Full result of resolving a ResearchSpec.

    Attributes:
        input_spec:           The original, unmodified ResearchSpec.
        resolved_spec:        The resolved (possibly partial) spec. Present in all
                              states; None only on internal programming errors.
        status:               "resolved" / "needs_user_input" / "invalid".
        missing_fields:       Required fields still absent after defaults were applied.
        field_options:        Valid registry choices for each missing field.
        recommended_defaults: What the resolver would fill for missing fields in
                              recommended mode. Shown to the user as suggestions.
        assumptions:          Human-readable descriptions of every default applied.
        defaulted_fields:     Mapping of field name → value filled by defaults.
        warnings:             Non-blocking advisory messages.
        errors:               Blocking compatibility errors (non-empty → invalid).
        requires_confirmation: True when recommended defaults were applied and the
                               user should review the design before execution.
    """

    input_spec: ResearchSpec
    resolved_spec: ResolvedResearchSpec | None
    status: ResolutionStatus
    missing_fields: list[str] = field(default_factory=list)
    field_options: dict[str, list[Any]] = field(default_factory=dict)
    recommended_defaults: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    defaulted_fields: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    requires_confirmation: bool = False

    # -----------------------------------------------------------------

    @property
    def is_resolved(self) -> bool:
        return self.status == "resolved"

    @property
    def needs_input(self) -> bool:
        return self.status == "needs_user_input"

    @property
    def is_invalid(self) -> bool:
        return self.status == "invalid"

    def summary(self) -> str:
        """Single-line human-readable status string."""
        if self.status == "resolved":
            flag = "(requires confirmation)" if self.requires_confirmation else ""
            return f"resolved {flag}".strip()
        if self.status == "needs_user_input":
            return f"needs_user_input — missing: {self.missing_fields}"
        return f"invalid — errors: {self.errors}"
