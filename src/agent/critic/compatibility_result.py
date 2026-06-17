"""CompatibilityIssue and CompatibilityResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class CompatibilityIssue:
    """A single compatibility finding."""
    rule_id: str
    severity: Severity
    message: str
    field: str | None = None


@dataclass
class CompatibilityResult:
    """Aggregated result of check_compatibility()."""

    errors: list[CompatibilityIssue] = field(default_factory=list)
    warnings: list[CompatibilityIssue] = field(default_factory=list)
    info: list[CompatibilityIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when there are no error-severity issues."""
        return not self.errors

    @property
    def all_issues(self) -> list[CompatibilityIssue]:
        return self.errors + self.warnings + self.info

    def error_messages(self) -> list[str]:
        return [i.message for i in self.errors]

    def warning_messages(self) -> list[str]:
        return [i.message for i in self.warnings]
