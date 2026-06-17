"""next_action_critic — validates NextActionRecommendationResult.

Phase 5B: Memory-Aware Next-Action Planner.

Critic checks (CRITIC_A through CRITIC_J):

  A. No candidate may claim guaranteed performance improvement.
  B. High-priority candidates should cite evidence IDs when evidence exists.
  C. No candidate may have creates_draft=True or executes_experiment=True.
  D. Every candidate must require user confirmation.
  E. Every candidate must include risks or failure_modes_to_monitor.
  F. status="recommended" requires at least one candidate.
  G. status="needs_more_evidence" summary must explain what evidence is missing.
  H. Every candidate priority must be a valid value.
  I. Every candidate confidence must be a valid value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.agent.next_actions.next_action import VALID_CONFIDENCES, VALID_PRIORITIES
from src.agent.next_actions.next_action_result import NextActionRecommendationResult

_GUARANTEE_PHRASES: frozenset[str] = frozenset(
    {
        "will improve",
        "guaranteed",
        "will increase",
        "will outperform",
        "will definitely",
        "certainly improve",
        "is guaranteed",
        "will boost",
    }
)

_NEEDS_MORE_EVIDENCE_KEYWORDS: frozenset[str] = frozenset(
    {"evidence", "memory", "baseline", "index", "artefact", "run"}
)


@dataclass
class NextActionCriticIssue:
    check_id: str          # "CRITIC_A" through "CRITIC_J"
    severity: str          # "error" | "warning"
    message: str
    candidate_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "check_id": self.check_id,
            "severity": self.severity,
            "message": self.message,
            "candidate_id": self.candidate_id,
        }


@dataclass
class NextActionCriticResult:
    issues: list[NextActionCriticIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[NextActionCriticIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[NextActionCriticIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "issues": [i.to_dict() for i in self.issues],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "is_valid": self.is_valid,
        }


def critique_next_action_recommendations(
    result: NextActionRecommendationResult,
) -> NextActionCriticResult:
    """Validate a NextActionRecommendationResult against Phase 5B governance checks."""
    issues: list[NextActionCriticIssue] = []

    has_evidence = (
        result.memory_context is not None and not result.memory_context.is_empty()
    )

    # --- Per-candidate checks ---
    for candidate in result.candidates:
        cid = candidate.action_id

        # A. No performance guarantees
        text = " ".join(
            [
                candidate.rationale or "",
                candidate.description or "",
                candidate.expected_benefit or "",
            ]
        ).lower()
        for phrase in _GUARANTEE_PHRASES:
            if phrase in text:
                issues.append(
                    NextActionCriticIssue(
                        check_id="CRITIC_A",
                        severity="error",
                        message=(
                            f"Candidate '{cid}' claims a performance guarantee: "
                            f"found phrase '{phrase}'."
                        ),
                        candidate_id=cid,
                    )
                )
                break

        # B. High-priority candidates need evidence IDs when evidence exists
        if (
            has_evidence
            and candidate.priority == "high"
            and not candidate.supporting_evidence_ids
        ):
            issues.append(
                NextActionCriticIssue(
                    check_id="CRITIC_B",
                    severity="warning",
                    message=(
                        f"Candidate '{cid}' is high priority but carries no "
                        f"supporting_evidence_ids despite memory evidence being available."
                    ),
                    candidate_id=cid,
                )
            )

        # C. No creates_draft or executes_experiment
        if candidate.creates_draft:
            issues.append(
                NextActionCriticIssue(
                    check_id="CRITIC_C",
                    severity="error",
                    message=(
                        f"Candidate '{cid}' has creates_draft=True. "
                        f"Phase 5B candidates must not create drafts."
                    ),
                    candidate_id=cid,
                )
            )
        if candidate.executes_experiment:
            issues.append(
                NextActionCriticIssue(
                    check_id="CRITIC_C",
                    severity="error",
                    message=(
                        f"Candidate '{cid}' has executes_experiment=True. "
                        f"Phase 5B candidates must not execute experiments."
                    ),
                    candidate_id=cid,
                )
            )

        # D. Every candidate requires user confirmation
        if not candidate.requires_user_confirmation:
            issues.append(
                NextActionCriticIssue(
                    check_id="CRITIC_D",
                    severity="error",
                    message=(
                        f"Candidate '{cid}' has requires_user_confirmation=False. "
                        f"All Phase 5B candidates must require confirmation."
                    ),
                    candidate_id=cid,
                )
            )

        # E. Every candidate has risks or failure modes to monitor
        if not candidate.main_risks and not candidate.failure_modes_to_monitor:
            issues.append(
                NextActionCriticIssue(
                    check_id="CRITIC_E",
                    severity="error",
                    message=(
                        f"Candidate '{cid}' has neither main_risks nor "
                        f"failure_modes_to_monitor. At least one is required."
                    ),
                    candidate_id=cid,
                )
            )

        # H. Priority is valid
        if candidate.priority not in VALID_PRIORITIES:
            issues.append(
                NextActionCriticIssue(
                    check_id="CRITIC_H",
                    severity="error",
                    message=(
                        f"Candidate '{cid}' has invalid priority {candidate.priority!r}. "
                        f"Must be one of {sorted(VALID_PRIORITIES)}."
                    ),
                    candidate_id=cid,
                )
            )

        # I. Confidence is valid
        if candidate.confidence not in VALID_CONFIDENCES:
            issues.append(
                NextActionCriticIssue(
                    check_id="CRITIC_I",
                    severity="error",
                    message=(
                        f"Candidate '{cid}' has invalid confidence {candidate.confidence!r}. "
                        f"Must be one of {sorted(VALID_CONFIDENCES)}."
                    ),
                    candidate_id=cid,
                )
            )

    # --- Result-level checks ---

    # F. Recommended status requires at least one candidate
    if result.status == "recommended" and not result.candidates:
        issues.append(
            NextActionCriticIssue(
                check_id="CRITIC_F",
                severity="error",
                message=(
                    "status='recommended' but no candidates are present. "
                    "Either provide candidates or change status to 'needs_more_evidence'."
                ),
            )
        )

    # G. needs_more_evidence summary must explain what evidence is missing
    if result.status == "needs_more_evidence":
        summary_lower = (result.summary or "").lower()
        has_explanation = any(
            kw in summary_lower for kw in _NEEDS_MORE_EVIDENCE_KEYWORDS
        )
        if result.summary and not has_explanation:
            issues.append(
                NextActionCriticIssue(
                    check_id="CRITIC_G",
                    severity="warning",
                    message=(
                        "status='needs_more_evidence' but the summary does not "
                        "explain what evidence is missing (expected keywords: "
                        f"{sorted(_NEEDS_MORE_EVIDENCE_KEYWORDS)})."
                    ),
                )
            )

    return NextActionCriticResult(issues=issues)
