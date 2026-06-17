"""Convert a NextActionCandidate into a ResearchSpecProposal (Phase 5D).

Main entry point: propose_spec_from_next_action()

All mapping is deterministic — no LLM calls.
Governance invariants (always enforced):
  - Does not create drafts.
  - Does not execute experiments.
  - Does not approve anything.
  - Every unsupported registry value is reported, never silently applied.
  - Memory recommendations do not override registry/compatibility validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from dataclasses import fields as _dfields
from typing import TYPE_CHECKING, Any

from src.agent.next_actions.next_action import NextActionCandidate
from src.agent.next_actions.spec_proposal import (
    ResearchSpecProposal,
    ResearchSpecProposalResult,
    summarise_spec_proposal,
)

if TYPE_CHECKING:
    from src.agent.registry.registry_schema import ResearchConfigRegistry
    from src.agent.resolver.resolution_result import ResearchSpecResolutionResult
    from src.agent.resolver.resolved_spec import ResolvedResearchSpec


# ---------------------------------------------------------------------------
# Internal result carrier (private)
# ---------------------------------------------------------------------------


@dataclass
class _ActionResult:
    """Outcome of one action-type handler."""

    status: str  # "ok" | "needs_user_input" | "unsupported_action"
    changes: dict[str, Any] = field(default_factory=dict)
    unsupported_changes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    message: str = ""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_proposal_id(candidate: NextActionCandidate) -> str:
    key = (
        f"{candidate.action_id}:{candidate.action_type}:"
        + json.dumps(candidate.suggested_spec_changes, sort_keys=True, default=str)
    )
    return "p5d_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def _spec_field_names() -> frozenset[str]:
    from src.agent.intent.research_spec import ResearchSpec
    return frozenset(f.name for f in _dfields(ResearchSpec))


def _spec_from_base_plus_changes(
    base: Any,  # ResearchSpec
    changes: dict[str, Any],
) -> Any:  # ResearchSpec
    """Merge base spec fields with proposed changes into a new ResearchSpec."""
    from src.agent.intent.research_spec import ResearchSpec
    merged = {**base.to_dict(), **changes}
    # Remove keys whose value is None so ResearchSpec defaults apply cleanly
    merged = {k: v for k, v in merged.items() if v is not None}
    return ResearchSpec.from_dict(merged)


def _derive_base_spec_from_resolved(resolved: Any) -> Any:  # ResearchSpec
    """Build a ResearchSpec from a ResolvedResearchSpec.

    Content fields that are empty collections are treated as 'not set'
    (mirroring ResearchSpec's convention of None == unspecified).

    Hard-unsupported structural fields (universe, tickers, asset_class,
    frequency, data_source) are excluded so that the derived spec can
    round-trip through resolve → draft without triggering unsupported_plan.
    Those fields are always re-populated from the preset by the resolver.
    """
    from src.agent.drafting.draft_mapping import _HARD_UNSUPPORTED_SPEC_FIELDS
    from src.agent.intent.research_spec import ResearchSpec
    rs_names = frozenset(f.name for f in _dfields(ResearchSpec))
    result: dict[str, Any] = {}
    for name in rs_names:
        if name in _HARD_UNSUPPORTED_SPEC_FIELDS:
            continue  # resolver re-populates these from the preset
        val = getattr(resolved, name, None)
        # Empty list or empty dict → treat as unset
        if isinstance(val, list) and len(val) == 0:
            val = None
        if isinstance(val, dict) and len(val) == 0:
            val = None
        if val is not None:
            result[name] = val
    return ResearchSpec.from_dict(result)


def _fail_result(
    status: str,
    message: str,
    unsupported: list[str] | None = None,
    warnings: list[str] | None = None,
) -> ResearchSpecProposalResult:
    return ResearchSpecProposalResult(
        status=status,  # type: ignore[arg-type]
        proposal=None,
        resolution_result=None,
        warnings=warnings or [],
        errors=[message] if message else [],
        summary=message,
    )


# ---------------------------------------------------------------------------
# Action-type handlers
# ---------------------------------------------------------------------------


def _handle_model_comparison(
    changes: dict[str, Any],
    registry: ResearchConfigRegistry,
) -> _ActionResult:
    model = changes.get("model")
    if not model:
        return _ActionResult(
            status="needs_user_input",
            message=(
                "model_comparison requires 'model' in suggested_spec_changes. "
                "Specify a registry model name."
            ),
        )
    if model not in registry.model_names():
        return _ActionResult(
            status="unsupported_action",
            unsupported_changes=[
                f"model '{model}' is not in the registry. "
                f"Available: {sorted(registry.model_names())}"
            ],
            message=f"model '{model}' is not in the registry.",
        )
    allowed = {"model", "model_params"}
    filtered: dict[str, Any] = {}
    unsupported: list[str] = []
    for key, val in changes.items():
        if key in allowed:
            filtered[key] = val
        else:
            unsupported.append(
                f"'{key}' is not a supported change for model_comparison "
                f"(allowed: {sorted(allowed)})"
            )
    return _ActionResult(status="ok", changes=filtered, unsupported_changes=unsupported)


def _handle_feature_ablation(
    changes: dict[str, Any],
    registry: ResearchConfigRegistry,
) -> _ActionResult:
    features = changes.get("features")
    families = changes.get("feature_families")

    if not features and not families:
        return _ActionResult(
            status="needs_user_input",
            message=(
                "feature_ablation requires 'features' or 'feature_families' in "
                "suggested_spec_changes. Specify the target feature list."
            ),
        )

    filtered: dict[str, Any] = {}
    unsupported: list[str] = []

    if features is not None:
        valid = [f for f in features if f in registry.feature_names()]
        invalid = [f for f in features if f not in registry.feature_names()]
        for f in invalid:
            unsupported.append(f"feature '{f}' is not in the registry")
        if valid:
            filtered["features"] = valid
        elif features:
            # All specified features were unsupported
            return _ActionResult(
                status="unsupported_action",
                unsupported_changes=unsupported,
                message="All specified features are not in the registry.",
            )

    if families is not None:
        valid_families = registry.feature_family_names()
        valid_fam = [f for f in families if f in valid_families]
        invalid_fam = [f for f in families if f not in valid_families]
        for f in invalid_fam:
            unsupported.append(
                f"feature_family '{f}' is not a known family. "
                f"Available: {sorted(valid_families)}"
            )
        if valid_fam:
            filtered["feature_families"] = valid_fam
        elif families:
            return _ActionResult(
                status="unsupported_action",
                unsupported_changes=unsupported,
                message="All specified feature families are not in the registry.",
            )

    for key in changes:
        if key not in ("features", "feature_families"):
            unsupported.append(
                f"'{key}' is not a supported change for feature_ablation "
                "(allowed: features, feature_families)"
            )

    return _ActionResult(status="ok", changes=filtered, unsupported_changes=unsupported)


def _handle_cost_sensitivity(changes: dict[str, Any]) -> _ActionResult:
    bps = changes.get("transaction_cost_bps")
    if bps is None:
        return _ActionResult(
            status="needs_user_input",
            message=(
                "cost_sensitivity requires 'transaction_cost_bps' in suggested_spec_changes."
            ),
        )
    try:
        bps_float = float(bps)
    except (TypeError, ValueError):
        return _ActionResult(
            status="unsupported_action",
            unsupported_changes=[
                f"transaction_cost_bps must be numeric; got {bps!r}"
            ],
            message="transaction_cost_bps must be a non-negative number.",
        )
    if bps_float < 0:
        return _ActionResult(
            status="unsupported_action",
            unsupported_changes=[
                f"transaction_cost_bps={bps_float} is negative; must be ≥ 0"
            ],
            message="transaction_cost_bps must be non-negative.",
        )

    filtered: dict[str, Any] = {"transaction_cost_bps": bps_float}
    unsupported: list[str] = []
    for key in changes:
        if key != "transaction_cost_bps":
            unsupported.append(
                f"'{key}' is not a supported change for cost_sensitivity"
            )
    return _ActionResult(status="ok", changes=filtered, unsupported_changes=unsupported)


def _handle_validation_robustness(
    changes: dict[str, Any],
    registry: ResearchConfigRegistry,
) -> _ActionResult:
    allowed = {"validation_protocol", "train_window", "test_window"}
    explicit = {k: v for k, v in changes.items() if k in allowed and v is not None}

    if not explicit:
        return _ActionResult(
            status="needs_user_input",
            message=(
                "validation_robustness requires explicit train_window, test_window, "
                "or validation_protocol in suggested_spec_changes."
            ),
        )

    filtered: dict[str, Any] = {}
    unsupported: list[str] = []

    if "validation_protocol" in explicit:
        vp = explicit["validation_protocol"]
        if vp not in registry.validation_names():
            unsupported.append(
                f"validation_protocol '{vp}' is not in the registry. "
                f"Available: {sorted(registry.validation_names())}"
            )
        else:
            filtered["validation_protocol"] = vp

    if "train_window" in explicit:
        filtered["train_window"] = explicit["train_window"]

    if "test_window" in explicit:
        filtered["test_window"] = explicit["test_window"]

    for key in changes:
        if key not in allowed:
            unsupported.append(
                f"'{key}' is not a supported change for validation_robustness "
                f"(allowed: {sorted(allowed)})"
            )

    if not filtered:
        return _ActionResult(
            status="unsupported_action",
            unsupported_changes=unsupported,
            message="All requested validation_robustness changes are unsupported.",
        )

    return _ActionResult(status="ok", changes=filtered, unsupported_changes=unsupported)


def _handle_portfolio_construction_test(
    changes: dict[str, Any],
    registry: ResearchConfigRegistry,
) -> _ActionResult:
    """Portfolio construction fields expressible in ResearchSpec."""
    allowed_in_spec = {"signal_params", "risk_constraints"}
    filtered: dict[str, Any] = {}
    unsupported: list[str] = []

    for key, val in changes.items():
        if key in allowed_in_spec:
            filtered[key] = val
        else:
            unsupported.append(
                f"'{key}' is not expressible in ResearchSpec for portfolio_construction_test. "
                f"Supported: {sorted(allowed_in_spec)}"
            )

    if not filtered:
        return _ActionResult(
            status="unsupported_action",
            unsupported_changes=unsupported,
            message=(
                "portfolio_construction_test changes are not currently expressible "
                "in ResearchSpec. Portfolio construction parameters beyond "
                "signal_params and risk_constraints require direct config modification."
            ),
        )
    return _ActionResult(status="ok", changes=filtered, unsupported_changes=unsupported)


def _handle_universe_review(
    changes: dict[str, Any],
    registry: ResearchConfigRegistry,
) -> _ActionResult:
    allowed = {"universe", "asset_class", "tickers"}
    filtered: dict[str, Any] = {}
    unsupported: list[str] = []

    universe = changes.get("universe")
    if universe is not None:
        if universe not in registry.universe_names():
            unsupported.append(
                f"universe '{universe}' is not in the registry. "
                f"Available: {sorted(registry.universe_names())}"
            )
        else:
            filtered["universe"] = universe

    asset_class = changes.get("asset_class")
    if asset_class is not None:
        if asset_class not in registry.asset_class_names():
            unsupported.append(
                f"asset_class '{asset_class}' is not in the registry. "
                f"Available: {sorted(registry.asset_class_names())}"
            )
        else:
            filtered["asset_class"] = asset_class

    if "tickers" in changes:
        # Tickers are not validated against registry (user-defined)
        filtered["tickers"] = changes["tickers"]

    for key in changes:
        if key not in allowed:
            unsupported.append(
                f"'{key}' is not a supported change for universe_review "
                f"(allowed: {sorted(allowed)})"
            )

    if not filtered:
        return _ActionResult(
            status="needs_user_input",
            unsupported_changes=unsupported,
            message=(
                "universe_review requires 'universe' or 'asset_class' in "
                "suggested_spec_changes."
            ),
        )
    return _ActionResult(status="ok", changes=filtered, unsupported_changes=unsupported)


def _handle_repeat_baseline(base_spec: Any) -> _ActionResult:
    if base_spec is None:
        return _ActionResult(
            status="needs_user_input",
            message=(
                "repeat_baseline requires a base_spec or resolved_spec to determine "
                "what to repeat. Provide the spec you want to replicate."
            ),
        )
    return _ActionResult(
        status="ok",
        changes={},
        warnings=[
            "repeat_baseline produces no spec changes; the proposed_spec is "
            "identical to the base spec."
        ],
    )


def _handle_custom(
    changes: dict[str, Any],
    registry: ResearchConfigRegistry,
) -> _ActionResult:
    if not changes:
        return _ActionResult(
            status="needs_user_input",
            message=(
                "custom action_type requires suggested_spec_changes to contain "
                "the intended specification changes."
            ),
        )

    spec_names = _spec_field_names()
    filtered: dict[str, Any] = {}
    unsupported: list[str] = []

    for key, val in changes.items():
        if key not in spec_names:
            unsupported.append(f"'{key}' is not a valid ResearchSpec field")
            continue
        # Per-field registry validation
        if key == "model" and val is not None:
            if val not in registry.model_names():
                unsupported.append(
                    f"model '{val}' is not in the registry. "
                    f"Available: {sorted(registry.model_names())}"
                )
                continue
        elif key == "features" and isinstance(val, list):
            valid = [f for f in val if f in registry.feature_names()]
            invalid = [f for f in val if f not in registry.feature_names()]
            for f in invalid:
                unsupported.append(f"feature '{f}' is not in the registry")
            if valid:
                filtered["features"] = valid
            continue
        elif key == "validation_protocol" and val is not None:
            if val not in registry.validation_names():
                unsupported.append(
                    f"validation_protocol '{val}' is not in the registry. "
                    f"Available: {sorted(registry.validation_names())}"
                )
                continue
        elif key == "universe" and val is not None:
            if val not in registry.universe_names():
                unsupported.append(
                    f"universe '{val}' is not in the registry."
                )
                continue
        elif key == "asset_class" and val is not None:
            if val not in registry.asset_class_names():
                unsupported.append(
                    f"asset_class '{val}' is not in the registry."
                )
                continue
        filtered[key] = val

    if not filtered:
        return _ActionResult(
            status="unsupported_action",
            unsupported_changes=unsupported,
            message="All requested changes for custom action are unsupported or invalid.",
        )
    return _ActionResult(status="ok", changes=filtered, unsupported_changes=unsupported)


# ---------------------------------------------------------------------------
# Action dispatcher
# ---------------------------------------------------------------------------


def _dispatch_action(
    candidate: NextActionCandidate,
    base_spec: Any,
    registry: ResearchConfigRegistry,
) -> _ActionResult:
    changes = dict(candidate.suggested_spec_changes)
    at = candidate.action_type

    if at == "model_comparison":
        return _handle_model_comparison(changes, registry)
    if at == "feature_ablation":
        return _handle_feature_ablation(changes, registry)
    if at == "cost_sensitivity":
        return _handle_cost_sensitivity(changes)
    if at == "validation_robustness":
        return _handle_validation_robustness(changes, registry)
    if at == "portfolio_construction_test":
        return _handle_portfolio_construction_test(changes, registry)
    if at == "universe_review":
        return _handle_universe_review(changes, registry)
    if at == "repeat_baseline":
        return _handle_repeat_baseline(base_spec)
    if at == "custom":
        return _handle_custom(changes, registry)
    if at in ("diagnostic_review",):
        return _ActionResult(
            status="unsupported_action",
            message=(
                "diagnostic_review is an analysis action, not an experiment spec change. "
                "No ResearchSpec proposal is generated. "
                "Review available artefacts, then consider a feature_ablation or "
                "model_comparison experiment."
            ),
        )
    if at in ("stop_or_deprioritise", "needs_more_evidence"):
        return _ActionResult(
            status="unsupported_action",
            message=(
                f"{at} does not produce a ResearchSpec proposal. "
                "This is an advisory recommendation only."
            ),
        )
    # Unknown action type
    return _ActionResult(
        status="unsupported_action",
        message=f"Unknown action_type '{at}' cannot be mapped to a ResearchSpec.",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def propose_spec_from_next_action(
    candidate: NextActionCandidate,
    base_spec: Any = None,   # ResearchSpec | None
    resolved_spec: Any = None,  # ResolvedResearchSpec | None
    registry: ResearchConfigRegistry | None = None,
    mode: str = "recommended",
) -> ResearchSpecProposalResult:
    """Convert a NextActionCandidate into a ResearchSpecProposal.

    Args:
        candidate:     The candidate from Phase 5B recommend_next_actions().
        base_spec:     Optional base ResearchSpec to modify. Takes priority
                       over resolved_spec.
        resolved_spec: Optional ResolvedResearchSpec; used as base when
                       base_spec is not provided.
        registry:      Pre-loaded registry. Loaded from disk if None.
        mode:          Resolution mode ('recommended' | 'strict' | 'demo').

    Returns:
        ResearchSpecProposalResult with status:
            proposal_created  — proposed spec resolves cleanly
            needs_user_input  — more information required (missing fields)
            unsupported_action — action type cannot be mapped to a spec change
            invalid_candidate  — candidate is structurally invalid
            resolution_failed  — proposed spec failed compatibility validation

    Governance:
        approval_required          = True   always
        execution_performed        = False  always
        creates_draft              = False  always
        requires_user_confirmation = True   always
        Registry validation is always applied; unsupported values are never
        silently accepted into the proposed spec.
    """
    from src.agent.intent.research_spec import ResearchSpec
    from src.agent.registry.registry_loader import load_registry
    from src.agent.resolver.resolve_research_spec import resolve_research_spec

    # 1. Load registry
    if registry is None:
        registry = load_registry()

    # 2. Validate candidate basics
    if not candidate.action_id or not candidate.action_type:
        return _fail_result(
            "invalid_candidate",
            "Candidate has an empty action_id or action_type.",
        )

    # 3. Determine effective base spec
    effective_base: ResearchSpec | None = base_spec
    if effective_base is None and resolved_spec is not None:
        effective_base = _derive_base_spec_from_resolved(resolved_spec)

    # 4. Dispatch to action-type handler
    action_result = _dispatch_action(candidate, effective_base, registry)

    # 5. Handle handler-level failures (no spec built yet)
    if action_result.status == "needs_user_input":
        return ResearchSpecProposalResult(
            status="needs_user_input",
            proposal=None,
            resolution_result=None,
            warnings=action_result.warnings,
            errors=[action_result.message] if action_result.message else [],
            summary=(
                f"Phase 5D Spec Proposal [needs_user_input]. "
                f"{action_result.message} "
                "User confirmation required before creating any experiment spec. "
                "No draft created. No approval. No execution performed."
            ),
        )

    if action_result.status == "unsupported_action":
        return ResearchSpecProposalResult(
            status="unsupported_action",
            proposal=None,
            resolution_result=None,
            warnings=list(action_result.warnings) + list(action_result.unsupported_changes),
            errors=[action_result.message] if action_result.message else [],
            summary=(
                f"Phase 5D Spec Proposal [unsupported_action]. "
                f"{action_result.message} "
                "User confirmation required before creating any experiment spec. "
                "No draft created. No approval. No execution performed."
            ),
        )

    # 6. Build proposed spec
    if effective_base is not None:
        proposed_spec = _spec_from_base_plus_changes(effective_base, action_result.changes)
    elif action_result.changes:
        proposed_spec = ResearchSpec.from_dict(action_result.changes)
    else:
        # No base spec and no changes — can't build anything
        return ResearchSpecProposalResult(
            status="needs_user_input",
            proposal=None,
            resolution_result=None,
            warnings=[],
            errors=["No base_spec or resolved_spec provided and no changes available."],
            summary=(
                "Phase 5D Spec Proposal [needs_user_input]. "
                "Provide a base_spec or resolved_spec to use as the starting point. "
                "User confirmation required before creating any experiment spec. "
                "No draft created. No approval. No execution performed."
            ),
        )

    # 7. Run resolver on proposed spec
    try:
        resolution_result = resolve_research_spec(
            proposed_spec, registry=registry, mode=mode
        )
    except Exception as exc:
        return _fail_result(
            "resolution_failed",
            f"Resolution raised an unexpected error: {exc}",
        )

    # 8. Determine proposal status from resolution outcome
    if resolution_result.status == "needs_user_input":
        proposal_status: str = "needs_user_input"
    elif resolution_result.status == "invalid":
        proposal_status = "resolution_failed"
    else:
        proposal_status = "proposal_created"

    # 9. Build provenance diffs
    if effective_base is not None:
        base_non_none = {
            k: v for k, v in effective_base.to_dict().items() if v is not None
        }
        unchanged_fields = {
            k: v
            for k, v in base_non_none.items()
            if k not in action_result.changes
        }
    else:
        unchanged_fields = {}

    # 10. Assemble proposal
    proposal = ResearchSpecProposal(
        proposal_id=_make_proposal_id(candidate),
        source_candidate_id=candidate.action_id,
        source_action_type=candidate.action_type,
        base_spec=effective_base,
        proposed_spec=proposed_spec,
        proposed_changes=dict(action_result.changes),
        unchanged_fields=unchanged_fields,
        rationale=candidate.rationale,
        supporting_evidence_ids=list(candidate.supporting_evidence_ids),
        warnings=list(action_result.warnings),
        unsupported_changes=list(action_result.unsupported_changes),
        resolution_status=resolution_result.status,
        resolution_summary=resolution_result.summary(),
    )

    all_warnings = (
        list(action_result.warnings)
        + list(resolution_result.warnings)
    )

    result = ResearchSpecProposalResult(
        status=proposal_status,  # type: ignore[arg-type]
        proposal=proposal,
        resolution_result=resolution_result,
        warnings=all_warnings,
        errors=list(resolution_result.errors),
        summary="",
    )
    result.summary = summarise_spec_proposal(result)
    return result
