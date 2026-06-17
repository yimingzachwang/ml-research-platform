"""create_draft_from_plan — bridge from ExperimentPlan to ExperimentDraft.

Translates an approved ExperimentPlan's resolved spec into a validated,
unapproved ExperimentDraft using the existing config-generation layer.

Governance boundaries (Phase 3B):
  - Never executes experiments.
  - Never approves drafts automatically.
  - Never calls render_to_yaml() or writes to configs/experiments/.
  - Never bypasses validate_draft().
  - Hard-unsupported fields (universe, tickers, …) block draft creation.
  - Soft-unsupported fields (start_date, end_date) are reported in
    result.unsupported_fields but allow draft_created (partial draft).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.agent.drafting.draft_mapping import (
    detect_hard_unsupported_fields,
    detect_unsupported_fields,
    map_spec_to_changes,
)
from src.agent.drafting.draft_result import PlanDraftResult
from src.agent.planner.experiment_plan import ExperimentPlan
from src.experiments.config_io import load_config
from src.orchestration.config_generation.draft_generator import (
    _next_available_proposed_name,
)
from src.orchestration.config_generation.draft_schema import (
    ExperimentDraft,
    compute_draft_hash,
)
from src.orchestration.config_generation.draft_validator import validate_draft
from src.orchestration.utils.filesystem import experiment_config_path

# ---------------------------------------------------------------------------
# Preset → base experiment config name
# ---------------------------------------------------------------------------

_PRESET_TO_CONFIG_NAME: dict[str, str] = {
    "canonical_multi_asset_etf": "canonical_ml_multi_asset",
    "single_asset_ml":           "canonical_ml_showcase",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_base_experiment(plan: ExperimentPlan) -> str | None:
    """Return the base experiment config name for this plan.

    Resolution order:
      1. plan.baseline: if it is a known preset key, map it; otherwise use as-is.
      2. plan.resolved_spec.preset: map through _PRESET_TO_CONFIG_NAME.

    Returns None when no mapping can be established.
    """
    if plan.baseline:
        return _PRESET_TO_CONFIG_NAME.get(plan.baseline, plan.baseline)
    preset = plan.resolved_spec.preset
    if preset:
        return _PRESET_TO_CONFIG_NAME.get(preset)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_draft_from_plan(
    plan: ExperimentPlan,
    configs_base: Path | str | None = None,
    base: Path | str | None = None,
    reports_base: Path | str | None = None,
) -> PlanDraftResult:
    """Translate an ExperimentPlan into a validated, unapproved ExperimentDraft.

    Hard-unsupported fields (universe, tickers, frequency, data_source, asset_class)
    block draft creation and return status=unsupported_plan.

    Soft-unsupported fields (start_date, end_date) are reported in
    result.unsupported_fields but do not block draft creation — the result status
    is still draft_created, indicating a partial draft (the date constraint cannot
    be expressed as a DraftChange in Phase 3B).

    Args:
        plan:         A complete, valid ExperimentPlan produced by create_experiment_plan().
        configs_base: Override for configs/experiments/ (used in tests and staging).
        base:         Override for results/experiments/ (collision avoidance).
        reports_base: Override for reports/ (collision avoidance).

    Returns:
        PlanDraftResult with status in:
          draft_created         — draft assembled, validate_draft() passed, not approved.
                                  result.unsupported_fields may contain date fields.
          needs_base_experiment — no base experiment config identified or found on disk.
          unsupported_plan      — hard-unsupported fields present (universe, tickers, …).
          invalid_plan          — resolved spec is not complete or not valid.
          validation_failed     — draft assembled but validate_draft() rejected it.
    """
    resolved = plan.resolved_spec

    # Guard: plan must be complete and valid
    if not resolved.is_complete() or not resolved.is_valid():
        errors = list(resolved.missing_fields) + list(resolved.compatibility_errors)
        return PlanDraftResult(
            status="invalid_plan",
            plan=plan,
            errors=errors,
            summary=f"Plan is not complete/valid: {'; '.join(errors)}",
        )

    # Guard: hard-unsupported structural fields — block draft creation
    hard_unsupported = detect_hard_unsupported_fields(resolved)
    if hard_unsupported:
        all_unsupported = detect_unsupported_fields(resolved)
        return PlanDraftResult(
            status="unsupported_plan",
            plan=plan,
            unsupported_fields=all_unsupported,
            errors=[
                f"Plan requires changes to fields outside the current change-path "
                f"whitelist: {hard_unsupported}. Universe and ticker changes are deferred "
                "to a future phase."
            ],
            summary=(
                f"Blocking unsupported fields: {', '.join(hard_unsupported)}. "
                "These cannot be translated to DraftChanges in Phase 3B."
            ),
        )

    # Collect ALL unsupported fields (hard + soft) for reporting.
    # At this point hard_unsupported is empty; all_unsupported = soft only (dates).
    all_unsupported = detect_unsupported_fields(resolved)

    # Identify base experiment
    base_experiment = _resolve_base_experiment(plan)
    if not base_experiment:
        return PlanDraftResult(
            status="needs_base_experiment",
            plan=plan,
            unsupported_fields=all_unsupported,
            errors=[
                "Cannot identify a base experiment config for this plan. "
                f"Plan baseline={plan.baseline!r}, preset={resolved.preset!r}. "
                "Provide a named baseline or use a supported preset "
                f"({list(_PRESET_TO_CONFIG_NAME.keys())})."
            ],
            summary="No base experiment config identified for this plan.",
        )

    # Load base config (needed to read current_values for DraftChanges)
    config_path = experiment_config_path(base_experiment, configs_base)
    try:
        base_config = load_config(config_path)
    except FileNotFoundError:
        return PlanDraftResult(
            status="needs_base_experiment",
            plan=plan,
            base_experiment=base_experiment,
            unsupported_fields=all_unsupported,
            errors=[
                f"Base experiment config not found: {config_path}. "
                "Ensure the base experiment YAML exists before creating a draft."
            ],
            summary=(
                f"Base experiment config '{base_experiment}' not found at {config_path}."
            ),
        )

    # Map spec → DraftChanges
    changes, mapping_warnings = map_spec_to_changes(resolved, base_config)

    # Choose a proposed name that does not collide with existing artefacts
    proposed_name = _next_available_proposed_name(
        f"{base_experiment}_v2",
        base=base,
        configs_base=configs_base,
        reports_base=reports_base,
    )

    # Assemble the draft (unapproved, not yet persisted)
    draft_id = str(uuid.uuid4())
    draft_hash = compute_draft_hash(base_experiment, proposed_name, changes)
    draft = ExperimentDraft(
        draft_id=draft_id,
        draft_hash=draft_hash,
        base_experiment=base_experiment,
        source_proposal_hash=plan.plan_id,
        proposed_name=proposed_name,
        changes=changes,
        generated_at=datetime.now(UTC).isoformat(),
        approved=False,
    )

    # Validate against the authoritative schema — never bypass
    validation_result = validate_draft(draft, configs_base=configs_base)
    if not validation_result.is_valid:
        return PlanDraftResult(
            status="validation_failed",
            plan=plan,
            draft=draft,
            base_experiment=base_experiment,
            unsupported_fields=all_unsupported,
            errors=validation_result.errors,
            warnings=mapping_warnings,
            validation_result=validation_result,
            summary=(
                f"Draft validation failed for '{proposed_name}' "
                f"(base: '{base_experiment}'): "
                + "; ".join(validation_result.errors)
            ),
        )

    n = len(changes)
    date_note = (
        f" Note: {len(all_unsupported)} date field(s) not expressible as changes: "
        f"{all_unsupported}."
        if all_unsupported else ""
    )
    return PlanDraftResult(
        status="draft_created",
        plan=plan,
        draft=draft,
        base_experiment=base_experiment,
        unsupported_fields=all_unsupported,
        warnings=mapping_warnings,
        validation_result=validation_result,
        summary=(
            f"Draft '{proposed_name}' created from '{base_experiment}' "
            f"with {n} change(s). Awaiting approval.{date_note}"
        ),
    )
