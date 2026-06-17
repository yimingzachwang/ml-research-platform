"""Memory-aware next-action planner.

Phase 5B: Memory-Aware Next-Action Planner.

Main entry point: recommend_next_actions(...)

All candidate generation is deterministic — no LLM calls.
Memory context is consumed read-only; nothing is written or executed.

Governance invariants:
  - Does not create drafts.
  - Does not execute experiments.
  - Does not approve anything.
  - Recommendations inform researcher decisions; they do not trigger actions.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.agent.memory.memory_context import ResearchMemoryContext
from src.agent.memory.memory_query import (
    ResearchMemoryQuery,
    query_from_experiment_plan,
    query_from_research_spec,
    query_from_resolved_spec,
)
from src.agent.next_actions.next_action import NextActionCandidate
from src.agent.next_actions.next_action_result import (
    NextActionRecommendationResult,
    summarise_next_action_result,
)

if TYPE_CHECKING:
    from src.agent.intent.research_spec import ResearchSpec
    from src.agent.planner.experiment_plan import ExperimentPlan
    from src.agent.resolver.resolved_spec import ResolvedResearchSpec

# ---------------------------------------------------------------------------
# Failure mode groups
# ---------------------------------------------------------------------------

_FM_SIGNAL_QUALITY: frozenset[str] = frozenset(
    {
        "poor_oos_consistency",
        "weak_ic",
        "coefficient_instability",
        "high_split_sharpe_variance",
        "catastrophic_split",
    }
)

_FM_OVERFITTING: frozenset[str] = frozenset(
    {
        "high_split_sharpe_variance",
        "coefficient_instability",
        "catastrophic_split",
    }
)

_FM_TURNOVER: frozenset[str] = frozenset({"high_turnover"})

_FM_PERFORMANCE: frozenset[str] = frozenset(
    {"negative_sharpe", "weak_sharpe", "negative_return", "severe_drawdown"}
)

_CRITICAL_FM: frozenset[str] = (
    _FM_SIGNAL_QUALITY | _FM_OVERFITTING | _FM_TURNOVER | _FM_PERFORMANCE
)

# Allocation negligible-impact detection
_ALLOCATION_KW: frozenset[str] = frozenset(
    {"allocation", "temperature", "softmax", "weighting"}
)
_NEGLIGIBLE_KW: tuple[str, ...] = (
    "negligible",
    "no improvement",
    "similar performance",
    "marginal",
    "minimal impact",
    "unchanged",
    "no significant",
    "no effect",
)

# Candidate ranking
_PRIORITY_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}
_CONFIDENCE_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _count_failure_modes(ctx: ResearchMemoryContext) -> Counter[str]:
    counts: Counter[str] = Counter()
    for ev in ctx.evidence_items:
        for fm in ev.failure_modes:
            counts[fm] += 1
    return counts


def _evidence_ids_for_modes(
    ctx: ResearchMemoryContext, modes: frozenset[str], limit: int = 3
) -> list[str]:
    return [
        ev.memory_id
        for ev in ctx.evidence_items
        if any(fm in modes for fm in ev.failure_modes)
    ][:limit]


def _find_allocation_negligible_evidence(ctx: ResearchMemoryContext) -> list[str]:
    """Return memory_ids of items that suggest allocation changes had negligible impact."""
    result: list[str] = []
    for ev in ctx.evidence_items:
        s = ev.short_summary.lower()
        has_alloc = any(kw in s for kw in _ALLOCATION_KW)
        has_neg = any(kw in s for kw in _NEGLIGIBLE_KW)
        if has_alloc and has_neg:
            result.append(ev.memory_id)
    return result


def _extract_objective(
    objective: str | None,
    research_spec: Any,
    resolved_spec: Any,
    experiment_plan: Any,
) -> str | None:
    if objective:
        return objective
    if experiment_plan is not None:
        obj = getattr(experiment_plan, "research_objective", None)
        if obj:
            return obj
    if resolved_spec is not None:
        obj = getattr(resolved_spec, "objective", None)
        if obj:
            return obj
    if research_spec is not None:
        obj = getattr(research_spec, "objective", None)
        if obj:
            return obj
    return None


def _build_query_from_inputs(
    objective: str | None,
    research_spec: Any,
    resolved_spec: Any,
    experiment_plan: Any,
) -> ResearchMemoryQuery | None:
    if experiment_plan is not None:
        return query_from_experiment_plan(experiment_plan)
    if resolved_spec is not None:
        return query_from_resolved_spec(resolved_spec)
    if research_spec is not None:
        return query_from_research_spec(research_spec)
    if objective:
        tokens = list(dict.fromkeys(re.findall(r"[a-z0-9_]+", objective.lower())))
        kq = " ".join(tokens) or None
        return ResearchMemoryQuery(
            keyword_query=kq,
            semantic_query=objective,
            source="objective",
        )
    return None


def _rank_candidates(
    candidates: list[NextActionCandidate],
) -> list[NextActionCandidate]:
    return sorted(
        candidates,
        key=lambda c: (
            _PRIORITY_ORDER.get(c.priority, 3),
            _CONFIDENCE_ORDER.get(c.confidence, 3),
            c.action_id,
        ),
    )


def _deduplicate_by_action_type(
    candidates: list[NextActionCandidate],
) -> list[NextActionCandidate]:
    seen: set[str] = set()
    result: list[NextActionCandidate] = []
    for c in candidates:
        if c.action_type not in seen:
            seen.add(c.action_type)
            result.append(c)
    return result


def _build_evidence_summary(ctx: ResearchMemoryContext) -> str:
    if ctx.is_empty():
        return "No prior research evidence available in memory."
    total = len(ctx.evidence_items)
    parts: list[str] = [f"{total} memory item(s) retrieved."]
    if ctx.experiment_summary:
        exps = ", ".join(ctx.experiment_summary[:3])
        parts.append(f"Experiments: {exps}.")
    fm = ctx.failure_mode_summary
    if fm:
        fm_parts = [f"{k} ({v}x)" for k, v in list(fm.items())[:3]]
        parts.append(f"Failure modes: {', '.join(fm_parts)}.")
    else:
        parts.append("No failure modes detected in retrieved evidence.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Candidate generators (one per rule)
# ---------------------------------------------------------------------------


def _rule_no_evidence() -> list[NextActionCandidate]:
    return [
        NextActionCandidate(
            action_id="a5b_repeat_baseline_001",
            title="Establish a baseline: run a reference experiment first",
            action_type="repeat_baseline",
            description=(
                "No prior research evidence is available in memory. "
                "Before recommending specific changes, a reference experiment "
                "should be run to establish baseline performance metrics."
            ),
            rationale=(
                "The available memory contains no prior experiment results. "
                "This is a cautious next step because any recommendation without "
                "a baseline is speculative."
            ),
            supporting_evidence_ids=[],
            main_risks=["Baseline experiment may still fail to produce useful evidence"],
            failure_modes_to_monitor=["poor_oos_consistency", "weak_ic"],
            priority="medium",
            confidence="low",
        ),
        NextActionCandidate(
            action_id="a5b_diagnostic_review_003",
            title="Collect and index available diagnostic evidence",
            action_type="diagnostic_review",
            description=(
                "Review and index available experiment artefacts. "
                "No prior memory is indexed; building the memory index first "
                "will enable evidence-backed recommendations in subsequent phases."
            ),
            rationale=(
                "The available memory is empty. "
                "This is a cautious next step because a diagnostic review "
                "establishes the research starting point before committing to "
                "specific directions."
            ),
            supporting_evidence_ids=[],
            main_risks=["No artefacts may be available to review"],
            failure_modes_to_monitor=[],
            priority="low",
            confidence="low",
        ),
    ]


def _rule_signal_quality(
    ctx: ResearchMemoryContext,
    signal_fm: set[str],
) -> list[NextActionCandidate]:
    evidence_ids = _evidence_ids_for_modes(ctx, _FM_SIGNAL_QUALITY)
    priority = (
        "high"
        if ("poor_oos_consistency" in signal_fm or "weak_ic" in signal_fm)
        else "medium"
    )
    confidence = "medium" if len(evidence_ids) >= 1 else "low"
    fm_names = ", ".join(sorted(signal_fm))
    return [
        NextActionCandidate(
            action_id="a5b_feature_ablation_001",
            title="Feature ablation to isolate signal quality issues",
            action_type="feature_ablation",
            description=(
                "Run a systematic feature ablation study to identify which features "
                "contribute positively to OOS signal quality and which degrade it. "
                "Prior evidence points to signal quality or ranking instability."
            ),
            rationale=(
                f"Prior evidence indicates failure modes: {fm_names}. "
                "The available memory suggests signal quality or ranking stability "
                "is the primary bottleneck, not portfolio construction."
            ),
            supporting_evidence_ids=evidence_ids,
            main_risks=[
                "Ablation may inadvertently remove genuinely useful features",
                "Results may be noisy if the hold-out sample is small",
            ],
            failure_modes_to_monitor=sorted(signal_fm),
            priority=priority,
            confidence=confidence,
        ),
        NextActionCandidate(
            action_id="a5b_diagnostic_review_001",
            title="Diagnostic review: IC regime and coefficient stability",
            action_type="diagnostic_review",
            description=(
                "Review IC regime decomposition and coefficient stability plots "
                "across walk-forward splits to understand where signal quality breaks down."
            ),
            rationale=(
                f"Prior evidence suggests {fm_names}. "
                "This is a cautious next step because a diagnostic review can "
                "pinpoint specific splits or feature families driving instability "
                "before committing to a full re-design."
            ),
            supporting_evidence_ids=evidence_ids,
            main_risks=[
                "Diagnostic review is advisory; requires follow-up experiment to act on findings"
            ],
            failure_modes_to_monitor=sorted(signal_fm),
            priority="medium",
            confidence=confidence,
        ),
    ]


def _rule_overfitting(
    ctx: ResearchMemoryContext,
    overfit_fm: set[str],
) -> list[NextActionCandidate]:
    evidence_ids = _evidence_ids_for_modes(ctx, _FM_OVERFITTING)
    priority = "high" if "high_split_sharpe_variance" in overfit_fm else "medium"
    confidence = "medium" if evidence_ids else "low"
    fm_names = ", ".join(sorted(overfit_fm))
    return [
        NextActionCandidate(
            action_id="a5b_validation_robustness_001",
            title="Validation robustness check across extended split windows",
            action_type="validation_robustness",
            description=(
                "Evaluate the strategy with extended or alternative walk-forward "
                "split configurations to assess consistency under different regimes."
            ),
            rationale=(
                f"Prior evidence indicates overfitting-related failure modes: {fm_names}. "
                "The available memory suggests high inter-split Sharpe variance. "
                "A validation robustness check will establish whether the degradation "
                "is systematic or split-specific."
            ),
            supporting_evidence_ids=evidence_ids,
            main_risks=[
                "Changing split windows may conflate regime effects with overfitting",
                "Longer training windows reduce out-of-sample evaluation space",
            ],
            failure_modes_to_monitor=sorted(overfit_fm),
            priority=priority,
            confidence=confidence,
        ),
        NextActionCandidate(
            action_id="a5b_model_comparison_001",
            title="Model comparison: more regularised or simpler baseline",
            action_type="model_comparison",
            description=(
                "Compare the current model against a more regularised variant "
                "or a simpler baseline to assess whether model complexity drives instability."
            ),
            rationale=(
                f"Prior evidence indicates {fm_names}. "
                "This is a cautious next step because increased regularisation "
                "or a simpler model may reduce coefficient instability without "
                "requiring feature re-engineering."
            ),
            supporting_evidence_ids=evidence_ids,
            main_risks=[
                "Simpler models may have insufficient capacity for the full feature set",
                "Regularisation sweep requires a well-defined baseline for comparison",
            ],
            failure_modes_to_monitor=["coefficient_instability", "high_split_sharpe_variance"],
            priority="medium",
            confidence=confidence,
        ),
    ]


def _rule_turnover(
    ctx: ResearchMemoryContext,
    turnover_fm: set[str],
) -> list[NextActionCandidate]:
    evidence_ids = _evidence_ids_for_modes(ctx, _FM_TURNOVER)
    confidence = "medium" if evidence_ids else "low"
    return [
        NextActionCandidate(
            action_id="a5b_cost_sensitivity_001",
            title="Cost sensitivity analysis for high-turnover regime",
            action_type="cost_sensitivity",
            description=(
                "Evaluate the strategy across a range of transaction cost assumptions. "
                "Prior evidence indicates elevated portfolio turnover."
            ),
            rationale=(
                "The available memory indicates high_turnover was flagged. "
                "This is a cautious next step because cost sensitivity analysis "
                "will clarify the net-of-cost performance boundary before further "
                "feature or model changes are pursued."
            ),
            supporting_evidence_ids=evidence_ids,
            main_risks=[
                "Cost sensitivity analysis alone does not resolve signal quality issues"
            ],
            failure_modes_to_monitor=["high_turnover"],
            priority="high",
            confidence=confidence,
        ),
    ]


def _rule_allocation_negligible(
    ctx: ResearchMemoryContext,
    alloc_ids: list[str],
    signal_active: bool,
    overfit_active: bool,
) -> list[NextActionCandidate]:
    if signal_active or overfit_active:
        # Higher-priority rules already fire; don't add a redundant diagnostic
        return []
    return [
        NextActionCandidate(
            action_id="a5b_diagnostic_review_002",
            title="Diagnostic review: focus on feature and signal quality",
            action_type="diagnostic_review",
            description=(
                "Review feature IC, signal stability, and model diagnostics. "
                "Prior evidence suggests allocation-parameter changes had negligible impact; "
                "signal quality is likely the primary bottleneck."
            ),
            rationale=(
                "The available memory indicates that allocation-parameter changes "
                "produced marginal improvements. "
                "This is a cautious next step because diagnostic review of signal quality "
                "will clarify whether a deeper re-design is warranted."
            ),
            supporting_evidence_ids=alloc_ids[:3],
            main_risks=[
                "Diagnostic review is advisory; requires follow-up experiment to act on findings"
            ],
            failure_modes_to_monitor=["weak_ic", "poor_oos_consistency"],
            priority="medium",
            confidence="low",
        ),
    ]


def _rule_strong_baseline(ctx: ResearchMemoryContext) -> list[NextActionCandidate]:
    all_ids = [ev.memory_id for ev in ctx.evidence_items][:3]
    confidence = "medium" if all_ids else "low"
    return [
        NextActionCandidate(
            action_id="a5b_validation_robustness_002",
            title="Validation robustness: stress-test the baseline",
            action_type="validation_robustness",
            description=(
                "Stress-test the strategy's walk-forward validation by varying split "
                "windows and evaluation periods. The current evidence shows no critical "
                "failure modes; confirm that this holds across alternative configurations."
            ),
            rationale=(
                "Prior evidence suggests a stable baseline with no critical failure modes. "
                "This is a cautious next step before investing in further feature or model "
                "changes."
            ),
            supporting_evidence_ids=all_ids,
            main_risks=[
                "Robustness check may reveal regime sensitivity not apparent in standard splits"
            ],
            failure_modes_to_monitor=["poor_oos_consistency", "high_split_sharpe_variance"],
            priority="medium",
            confidence=confidence,
        ),
        NextActionCandidate(
            action_id="a5b_cost_sensitivity_002",
            title="Cost sensitivity: net-of-cost performance boundary",
            action_type="cost_sensitivity",
            description=(
                "Evaluate the strategy across a range of transaction cost assumptions "
                "to determine the minimum required edge before any deployment decision."
            ),
            rationale=(
                "Prior evidence suggests a stable baseline. "
                "This is a cautious next step because cost sensitivity is a prerequisite "
                "for any future deployment consideration."
            ),
            supporting_evidence_ids=all_ids,
            main_risks=["Cost assumptions may not reflect live market conditions"],
            failure_modes_to_monitor=["high_turnover"],
            priority="medium",
            confidence=confidence,
        ),
    ]


# ---------------------------------------------------------------------------
# Main planner function
# ---------------------------------------------------------------------------


def recommend_next_actions(
    objective: str | None = None,
    research_spec: ResearchSpec | None = None,
    resolved_spec: ResolvedResearchSpec | None = None,
    experiment_plan: ExperimentPlan | None = None,
    memory_context: ResearchMemoryContext | None = None,
    max_candidates: int = 3,
    memory_base: Path | str | None = None,
) -> NextActionRecommendationResult:
    """Return a memory-aware set of recommended next research actions.

    Does not create drafts, execute experiments, or approve anything.
    All candidate generation is deterministic; no LLM calls are made.

    If memory_context is not provided, a query is built from the best
    available input (experiment_plan > resolved_spec > research_spec > objective)
    and the existing Phase 5A connector is called.  Pass memory_context
    directly to avoid filesystem access (e.g. in unit tests).
    """
    # 1. Resolve objective
    obj = _extract_objective(objective, research_spec, resolved_spec, experiment_plan)

    # 2. Build or retrieve memory context
    if memory_context is None:
        query = _build_query_from_inputs(
            objective, research_spec, resolved_spec, experiment_plan
        )
        if query is not None:
            from src.agent.memory.memory_connector import retrieve_research_memory_context

            memory_context = retrieve_research_memory_context(
                query, memory_base=memory_base
            )
        else:
            memory_context = ResearchMemoryContext(
                query=ResearchMemoryQuery(
                    keyword_query=None, semantic_query=None, source="none"
                )
            )

    # 3. Analyse memory
    failure_counter = _count_failure_modes(memory_context)
    has_evidence = not memory_context.is_empty()

    # Active failure mode sets
    signal_fm = set(failure_counter.keys()) & _FM_SIGNAL_QUALITY
    overfit_fm = set(failure_counter.keys()) & _FM_OVERFITTING
    turnover_fm = set(failure_counter.keys()) & _FM_TURNOVER
    any_critical = bool(signal_fm | overfit_fm | turnover_fm | (set(failure_counter.keys()) & _FM_PERFORMANCE))

    # 4. Generate candidates
    candidates: list[NextActionCandidate] = []
    rejected_directions: list[str] = []
    cautions: list[str] = []

    if not has_evidence:
        # Rule E: No memory evidence
        candidates = _rule_no_evidence()
        status = "needs_more_evidence"
    else:
        # Rule A: Signal quality / OOS consistency
        if signal_fm:
            candidates.extend(_rule_signal_quality(memory_context, signal_fm))

        # Rule C: High turnover
        if turnover_fm:
            candidates.extend(_rule_turnover(memory_context, turnover_fm))

        # Rule D: Overfitting / model instability
        if overfit_fm:
            candidates.extend(_rule_overfitting(memory_context, overfit_fm))

        # Rule B: Allocation negligible impact
        alloc_ids = _find_allocation_negligible_evidence(memory_context)
        if alloc_ids:
            rejected_directions.append(
                "Avoid another allocation-temperature-only sweep until signal quality "
                "issues are addressed. Prior evidence suggests allocation-parameter "
                "changes have negligible impact on strategy performance."
            )
            cautions.append(
                "Allocation-parameter tuning appears to be a secondary concern relative "
                "to signal quality and model stability."
            )
            candidates.extend(
                _rule_allocation_negligible(
                    memory_context, alloc_ids, bool(signal_fm), bool(overfit_fm)
                )
            )

        # Rule F: Strong baseline — has evidence but no critical failure modes
        if not any_critical:
            candidates.extend(_rule_strong_baseline(memory_context))

        # Cautions from specific failure modes
        if signal_fm:
            cautions.append(
                "Feature ablation should be evaluated across all walk-forward splits, "
                "not just the most recent."
            )
        if overfit_fm:
            cautions.append(
                "Increasing regularisation without addressing signal quality may be "
                "insufficient if the feature set itself is the root cause."
            )

        status = "recommended" if candidates else "needs_more_evidence"

    # 5. Deduplicate (keep first occurrence per action_type) then rank
    candidates = _deduplicate_by_action_type(candidates)
    candidates = _rank_candidates(candidates)[:max_candidates]

    top_candidate = candidates[0] if candidates else None

    # 6. Build evidence summary and result summary
    evidence_summary = _build_evidence_summary(memory_context)

    result = NextActionRecommendationResult(
        status=status,
        objective=obj,
        candidates=candidates,
        top_candidate=top_candidate,
        memory_context=memory_context,
        evidence_summary=evidence_summary,
        rejected_directions=rejected_directions,
        recommended_cautions=cautions,
        warnings=[],
        errors=[],
        summary="",  # filled below
    )
    result.summary = summarise_next_action_result(result)
    return result
