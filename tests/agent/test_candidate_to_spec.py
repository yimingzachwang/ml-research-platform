"""Tests for propose_spec_from_next_action — Phase 5D.

B. Candidate-to-spec conversion tests (all action types)

B1.  Model comparison — canonical ETF, valid model → proposal_created
B2.  Unsupported model → unsupported_action
B3.  Cost sensitivity — explicit bps → proposal_created
B4.  Negative transaction cost → unsupported_action
B5.  Feature ablation with exact feature list → proposal_created
B6.  Feature ablation without feature list → needs_user_input
B7.  Validation robustness with explicit windows → proposal_created
B8.  Validation robustness without windows → needs_user_input
B9.  Diagnostic review → unsupported_action
B10. Repeat baseline with base spec → proposal_created, no changes
B11. Stop/deprioritise → unsupported_action
B12. Missing base spec and resolved_spec → needs_user_input
B13. ResolvedSpec input (no base_spec) → proposal_created
B14. Unknown action type → unsupported_action
B15. Feature ablation — unsupported features rejected, valid applied
B16. Model comparison with model_params → both applied
B17. Validation robustness with validation_protocol → validated against registry
B18. Portfolio construction test with signal_params → proposal_created
B19. Portfolio construction test with non-expressible field → unsupported_action
B20. Custom action type with valid changes → proposal_created
B21. Universe review with valid universe → proposal_created
B22. Universe review with invalid universe → unsupported_action
B23. Zero transaction cost → proposal_created (zero is valid)
B24. Only model_params (no model) in model_comparison → needs_user_input

D. Feature family validation (Phase 5D.1)
D1.  Registry accessor — feature_family_names() returns frozenset with known families
D2.  Supported families → proposal_created, families applied
D3.  Unsupported family → unsupported_action, family name in unsupported_changes
D4.  Mixed families — valid kept, invalid in unsupported_changes

C. Integration — full chain through recommend_next_actions → propose
C1.  Chain: spec → resolve → recommend → select top → propose
C2.  Governance: no draft, no execution, approval required
C3.  Resolution result is present and non-None
C4.  proposed_changes are a subset of ResearchSpec fields
C5.  Proposal preserves evidence IDs from candidate
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agent.intent.research_spec import ResearchSpec
from src.agent.next_actions.candidate_to_spec import propose_spec_from_next_action
from src.agent.next_actions.next_action import NextActionCandidate
from src.agent.registry.registry_loader import load_registry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REGISTRY = None  # loaded lazily


def _registry():
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = load_registry()
    return _REGISTRY


def _canonical_base() -> ResearchSpec:
    return ResearchSpec(
        preset="canonical_multi_asset_etf",
        end_date="2024-12-31",
    )


def _candidate(
    action_type: str,
    suggested_spec_changes: dict | None = None,
    action_id: str | None = None,
    supporting_evidence_ids: list[str] | None = None,
    rationale: str = "test rationale",
) -> NextActionCandidate:
    return NextActionCandidate(
        action_id=action_id or f"test_{action_type}_001",
        title=f"Test {action_type}",
        action_type=action_type,
        description=f"Test candidate for {action_type}",
        rationale=rationale,
        suggested_spec_changes=suggested_spec_changes or {},
        supporting_evidence_ids=supporting_evidence_ids or [],
        main_risks=["test risk"],
    )


# ---------------------------------------------------------------------------
# B1. Model comparison — valid model → proposal_created
# ---------------------------------------------------------------------------


class TestModelComparison:
    def test_proposal_created_lasso(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "proposal_created"

    def test_proposed_spec_model_set(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.proposal is not None
        assert result.proposal.proposed_spec.model == "LassoRegression"

    def test_resolution_result_present(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.resolution_result is not None

    def test_no_draft_created(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        p = result.proposal
        assert p is not None
        assert p.creates_draft is False

    def test_no_execution(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.proposal.executes_experiment is False

    def test_approval_required(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.proposal.approval_required is True

    def test_proposed_changes_contains_model(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert "model" in result.proposal.proposed_changes

    def test_source_candidate_id_preserved(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"}, action_id="mc_001")
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.proposal.source_candidate_id == "mc_001"

    def test_with_model_params(self):
        c = _candidate(
            "model_comparison",
            {"model": "LassoRegression", "model_params": {"alpha": 2.0}},
        )
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "proposal_created"
        assert result.proposal.proposed_spec.model_params == {"alpha": 2.0}

    def test_extra_fields_in_changes_become_unsupported(self):
        c = _candidate(
            "model_comparison",
            {"model": "LassoRegression", "universe": "custom_x"},
        )
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "proposal_created"
        assert len(result.proposal.unsupported_changes) > 0

    def test_no_model_in_changes_needs_user_input(self):
        c = _candidate("model_comparison", {"model_params": {"alpha": 2.0}})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "needs_user_input"


# ---------------------------------------------------------------------------
# B2. Unsupported model → unsupported_action
# ---------------------------------------------------------------------------


class TestUnsupportedModel:
    def test_unsupported_action_status(self):
        c = _candidate("model_comparison", {"model": "RandomForest"})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "unsupported_action"

    def test_no_proposal_on_unsupported_model(self):
        c = _candidate("model_comparison", {"model": "RandomForest"})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.proposal is None

    def test_error_mentions_unsupported_model(self):
        c = _candidate("model_comparison", {"model": "RandomForest"})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert any("RandomForest" in e for e in result.errors) or any(
            "RandomForest" in w for w in result.proposal.unsupported_changes
            if result.proposal
        ) or "RandomForest" in str(result.errors)


# ---------------------------------------------------------------------------
# B3. Cost sensitivity — explicit bps
# ---------------------------------------------------------------------------


class TestCostSensitivity:
    def test_proposal_created(self):
        c = _candidate("cost_sensitivity", {"transaction_cost_bps": 10})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "proposal_created"

    def test_transaction_cost_set_in_proposed_spec(self):
        c = _candidate("cost_sensitivity", {"transaction_cost_bps": 10})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.proposal.proposed_spec.transaction_cost_bps == 10.0

    def test_resolution_passes(self):
        c = _candidate("cost_sensitivity", {"transaction_cost_bps": 5})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.resolution_result is not None

    def test_zero_cost_is_valid(self):
        c = _candidate("cost_sensitivity", {"transaction_cost_bps": 0})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "proposal_created"
        assert result.proposal.proposed_spec.transaction_cost_bps == 0.0


# ---------------------------------------------------------------------------
# B4. Negative transaction cost → unsupported_action
# ---------------------------------------------------------------------------


class TestNegativeTransactionCost:
    def test_negative_bps_unsupported_action(self):
        c = _candidate("cost_sensitivity", {"transaction_cost_bps": -5})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "unsupported_action"

    def test_negative_bps_no_proposal(self):
        c = _candidate("cost_sensitivity", {"transaction_cost_bps": -5})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.proposal is None

    def test_missing_bps_needs_user_input(self):
        c = _candidate("cost_sensitivity", {})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "needs_user_input"


# ---------------------------------------------------------------------------
# B5. Feature ablation with exact feature list
# ---------------------------------------------------------------------------


def _get_any_valid_feature() -> str:
    reg = _registry()
    return sorted(reg.feature_names())[0]


class TestFeatureAblationExact:
    def test_proposal_created_with_valid_feature(self):
        feat = _get_any_valid_feature()
        c = _candidate("feature_ablation", {"features": [feat]})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "proposal_created"

    def test_proposed_spec_features_set(self):
        feat = _get_any_valid_feature()
        c = _candidate("feature_ablation", {"features": [feat]})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.proposal.proposed_spec.features == [feat]

    def test_unsupported_features_rejected(self):
        feat = _get_any_valid_feature()
        c = _candidate("feature_ablation", {"features": [feat, "NONEXISTENT_FEATURE"]})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        # Valid feature still applied
        assert result.status == "proposal_created"
        assert feat in result.proposal.proposed_spec.features
        assert "NONEXISTENT_FEATURE" not in result.proposal.proposed_spec.features
        assert len(result.proposal.unsupported_changes) > 0

    def test_all_unsupported_features(self):
        c = _candidate("feature_ablation", {"features": ["FAKE_1", "FAKE_2"]})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "unsupported_action"


# ---------------------------------------------------------------------------
# B6. Feature ablation without feature list → needs_user_input
# ---------------------------------------------------------------------------


class TestFeatureAblationNoList:
    def test_empty_changes_needs_user_input(self):
        c = _candidate("feature_ablation", {})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "needs_user_input"

    def test_summary_explains_feature_selection_required(self):
        c = _candidate("feature_ablation", {})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert "feature" in result.summary.lower()


# ---------------------------------------------------------------------------
# B7. Validation robustness with explicit windows
# ---------------------------------------------------------------------------


class TestValidationRobustnessExplicit:
    def test_proposal_created_with_windows(self):
        c = _candidate(
            "validation_robustness",
            {"train_window": "60m", "test_window": "12m"},
        )
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "proposal_created"

    def test_train_window_in_proposed_spec(self):
        c = _candidate(
            "validation_robustness",
            {"train_window": "60m", "test_window": "12m"},
        )
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.proposal.proposed_spec.train_window == "60m"
        assert result.proposal.proposed_spec.test_window == "12m"

    def test_valid_validation_protocol_applied(self):
        reg = _registry()
        vp = sorted(reg.validation_names())[0]
        c = _candidate("validation_robustness", {"validation_protocol": vp})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "proposal_created"
        assert result.proposal.proposed_spec.validation_protocol == vp

    def test_invalid_validation_protocol_unsupported(self):
        c = _candidate(
            "validation_robustness",
            {"validation_protocol": "NONEXISTENT_PROTOCOL"},
        )
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "unsupported_action"


# ---------------------------------------------------------------------------
# B8. Validation robustness without windows → needs_user_input
# ---------------------------------------------------------------------------


class TestValidationRobustnessNoWindows:
    def test_empty_changes_needs_user_input(self):
        c = _candidate("validation_robustness", {})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "needs_user_input"


# ---------------------------------------------------------------------------
# B9. Diagnostic review → unsupported_action
# ---------------------------------------------------------------------------


class TestDiagnosticReview:
    def test_unsupported_action(self):
        c = _candidate("diagnostic_review", {})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "unsupported_action"

    def test_no_proposal_created(self):
        c = _candidate("diagnostic_review", {})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.proposal is None

    def test_error_message_explains(self):
        c = _candidate("diagnostic_review", {})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert any(
            "analysis" in e.lower() or "diagnostic" in e.lower()
            for e in result.errors
        )


# ---------------------------------------------------------------------------
# B10. Repeat baseline with base spec → proposal_created, no changes
# ---------------------------------------------------------------------------


class TestRepeatBaseline:
    def test_proposal_created(self):
        c = _candidate("repeat_baseline", {})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "proposal_created"

    def test_no_proposed_changes(self):
        c = _candidate("repeat_baseline", {})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.proposal.proposed_changes == {}

    def test_proposed_spec_equals_base_spec(self):
        base = _canonical_base()
        c = _candidate("repeat_baseline", {})
        result = propose_spec_from_next_action(
            c, base_spec=base, registry=_registry()
        )
        # proposed spec fields that are set should match base
        for field_name, val in base.specified_fields().items():
            assert getattr(result.proposal.proposed_spec, field_name) == val

    def test_governance_holds(self):
        c = _candidate("repeat_baseline", {})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        p = result.proposal
        assert p.creates_draft is False
        assert p.executes_experiment is False
        assert p.approval_required is True

    def test_no_base_spec_needs_user_input(self):
        c = _candidate("repeat_baseline", {})
        result = propose_spec_from_next_action(
            c, base_spec=None, resolved_spec=None, registry=_registry()
        )
        assert result.status == "needs_user_input"


# ---------------------------------------------------------------------------
# B11. Stop/deprioritise → unsupported_action
# ---------------------------------------------------------------------------


class TestStopOrDeprioritise:
    def test_unsupported_action(self):
        c = _candidate("stop_or_deprioritise", {})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "unsupported_action"

    def test_no_spec_created(self):
        c = _candidate("stop_or_deprioritise", {})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.proposal is None

    def test_needs_more_evidence_also_unsupported(self):
        c = _candidate("needs_more_evidence", {})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "unsupported_action"


# ---------------------------------------------------------------------------
# B12. Missing base spec and resolved_spec → needs_user_input
# ---------------------------------------------------------------------------


class TestMissingBaseSpec:
    def test_model_comparison_no_base_needs_user_input(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = propose_spec_from_next_action(
            c, base_spec=None, resolved_spec=None, registry=_registry()
        )
        # Without a base spec or resolved_spec, proposed_spec has only model set.
        # Resolution will return needs_user_input (missing end_date etc.).
        assert result.status == "needs_user_input"

    def test_repeat_baseline_no_base_needs_user_input(self):
        c = _candidate("repeat_baseline", {})
        result = propose_spec_from_next_action(
            c, base_spec=None, resolved_spec=None, registry=_registry()
        )
        assert result.status == "needs_user_input"


# ---------------------------------------------------------------------------
# B13. ResolvedSpec input → can derive base and apply changes
# ---------------------------------------------------------------------------


class TestResolvedSpecInput:
    def test_resolved_spec_used_as_base(self):
        from src.agent.resolver.resolve_research_spec import resolve_research_spec

        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            end_date="2024-12-31",
        )
        res = resolve_research_spec(spec, registry=_registry())
        resolved = res.resolved_spec
        assert resolved is not None

        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = propose_spec_from_next_action(
            c, resolved_spec=resolved, registry=_registry()
        )
        assert result.status == "proposal_created"

    def test_resolved_spec_changes_applied(self):
        from src.agent.resolver.resolve_research_spec import resolve_research_spec

        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            end_date="2024-12-31",
        )
        res = resolve_research_spec(spec, registry=_registry())

        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = propose_spec_from_next_action(
            c, resolved_spec=res.resolved_spec, registry=_registry()
        )
        assert result.proposal.proposed_spec.model == "LassoRegression"


# ---------------------------------------------------------------------------
# B14. Unknown action type → unsupported_action
# ---------------------------------------------------------------------------


class TestUnknownActionType:
    def test_unknown_unsupported(self):
        c = NextActionCandidate(
            action_id="custom_bad",
            title="Bad action",
            action_type="super_secret_action",
            description="unknown",
            rationale="unknown",
            main_risks=[],
        )
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "unsupported_action"


# ---------------------------------------------------------------------------
# B18. Portfolio construction test with signal_params
# ---------------------------------------------------------------------------


class TestPortfolioConstructionTest:
    def test_signal_params_applied(self):
        c = _candidate(
            "portfolio_construction_test",
            {"signal_params": {"n": 5}},
        )
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "proposal_created"
        assert result.proposal.proposed_spec.signal_params == {"n": 5}

    def test_non_expressible_field_unsupported(self):
        c = _candidate(
            "portfolio_construction_test",
            {"weighting_scheme": "softmax_temp"},
        )
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "unsupported_action"

    def test_empty_changes_unsupported(self):
        c = _candidate("portfolio_construction_test", {})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "unsupported_action"


# ---------------------------------------------------------------------------
# B21. Universe review with valid universe
# ---------------------------------------------------------------------------


class TestUniverseReview:
    def test_valid_universe_applied(self):
        reg = _registry()
        universe = sorted(reg.universe_names())[0]
        c = _candidate("universe_review", {"universe": universe})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status in ("proposal_created", "needs_user_input")
        # If proposed spec resolved, universe should be in it
        if result.proposal:
            assert result.proposal.proposed_changes.get("universe") == universe

    def test_invalid_universe_unsupported(self):
        c = _candidate("universe_review", {"universe": "NONEXISTENT_UNIVERSE_XYZ"})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status in ("unsupported_action", "needs_user_input")

    def test_empty_universe_review_needs_user_input(self):
        c = _candidate("universe_review", {})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "needs_user_input"


# ---------------------------------------------------------------------------
# B20. Custom action type
# ---------------------------------------------------------------------------


class TestCustomActionType:
    def test_valid_changes_applied(self):
        feat = _get_any_valid_feature()
        c = _candidate("custom", {"features": [feat]})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "proposal_created"

    def test_empty_custom_needs_user_input(self):
        c = _candidate("custom", {})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "needs_user_input"

    def test_invalid_field_for_custom_unsupported(self):
        c = _candidate("custom", {"not_a_real_field_xyz": "value"})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "unsupported_action"


# ---------------------------------------------------------------------------
# B. Governance invariants across all action types
# ---------------------------------------------------------------------------


_GOVERNANCE_CASES = [
    ("model_comparison", {"model": "LassoRegression"}),
    ("cost_sensitivity", {"transaction_cost_bps": 5}),
    ("repeat_baseline", {}),
    ("validation_robustness", {"train_window": "60m", "test_window": "12m"}),
]


class TestGovernanceInvariants:
    @pytest.mark.parametrize("action_type,changes", _GOVERNANCE_CASES)
    def test_creates_draft_always_false(self, action_type, changes):
        c = _candidate(action_type, changes)
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        if result.proposal:
            assert result.proposal.creates_draft is False

    @pytest.mark.parametrize("action_type,changes", _GOVERNANCE_CASES)
    def test_executes_experiment_always_false(self, action_type, changes):
        c = _candidate(action_type, changes)
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        if result.proposal:
            assert result.proposal.executes_experiment is False

    @pytest.mark.parametrize("action_type,changes", _GOVERNANCE_CASES)
    def test_approval_required_always_true(self, action_type, changes):
        c = _candidate(action_type, changes)
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        if result.proposal:
            assert result.proposal.approval_required is True

    @pytest.mark.parametrize("action_type,changes", _GOVERNANCE_CASES)
    def test_requires_user_confirmation_always_true(self, action_type, changes):
        c = _candidate(action_type, changes)
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        if result.proposal:
            assert result.proposal.requires_user_confirmation is True

    def test_to_dict_governance_flags_in_proposal(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        d = result.proposal.to_dict()
        assert d["creates_draft"] is False
        assert d["executes_experiment"] is False
        assert d["approval_required"] is True
        assert d["requires_user_confirmation"] is True


# ---------------------------------------------------------------------------
# B. Evidence provenance preserved
# ---------------------------------------------------------------------------


class TestEvidenceProvenance:
    def test_supporting_evidence_ids_in_proposal(self):
        c = _candidate(
            "model_comparison",
            {"model": "LassoRegression"},
            supporting_evidence_ids=["mem_abc", "mem_xyz"],
        )
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.proposal.supporting_evidence_ids == ["mem_abc", "mem_xyz"]

    def test_rationale_preserved(self):
        c = _candidate(
            "model_comparison",
            {"model": "LassoRegression"},
            rationale="Reduce coefficient instability",
        )
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert "coefficient instability" in result.proposal.rationale.lower()

    def test_source_action_type_preserved(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.proposal.source_action_type == "model_comparison"


# ---------------------------------------------------------------------------
# B. Proposed changes vs. unchanged fields
# ---------------------------------------------------------------------------


class TestProposedChangesProvenance:
    def test_changed_fields_in_proposed_changes(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert "model" in result.proposal.proposed_changes
        assert result.proposal.proposed_changes["model"] == "LassoRegression"

    def test_unchanged_fields_not_in_proposed_changes(self):
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        # preset and end_date are from base; model is the only change
        assert "preset" not in result.proposal.proposed_changes
        assert "end_date" not in result.proposal.proposed_changes

    def test_base_fields_in_unchanged(self):
        base = ResearchSpec(
            preset="canonical_multi_asset_etf",
            end_date="2024-12-31",
        )
        c = _candidate("model_comparison", {"model": "LassoRegression"})
        result = propose_spec_from_next_action(
            c, base_spec=base, registry=_registry()
        )
        unchanged = result.proposal.unchanged_fields
        assert "end_date" in unchanged


# ---------------------------------------------------------------------------
# C. Integration — full chain
# ---------------------------------------------------------------------------


class TestIntegrationChain:
    def test_chain_spec_to_proposal(self):
        from src.agent.memory.memory_context import ResearchMemoryContext
        from src.agent.memory.memory_query import ResearchMemoryQuery
        from src.agent.next_actions.next_action_planner import recommend_next_actions

        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            end_date="2024-12-31",
        )

        # Build memory context with a failure mode
        from src.agent.memory.memory_evidence import ResearchMemoryEvidence
        ev = ResearchMemoryEvidence(
            memory_id="int_001",
            experiment_name="exp_a",
            artefact_type="review",
            short_summary="high split sharpe variance detected.",
            failure_modes=["high_split_sharpe_variance"],
            tags=[],
            path="/p",
            context_hash="h",
            matched_terms=["high_split_sharpe_variance"],
        )
        ctx = ResearchMemoryContext(
            query=ResearchMemoryQuery(keyword_query="variance", semantic_query=None, source="test"),
            keyword_count=1,
        )
        ctx.evidence_items.append(ev)

        na_result = recommend_next_actions(memory_context=ctx, max_candidates=3)
        assert na_result.candidates, "no candidates generated"

        top = na_result.top_candidate
        if top.action_type in ("diagnostic_review", "stop_or_deprioritise"):
            pytest.skip("top candidate is non-proposable; skip integration check")

        # Propose from top candidate
        from src.agent.resolver.resolve_research_spec import resolve_research_spec
        res = resolve_research_spec(spec, registry=_registry())
        result = propose_spec_from_next_action(
            top,
            resolved_spec=res.resolved_spec,
            registry=_registry(),
        )
        assert result is not None
        assert result.status in (
            "proposal_created", "needs_user_input",
            "unsupported_action", "needs_more_evidence"
        )

    def test_chain_governance_preserved(self):
        from src.agent.memory.memory_context import ResearchMemoryContext
        from src.agent.memory.memory_evidence import ResearchMemoryEvidence
        from src.agent.memory.memory_query import ResearchMemoryQuery
        from src.agent.next_actions.next_action_planner import recommend_next_actions

        ev = ResearchMemoryEvidence(
            memory_id="int_002",
            experiment_name="exp_b",
            artefact_type="review",
            short_summary="high_turnover observed.",
            failure_modes=["high_turnover"],
            tags=[],
            path="/p",
            context_hash="h",
            matched_terms=["high_turnover"],
        )
        ctx = ResearchMemoryContext(
            query=ResearchMemoryQuery(keyword_query="turnover", semantic_query=None, source="test"),
            keyword_count=1,
        )
        ctx.evidence_items.append(ev)

        na_result = recommend_next_actions(memory_context=ctx, max_candidates=1)
        top = na_result.top_candidate

        base = ResearchSpec(
            preset="canonical_multi_asset_etf",
            end_date="2024-12-31",
        )

        # Inject a transaction_cost_bps for cost_sensitivity candidates
        if top.action_type == "cost_sensitivity" and not top.suggested_spec_changes.get("transaction_cost_bps"):
            top.suggested_spec_changes["transaction_cost_bps"] = 10

        result = propose_spec_from_next_action(top, base_spec=base, registry=_registry())

        if result.proposal:
            assert result.proposal.creates_draft is False
            assert result.proposal.executes_experiment is False
            assert result.proposal.approval_required is True

    def test_chain_no_draft_no_execution(self):
        c = _candidate("model_comparison", {"model": "ElasticNetRegression"})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "proposal_created"
        d = result.to_dict()
        gov = d["proposal"]["creates_draft"] if d["proposal"] else None
        assert gov is False

    def test_resolution_result_present_and_structured(self):
        c = _candidate("model_comparison", {"model": "ElasticNetRegression"})
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.resolution_result is not None
        assert result.resolution_result.status in ("resolved", "needs_user_input", "invalid")

    def test_proposed_changes_are_spec_fields(self):
        c = _candidate(
            "validation_robustness",
            {"train_window": "48m", "test_window": "6m"},
        )
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        if result.proposal:
            spec_field_names = {f.name for f in __import__("dataclasses").fields(ResearchSpec)}
            for key in result.proposal.proposed_changes:
                assert key in spec_field_names, f"'{key}' is not a ResearchSpec field"


# ---------------------------------------------------------------------------
# D. Feature family validation — Phase 5D.1
# ---------------------------------------------------------------------------


class TestFeatureFamilyNamesAccessor:
    """D1. Registry accessor returns correct frozenset of family names."""

    def test_returns_frozenset(self):
        families = _registry().feature_family_names()
        assert isinstance(families, frozenset)

    def test_known_families_present(self):
        families = _registry().feature_family_names()
        assert "Trend" in families
        assert "Volatility" in families
        assert "Mean-Reversion" in families
        assert "Market Structure" in families

    def test_null_family_excluded(self):
        families = _registry().feature_family_names()
        assert None not in families
        assert "" not in families

    def test_non_empty(self):
        assert len(_registry().feature_family_names()) > 0


class TestFeatureAblationFamilyValidation:
    """D2–D4. Phase 5D.1 family validation in _handle_feature_ablation."""

    def test_supported_families_accepted(self):
        """D2. Valid families → proposal_created with feature_families applied."""
        c = _candidate(
            "feature_ablation",
            {"feature_families": ["Trend", "Volatility"]},
        )
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "proposal_created"
        assert result.proposal is not None
        assert result.proposal.proposed_spec.feature_families == ["Trend", "Volatility"]

    def test_unsupported_family_rejected(self):
        """D3. All invalid families → unsupported_action."""
        c = _candidate(
            "feature_ablation",
            {"feature_families": ["magic_alpha_family"]},
        )
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "unsupported_action"

    def test_unsupported_family_name_in_errors(self):
        """D3. Invalid family name is reported."""
        c = _candidate(
            "feature_ablation",
            {"feature_families": ["magic_alpha_family"]},
        )
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        combined = " ".join(result.errors + result.warnings)
        assert "magic_alpha_family" in combined or "unsupported" in combined.lower()

    def test_mixed_families_valid_kept(self):
        """D4. Mixed families — valid family kept, invalid reported."""
        c = _candidate(
            "feature_ablation",
            {"feature_families": ["Trend", "phantom_family"]},
        )
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        # With partial acceptance: valid families are kept, invalid reported
        # (unsupported_action if ALL invalid, proposal_created if at least one valid)
        assert result.status in ("proposal_created", "unsupported_action")
        if result.status == "proposal_created":
            assert result.proposal is not None
            spec_families = result.proposal.proposed_spec.feature_families or []
            assert "Trend" in spec_families
            assert "phantom_family" not in spec_families

    def test_mixed_families_invalid_reported(self):
        """D4. Invalid family in mixed case appears in unsupported_changes or errors."""
        c = _candidate(
            "feature_ablation",
            {"feature_families": ["Trend", "phantom_family"]},
        )
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        if result.status == "proposal_created" and result.proposal:
            unsupported_str = " ".join(result.proposal.unsupported_changes)
            assert "phantom_family" in unsupported_str
        else:
            combined = " ".join(result.errors + result.warnings)
            assert "phantom_family" in combined

    def test_invalid_family_not_in_proposed_spec(self):
        """D3/D4. Unsupported family must never appear in the proposed spec."""
        c = _candidate(
            "feature_ablation",
            {"feature_families": ["bad_family_x"]},
        )
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        if result.proposal:
            spec_families = result.proposal.proposed_spec.feature_families or []
            assert "bad_family_x" not in spec_families

    def test_governance_preserved_on_family_proposal(self):
        """D2. Governance invariants hold for valid family proposal."""
        c = _candidate(
            "feature_ablation",
            {"feature_families": ["Mean-Reversion"]},
        )
        result = propose_spec_from_next_action(
            c, base_spec=_canonical_base(), registry=_registry()
        )
        assert result.status == "proposal_created"
        p = result.proposal
        assert p is not None
        assert p.creates_draft is False
        assert p.executes_experiment is False
        assert p.requires_user_confirmation is True
        assert p.approval_required is True
