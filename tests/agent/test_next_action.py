"""NextActionCandidate unit tests — Phase 5B.

Covers:
  A. Construction and defaults
  B. Governance invariants (creates_draft, executes_experiment, requires_confirmation)
  C. to_dict() serialisation and key completeness
  D. VALID_ACTION_TYPES / VALID_PRIORITIES / VALID_CONFIDENCES constants
"""

from __future__ import annotations

import pytest

from src.agent.next_actions.next_action import (
    VALID_ACTION_TYPES,
    VALID_CONFIDENCES,
    VALID_PRIORITIES,
    NextActionCandidate,
)


def _candidate(**kwargs) -> NextActionCandidate:
    defaults = dict(
        action_id="test-001",
        title="Test action",
        action_type="feature_ablation",
        description="Test description",
        rationale="Prior evidence suggests this direction.",
        main_risks=["test risk"],
    )
    defaults.update(kwargs)
    return NextActionCandidate(**defaults)


# ---------------------------------------------------------------------------
# A. Construction and defaults
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_minimal_construction(self):
        c = _candidate()
        assert c.action_id == "test-001"
        assert c.title == "Test action"

    def test_default_priority_medium(self):
        c = _candidate()
        assert c.priority == "medium"

    def test_default_confidence_medium(self):
        c = _candidate()
        assert c.confidence == "medium"

    def test_default_evidence_ids_empty(self):
        c = _candidate()
        assert c.supporting_evidence_ids == []
        assert c.contradicting_evidence_ids == []

    def test_default_suggested_spec_changes_empty(self):
        c = _candidate()
        assert c.suggested_spec_changes == {}

    def test_default_suggested_plan_focus_none(self):
        c = _candidate()
        assert c.suggested_plan_focus is None

    def test_default_expected_benefit_none(self):
        c = _candidate()
        assert c.expected_benefit is None

    def test_default_failure_modes_empty(self):
        c = _candidate()
        assert c.failure_modes_to_monitor == []

    def test_default_requires_new_spec_true(self):
        c = _candidate()
        assert c.requires_new_spec is True

    def test_evidence_ids_stored(self):
        c = _candidate(supporting_evidence_ids=["mem1", "mem2"])
        assert c.supporting_evidence_ids == ["mem1", "mem2"]

    def test_spec_changes_stored(self):
        c = _candidate(suggested_spec_changes={"model": "RidgeRegressionModel"})
        assert c.suggested_spec_changes["model"] == "RidgeRegressionModel"


# ---------------------------------------------------------------------------
# B. Governance invariants
# ---------------------------------------------------------------------------


class TestGovernanceInvariants:
    def test_creates_draft_is_false_by_default(self):
        c = _candidate()
        assert c.creates_draft is False

    def test_executes_experiment_is_false_by_default(self):
        c = _candidate()
        assert c.executes_experiment is False

    def test_requires_user_confirmation_is_true_by_default(self):
        c = _candidate()
        assert c.requires_user_confirmation is True

    def test_to_dict_creates_draft_false(self):
        c = _candidate()
        assert c.to_dict()["creates_draft"] is False

    def test_to_dict_executes_experiment_false(self):
        c = _candidate()
        assert c.to_dict()["executes_experiment"] is False

    def test_to_dict_requires_user_confirmation_true(self):
        c = _candidate()
        assert c.to_dict()["requires_user_confirmation"] is True

    def test_governance_flags_independent_of_other_fields(self):
        c = _candidate(priority="high", confidence="high")
        assert c.creates_draft is False
        assert c.executes_experiment is False
        assert c.requires_user_confirmation is True


# ---------------------------------------------------------------------------
# C. to_dict() serialisation
# ---------------------------------------------------------------------------


class TestToDictSerialisation:
    _EXPECTED_KEYS = {
        "action_id",
        "title",
        "action_type",
        "description",
        "rationale",
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
        "suggested_spec_changes",
        "suggested_plan_focus",
        "expected_benefit",
        "main_risks",
        "failure_modes_to_monitor",
        "priority",
        "confidence",
        "requires_new_spec",
        "requires_user_confirmation",
        "creates_draft",
        "executes_experiment",
    }

    def test_all_keys_present(self):
        d = _candidate().to_dict()
        assert self._EXPECTED_KEYS == set(d.keys())

    def test_lists_are_copies(self):
        c = _candidate(supporting_evidence_ids=["m1"])
        d = c.to_dict()
        d["supporting_evidence_ids"].append("m2")
        assert c.supporting_evidence_ids == ["m1"]

    def test_dict_is_copy(self):
        c = _candidate(suggested_spec_changes={"model": "Ridge"})
        d = c.to_dict()
        d["suggested_spec_changes"]["extra"] = "x"
        assert "extra" not in c.suggested_spec_changes

    def test_action_id_serialised(self):
        c = _candidate(action_id="unique-42")
        assert c.to_dict()["action_id"] == "unique-42"

    def test_priority_serialised(self):
        c = _candidate(priority="high")
        assert c.to_dict()["priority"] == "high"

    def test_confidence_serialised(self):
        c = _candidate(confidence="low")
        assert c.to_dict()["confidence"] == "low"

    def test_none_fields_serialised_as_none(self):
        c = _candidate()
        d = c.to_dict()
        assert d["suggested_plan_focus"] is None
        assert d["expected_benefit"] is None

    def test_failure_modes_serialised(self):
        c = _candidate(failure_modes_to_monitor=["poor_oos_consistency"])
        assert c.to_dict()["failure_modes_to_monitor"] == ["poor_oos_consistency"]


# ---------------------------------------------------------------------------
# D. Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_valid_action_types_is_frozenset(self):
        assert isinstance(VALID_ACTION_TYPES, frozenset)

    def test_feature_ablation_in_action_types(self):
        assert "feature_ablation" in VALID_ACTION_TYPES

    def test_model_comparison_in_action_types(self):
        assert "model_comparison" in VALID_ACTION_TYPES

    def test_cost_sensitivity_in_action_types(self):
        assert "cost_sensitivity" in VALID_ACTION_TYPES

    def test_validation_robustness_in_action_types(self):
        assert "validation_robustness" in VALID_ACTION_TYPES

    def test_repeat_baseline_in_action_types(self):
        assert "repeat_baseline" in VALID_ACTION_TYPES

    def test_diagnostic_review_in_action_types(self):
        assert "diagnostic_review" in VALID_ACTION_TYPES

    def test_valid_priorities(self):
        assert VALID_PRIORITIES == {"high", "medium", "low"}

    def test_valid_confidences(self):
        assert VALID_CONFIDENCES == {"high", "medium", "low"}

    def test_all_spec_action_types_present(self):
        expected = {
            "feature_ablation",
            "model_comparison",
            "cost_sensitivity",
            "validation_robustness",
            "portfolio_construction_test",
            "universe_review",
            "diagnostic_review",
            "repeat_baseline",
            "stop_or_deprioritise",
            "needs_more_evidence",
            "custom",
        }
        assert expected == VALID_ACTION_TYPES
