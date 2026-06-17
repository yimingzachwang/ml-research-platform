"""Tests for src/agent/drafting/draft_mapping.py

Covers: detect_unsupported_fields, map_spec_to_changes, and _parse_window_months.
No disk I/O — all mapping is in-memory.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.agent.drafting.draft_mapping import (
    _parse_window_months,
    detect_hard_unsupported_fields,
    detect_unsupported_fields,
    map_spec_to_changes,
)
from src.agent.resolver.resolved_spec import ResolvedResearchSpec
from src.orchestration.config_generation.draft_schema import DraftChange

# ---------------------------------------------------------------------------
# Shared base config (minimal version-2 structure)
# ---------------------------------------------------------------------------

_BASE_CONFIG: dict[str, Any] = {
    "version": "2",
    "name": "canonical_ml_multi_asset",
    "model": {"type": "RidgeRegression", "params": {"alpha": 0.5}},
    "labels": {"type": "ranking_target", "params": {"horizon": 21}},
    "signal": {
        "type": "top_n",
        "params": {"n": 3, "n_long": None, "n_short": None, "threshold": None},
    },
    "validation": {
        "type": "rolling",
        "parameters": {"train_months": 48, "test_months": 12, "gap_days": 0},
    },
    "execution": {"transaction_cost_bps": 5.0},
    "portfolio_construction": {
        "weighting": {
            "scheme": "equal_weight",
            "prediction_normalization": "none",
            "temperature": None,
        }
    },
    "features": {
        "ticker": "SPY",
        "entries": [
            {"name": "mom_20", "type": "momentum", "params": {"lookback": 20}},
            {"name": "mom_60", "type": "momentum", "params": {"lookback": 60}},
        ],
    },
}


def _make_resolved(
    user_fields: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ResolvedResearchSpec:
    """Build a minimal valid ResolvedResearchSpec for mapping tests."""
    spec = ResolvedResearchSpec(
        preset="canonical_multi_asset_etf",
        tickers=["SPY", "TLT", "GLD"],
        start_date="2015-01-01",
        end_date="2024-12-31",
        model="RidgeRegression",
        model_params={"alpha": 0.5},
        signal="top_n",
        signal_params={"n": 3},
        label="ranking_target",
        label_horizon=21,
        train_window="48m",
        test_window="12m",
        transaction_cost_bps=5.0,
        user_specified_fields=user_fields or {},
        preset_fields={
            "model": "RidgeRegression",
            "model_params": {"alpha": 0.5},
            "signal": "top_n",
            "signal_params": {"n": 3},
            "label": "ranking_target",
            "label_horizon": 21,
            "train_window": "48m",
            "test_window": "12m",
            "transaction_cost_bps": 5.0,
        },
    )
    for k, v in kwargs.items():
        object.__setattr__(spec, k, v)
    return spec


# ---------------------------------------------------------------------------
# detect_unsupported_fields
# ---------------------------------------------------------------------------


class TestDetectUnsupportedFields:
    """Tests for both detect_unsupported_fields (hard + soft) and
    detect_hard_unsupported_fields (blocking only)."""

    # --- Hard-unsupported (structural, blocking) ---

    def test_no_unsupported_when_only_model(self):
        resolved = _make_resolved(user_fields={"model": "Lasso"})
        assert detect_unsupported_fields(resolved) == []

    def test_universe_field_is_unsupported(self):
        resolved = _make_resolved(user_fields={"universe": "global_equity"})
        assert "universe" in detect_unsupported_fields(resolved)

    def test_tickers_field_is_unsupported(self):
        resolved = _make_resolved(user_fields={"tickers": ["SPY", "QQQ"]})
        assert "tickers" in detect_unsupported_fields(resolved)

    def test_frequency_field_is_unsupported(self):
        resolved = _make_resolved(user_fields={"frequency": "weekly"})
        assert "frequency" in detect_unsupported_fields(resolved)

    def test_data_source_field_is_unsupported(self):
        resolved = _make_resolved(user_fields={"data_source": "bloomberg"})
        assert "data_source" in detect_unsupported_fields(resolved)

    def test_asset_class_field_is_unsupported(self):
        resolved = _make_resolved(user_fields={"asset_class": "equity"})
        assert "asset_class" in detect_unsupported_fields(resolved)

    def test_multiple_hard_unsupported_fields_all_returned(self):
        resolved = _make_resolved(user_fields={"tickers": ["SPY"], "universe": "global"})
        unsupported = detect_unsupported_fields(resolved)
        assert "tickers" in unsupported
        assert "universe" in unsupported

    def test_model_transaction_cost_not_unsupported(self):
        resolved = _make_resolved(
            user_fields={"model": "Lasso", "transaction_cost_bps": 10.0}
        )
        assert detect_unsupported_fields(resolved) == []

    # --- Requirement A: user-specified start_date is reported as unsupported ---

    def test_user_specified_start_date_is_soft_unsupported(self):
        """start_date in user_specified_fields must appear in detect_unsupported_fields."""
        resolved = _make_resolved(user_fields={"start_date": "2015-01-01"})
        result = detect_unsupported_fields(resolved)
        assert "start_date" in result

    # --- Requirement B: user-specified end_date is reported as unsupported ---

    def test_user_specified_end_date_is_soft_unsupported(self):
        """end_date in user_specified_fields must appear in detect_unsupported_fields."""
        resolved = _make_resolved(user_fields={"end_date": "2024-12-31"})
        result = detect_unsupported_fields(resolved)
        assert "end_date" in result

    def test_both_dates_reported_when_user_specified(self):
        resolved = _make_resolved(
            user_fields={"start_date": "2015-01-01", "end_date": "2024-12-31"}
        )
        result = detect_unsupported_fields(resolved)
        assert "start_date" in result
        assert "end_date" in result

    # --- Requirement C: preset/default dates do NOT create unsupported fields ---

    def test_defaulted_start_date_not_reported(self):
        """start_date from defaults (not user_specified_fields) must not appear."""
        resolved = _make_resolved(
            user_fields={"end_date": "2024-12-31"},  # only end_date is user-specified
            defaulted_fields={"start_date": "2015-01-01"},
        )
        result = detect_unsupported_fields(resolved)
        assert "start_date" not in result

    def test_preset_sourced_end_date_not_reported(self):
        """end_date from preset (not user_specified_fields) must not appear."""
        resolved = _make_resolved(
            user_fields={"preset": "canonical_multi_asset_etf"},
            preset_fields={"end_date": "2024-12-31"},
        )
        result = detect_unsupported_fields(resolved)
        assert "end_date" not in result

    def test_empty_user_fields_gives_no_unsupported(self):
        resolved = _make_resolved(user_fields={})
        assert detect_unsupported_fields(resolved) == []

    # --- Requirement D: preset is NOT treated as unsupported ---

    def test_preset_is_not_unsupported(self):
        """preset is provenance / base-selection metadata, not a draft field."""
        resolved = _make_resolved(user_fields={"preset": "canonical_multi_asset_etf"})
        assert "preset" not in detect_unsupported_fields(resolved)
        assert detect_unsupported_fields(resolved) == []

    def test_objective_not_unsupported(self):
        resolved = _make_resolved(user_fields={"objective": "explore momentum"})
        assert detect_unsupported_fields(resolved) == []

    def test_notes_not_unsupported(self):
        resolved = _make_resolved(user_fields={"notes": "some notes"})
        assert detect_unsupported_fields(resolved) == []

    # --- detect_hard_unsupported_fields — blocking subset only ---

    def test_hard_unsupported_does_not_include_dates(self):
        """start_date and end_date are soft, not hard — they must not appear in
        detect_hard_unsupported_fields."""
        resolved = _make_resolved(
            user_fields={"start_date": "2015-01-01", "end_date": "2024-12-31"}
        )
        assert detect_hard_unsupported_fields(resolved) == []

    def test_hard_unsupported_includes_tickers(self):
        resolved = _make_resolved(user_fields={"tickers": ["SPY", "QQQ"]})
        assert "tickers" in detect_hard_unsupported_fields(resolved)

    def test_hard_unsupported_includes_universe(self):
        resolved = _make_resolved(user_fields={"universe": "global_equity"})
        assert "universe" in detect_hard_unsupported_fields(resolved)

    def test_hard_and_soft_both_in_detect_all(self):
        """When both hard and soft unsupported are present, detect_unsupported_fields
        returns both; detect_hard_unsupported_fields returns only hard."""
        resolved = _make_resolved(
            user_fields={"tickers": ["SPY"], "end_date": "2024-12-31"}
        )
        all_u = detect_unsupported_fields(resolved)
        hard_u = detect_hard_unsupported_fields(resolved)
        assert "tickers" in all_u
        assert "end_date" in all_u
        assert "tickers" in hard_u
        assert "end_date" not in hard_u


# ---------------------------------------------------------------------------
# _parse_window_months
# ---------------------------------------------------------------------------


class TestParseWindowMonths:
    def test_parses_48m(self):
        assert _parse_window_months("48m") == 48

    def test_parses_12m(self):
        assert _parse_window_months("12m") == 12

    def test_parses_uppercase_M(self):
        assert _parse_window_months("24M") == 24

    def test_parses_with_whitespace(self):
        assert _parse_window_months(" 36m ") == 36

    def test_passes_through_int(self):
        assert _parse_window_months(60) == 60

    def test_returns_none_for_non_matching_string(self):
        assert _parse_window_months("rolling") is None

    def test_returns_none_for_float(self):
        assert _parse_window_months(48.0) is None

    def test_returns_none_for_none(self):
        assert _parse_window_months(None) is None

    def test_returns_none_for_empty_string(self):
        assert _parse_window_months("") is None


# ---------------------------------------------------------------------------
# map_spec_to_changes — scalar fields
# ---------------------------------------------------------------------------


class TestMapModelType:
    def test_model_change_generates_draft_change(self):
        resolved = _make_resolved(
            user_fields={"model": "Lasso"},
            model="Lasso",
        )
        changes, warnings = map_spec_to_changes(resolved, _BASE_CONFIG)
        model_changes = [c for c in changes if c.section == "model" and c.field == "type"]
        assert len(model_changes) == 1
        assert model_changes[0].proposed_value == "Lasso"
        assert model_changes[0].current_value == "RidgeRegression"

    def test_no_change_when_model_matches_base(self):
        resolved = _make_resolved(
            user_fields={"model": "RidgeRegression"},
            model="RidgeRegression",
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        model_changes = [c for c in changes if c.section == "model" and c.field == "type"]
        assert model_changes == []

    def test_model_not_in_user_fields_produces_no_change(self):
        resolved = _make_resolved(user_fields={})
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        model_changes = [c for c in changes if c.section == "model" and c.field == "type"]
        assert model_changes == []


class TestMapLabelHorizon:
    def test_label_horizon_generates_change(self):
        resolved = _make_resolved(
            user_fields={"label_horizon": 42},
            label_horizon=42,
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        horizon_changes = [
            c for c in changes if c.section == "labels" and c.field == "params.horizon"
        ]
        assert len(horizon_changes) == 1
        assert horizon_changes[0].proposed_value == 42
        assert horizon_changes[0].current_value == 21

    def test_label_horizon_no_change_when_matches(self):
        resolved = _make_resolved(
            user_fields={"label_horizon": 21},
            label_horizon=21,
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        assert not any(
            c.section == "labels" and c.field == "params.horizon" for c in changes
        )


class TestMapLabelType:
    def test_label_type_generates_change(self):
        resolved = _make_resolved(
            user_fields={"label": "forward_returns"},
            label="forward_returns",
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        label_changes = [c for c in changes if c.section == "labels" and c.field == "type"]
        assert len(label_changes) == 1
        assert label_changes[0].proposed_value == "forward_returns"


class TestMapSignalType:
    def test_signal_type_generates_change(self):
        resolved = _make_resolved(
            user_fields={"signal": "sign"},
            signal="sign",
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        signal_changes = [
            c for c in changes if c.section == "signal" and c.field == "type"
        ]
        assert len(signal_changes) == 1
        assert signal_changes[0].proposed_value == "sign"


class TestMapTransactionCost:
    def test_transaction_cost_generates_change(self):
        resolved = _make_resolved(
            user_fields={"transaction_cost_bps": 10.0},
            transaction_cost_bps=10.0,
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        cost_changes = [
            c for c in changes
            if c.section == "execution" and c.field == "transaction_cost_bps"
        ]
        assert len(cost_changes) == 1
        assert cost_changes[0].proposed_value == 10.0
        assert cost_changes[0].current_value == 5.0


# ---------------------------------------------------------------------------
# map_spec_to_changes — model_params
# ---------------------------------------------------------------------------


class TestMapModelParams:
    def test_model_params_alpha_generates_change(self):
        resolved = _make_resolved(
            user_fields={"model_params": {"alpha": 2.0}},
            model_params={"alpha": 2.0},
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        alpha_changes = [
            c for c in changes if c.section == "model" and c.field == "params.alpha"
        ]
        assert len(alpha_changes) == 1
        assert alpha_changes[0].proposed_value == 2.0
        assert alpha_changes[0].current_value == 0.5

    def test_model_params_alpha_no_change_when_matches(self):
        resolved = _make_resolved(
            user_fields={"model_params": {"alpha": 0.5}},
            model_params={"alpha": 0.5},
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        assert not any(
            c.section == "model" and c.field == "params.alpha" for c in changes
        )

    def test_model_params_unknown_key_ignored(self):
        resolved = _make_resolved(
            user_fields={"model_params": {"unknown_param": 99}},
            model_params={"unknown_param": 99},
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        assert not any(c.section == "model" for c in changes)

    def test_model_params_multiple_keys(self):
        resolved = _make_resolved(
            user_fields={"model_params": {"alpha": 1.0, "max_iter": 2000}},
            model_params={"alpha": 1.0, "max_iter": 2000},
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        model_param_changes = [c for c in changes if c.section == "model"]
        assert len(model_param_changes) == 2


# ---------------------------------------------------------------------------
# map_spec_to_changes — signal_params
# ---------------------------------------------------------------------------


class TestMapSignalParams:
    def test_signal_params_n_generates_change(self):
        resolved = _make_resolved(
            user_fields={"signal_params": {"n": 5}},
            signal_params={"n": 5},
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        n_changes = [
            c for c in changes if c.section == "signal" and c.field == "params.n"
        ]
        assert len(n_changes) == 1
        assert n_changes[0].proposed_value == 5

    def test_signal_params_n_no_change_when_matches(self):
        resolved = _make_resolved(
            user_fields={"signal_params": {"n": 3}},
            signal_params={"n": 3},
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        assert not any(c.section == "signal" and c.field == "params.n" for c in changes)


# ---------------------------------------------------------------------------
# map_spec_to_changes — validation windows
# ---------------------------------------------------------------------------


class TestMapValidationWindows:
    def test_train_window_48m_generates_change(self):
        base = dict(_BASE_CONFIG)
        base["validation"] = {
            "type": "rolling",
            "parameters": {"train_months": 36, "test_months": 12, "gap_days": 0},
        }
        resolved = _make_resolved(
            user_fields={"train_window": "48m"},
            train_window="48m",
        )
        changes, warnings = map_spec_to_changes(resolved, base)
        train_changes = [
            c for c in changes
            if c.section == "validation" and c.field == "parameters.train_months"
        ]
        assert len(train_changes) == 1
        assert train_changes[0].proposed_value == 48
        assert train_changes[0].current_value == 36
        assert warnings == []

    def test_test_window_6m_generates_change(self):
        base = dict(_BASE_CONFIG)
        base["validation"] = {
            "type": "rolling",
            "parameters": {"train_months": 48, "test_months": 12, "gap_days": 0},
        }
        resolved = _make_resolved(
            user_fields={"test_window": "6m"},
            test_window="6m",
        )
        changes, warnings = map_spec_to_changes(resolved, base)
        test_changes = [
            c for c in changes
            if c.section == "validation" and c.field == "parameters.test_months"
        ]
        assert len(test_changes) == 1
        assert test_changes[0].proposed_value == 6
        assert warnings == []

    def test_invalid_window_format_produces_warning(self):
        resolved = _make_resolved(
            user_fields={"train_window": "rolling_48"},
            train_window="rolling_48",
        )
        changes, warnings = map_spec_to_changes(resolved, _BASE_CONFIG)
        assert not any(
            c.field == "parameters.train_months" for c in changes
        )
        assert any("train_window" in w for w in warnings)

    def test_window_no_change_when_matches_base(self):
        resolved = _make_resolved(
            user_fields={"train_window": "48m"},
            train_window="48m",
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        assert not any(c.field == "parameters.train_months" for c in changes)


# ---------------------------------------------------------------------------
# map_spec_to_changes — features
# ---------------------------------------------------------------------------


class TestMapFeatures:
    def test_feature_specs_adds_new_features(self):
        resolved = _make_resolved(
            user_fields={"features": ["vol_21"]},
            features=["vol_21"],
            feature_specs=[
                {"name": "vol_21", "type": "rolling_volatility", "params": {"window": 21}}
            ],
        )
        changes, warnings = map_spec_to_changes(resolved, _BASE_CONFIG)
        add_changes = [
            c for c in changes if c.section == "features" and c.field == "entries.add"
        ]
        assert len(add_changes) == 1
        assert add_changes[0].proposed_value["name"] == "vol_21"
        # Base config has mom_20 and mom_60 which are not in the spec — governance
        # requires an implied-removal warning (not silent suppression).
        assert any("NOT removed" in w for w in warnings), (
            "Expected implied-removal warning for base features absent from spec"
        )

    def test_feature_specs_skips_features_already_in_base(self):
        resolved = _make_resolved(
            user_fields={"features": ["mom_20"]},
            features=["mom_20"],
            feature_specs=[
                {"name": "mom_20", "type": "momentum", "params": {"lookback": 20}}
            ],
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        add_changes = [c for c in changes if c.field == "entries.add"]
        assert add_changes == []

    def test_feature_names_without_specs_produces_warning(self):
        resolved = _make_resolved(
            user_fields={"features": ["vol_21"]},
            features=["vol_21"],
            feature_specs=[],
        )
        changes, warnings = map_spec_to_changes(resolved, _BASE_CONFIG)
        add_changes = [c for c in changes if c.field == "entries.add"]
        assert add_changes == []
        assert any("feature_specs" in w or "feature names" in w.lower() for w in warnings)

    def test_feature_families_in_user_fields_triggers_feature_mapping(self):
        resolved = _make_resolved(
            user_fields={"feature_families": ["momentum"]},
            feature_families=["momentum"],
            feature_specs=[
                {"name": "mom_252", "type": "momentum", "params": {"lookback": 252}}
            ],
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        add_changes = [c for c in changes if c.field == "entries.add"]
        assert len(add_changes) == 1
        assert add_changes[0].proposed_value["name"] == "mom_252"

    def test_multiple_new_features_all_added(self):
        resolved = _make_resolved(
            user_fields={"features": ["vol_21", "zscore_20"]},
            features=["vol_21", "zscore_20"],
            feature_specs=[
                {"name": "vol_21", "type": "rolling_volatility", "params": {"window": 21}},
                {"name": "zscore_20", "type": "rolling_zscore", "params": {"window": 20}},
            ],
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        add_changes = [c for c in changes if c.field == "entries.add"]
        assert len(add_changes) == 2

    # --- Requirement 7B: missing required feature params are warned about ---

    def test_feature_with_missing_required_params_produces_warning_and_no_add(self):
        """A feature spec with missing required params must generate a warning
        and NOT produce an entries.add change (validate_draft would reject it)."""
        resolved = _make_resolved(
            user_fields={"features": ["momentum_bad"]},
            features=["momentum_bad"],
            feature_specs=[
                # momentum requires 'lookback' param — omitted intentionally
                {"name": "momentum_bad", "type": "momentum", "params": {}}
            ],
        )
        changes, warnings = map_spec_to_changes(resolved, _BASE_CONFIG)
        add_changes = [c for c in changes if c.field == "entries.add"]
        assert add_changes == [], "No add change should be generated for invalid feature"
        assert any("momentum_bad" in w or "missing" in w.lower() for w in warnings)

    def test_feature_with_invalid_type_produces_warning_and_no_add(self):
        """A feature spec with an unrecognised type must warn and not add."""
        resolved = _make_resolved(
            user_fields={"features": ["bogus_feature"]},
            features=["bogus_feature"],
            feature_specs=[
                {"name": "bogus_feature", "type": "NonExistentFeatureType", "params": {}}
            ],
        )
        changes, warnings = map_spec_to_changes(resolved, _BASE_CONFIG)
        add_changes = [c for c in changes if c.field == "entries.add"]
        assert add_changes == []
        assert any("bogus_feature" in w or "unsupported type" in w.lower() for w in warnings)

    # --- Requirement 7C: feature removals are not silently ignored ---

    def test_base_features_not_in_spec_produce_implied_removal_warning(self):
        """When user specifies a feature set and the base config has extra features,
        the unmapped removals must be reported in warnings (not silently ignored).
        Phase 3B is add-only; Phase 3C will add remove support."""
        resolved = _make_resolved(
            user_fields={"features": ["vol_21"]},
            features=["vol_21"],
            feature_specs=[
                {"name": "vol_21", "type": "rolling_volatility", "params": {"window": 21}}
            ],
        )
        # Base config has mom_20 and mom_60 — these are "implied removals"
        changes, warnings = map_spec_to_changes(resolved, _BASE_CONFIG)
        assert any(
            "not in your specified" in w or "NOT removed" in w or "implied" in w.lower()
            for w in warnings
        ), f"Expected implied-removal warning but got: {warnings}"

    def test_no_implied_removal_warning_when_spec_covers_all_base_features(self):
        """When user spec includes ALL base features (plus new ones), no implied
        removal warning should fire."""
        resolved = _make_resolved(
            user_fields={"features": ["mom_20", "mom_60", "vol_21"]},
            features=["mom_20", "mom_60", "vol_21"],
            feature_specs=[
                {"name": "mom_20", "type": "momentum", "params": {"lookback": 20}},
                {"name": "mom_60", "type": "momentum", "params": {"lookback": 60}},
                {"name": "vol_21", "type": "rolling_volatility", "params": {"window": 21}},
            ],
        )
        changes, warnings = map_spec_to_changes(resolved, _BASE_CONFIG)
        # All base features are covered — no implied removals
        assert not any("NOT removed" in w for w in warnings), (
            f"Unexpected implied-removal warning: {warnings}"
        )


# ---------------------------------------------------------------------------
# map_spec_to_changes — skipped fields (no change-path equivalent)
# ---------------------------------------------------------------------------


class TestSkippedFields:
    """Tests for fields that produce no DraftChanges and no warnings.

    start_date and end_date are NOT silently ignored — they are reported
    as soft-unsupported by detect_unsupported_fields(). But map_spec_to_changes
    correctly generates no DraftChange for them (there is no change-path for
    date_range). That is expected behaviour.

    preset is provenance metadata and is silently skipped.
    """

    def test_start_date_generates_no_change(self):
        resolved = _make_resolved(
            user_fields={"start_date": "2015-01-01"},
        )
        changes, warnings = map_spec_to_changes(resolved, _BASE_CONFIG)
        assert changes == []
        assert warnings == []

    def test_end_date_generates_no_change(self):
        resolved = _make_resolved(
            user_fields={"end_date": "2024-12-31"},
        )
        changes, warnings = map_spec_to_changes(resolved, _BASE_CONFIG)
        assert changes == []
        assert warnings == []

    def test_preset_generates_no_change(self):
        resolved = _make_resolved(
            user_fields={"preset": "canonical_multi_asset_etf"},
        )
        changes, warnings = map_spec_to_changes(resolved, _BASE_CONFIG)
        assert changes == []
        assert warnings == []

    def test_notes_generates_no_change(self):
        resolved = _make_resolved(
            user_fields={"notes": "test run"},
        )
        changes, warnings = map_spec_to_changes(resolved, _BASE_CONFIG)
        assert changes == []
        assert warnings == []


# ---------------------------------------------------------------------------
# Only user_specified_fields drive changes
# ---------------------------------------------------------------------------


class TestOnlyUserFieldsDriveChanges:
    def test_preset_fields_do_not_generate_changes(self):
        """Fields from presets should not produce changes even if their value
        differs from the base config, because they were not user-specified."""
        resolved = _make_resolved(
            user_fields={"start_date": "2015-01-01", "end_date": "2024-12-31"},
            # model comes from preset, not user
            model="Lasso",  # different from base config, but preset-sourced
            preset_fields={"model": "Lasso"},
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        model_changes = [c for c in changes if c.field == "type" and c.section == "model"]
        assert model_changes == []

    def test_multiple_user_fields_all_generate_changes(self):
        resolved = _make_resolved(
            user_fields={
                "model": "Lasso",
                "label_horizon": 42,
                "transaction_cost_bps": 10.0,
            },
            model="Lasso",
            label_horizon=42,
            transaction_cost_bps=10.0,
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        sections_changed = {c.section for c in changes}
        assert "model" in sections_changed
        assert "labels" in sections_changed
        assert "execution" in sections_changed

    def test_empty_user_fields_produces_no_changes(self):
        resolved = _make_resolved(user_fields={})
        changes, warnings = map_spec_to_changes(resolved, _BASE_CONFIG)
        assert changes == []
        assert warnings == []


# ---------------------------------------------------------------------------
# DraftChange properties
# ---------------------------------------------------------------------------


class TestDraftChangeProperties:
    def test_draft_change_has_rationale(self):
        resolved = _make_resolved(
            user_fields={"model": "Lasso"},
            model="Lasso",
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        assert changes
        assert all(isinstance(c.rationale, str) and c.rationale for c in changes)

    def test_draft_change_is_dataclass_instance(self):
        resolved = _make_resolved(
            user_fields={"label_horizon": 42},
            label_horizon=42,
        )
        changes, _ = map_spec_to_changes(resolved, _BASE_CONFIG)
        assert all(isinstance(c, DraftChange) for c in changes)
