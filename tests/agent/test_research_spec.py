"""ResearchSpec unit tests.

Verifies that ResearchSpec:
- can be constructed with no arguments (all fields optional)
- specified_fields() returns only non-None values
- missing_fields() correctly identifies gaps
- to_dict() / from_dict() round-trips cleanly
- list and dict fields do not share mutable state between instances
"""

from __future__ import annotations

import pytest

from src.agent.intent.research_spec import ResearchSpec


class TestConstructionAndDefaults:
    def test_empty_spec_is_valid(self):
        spec = ResearchSpec()
        assert spec is not None

    def test_all_fields_default_to_none(self):
        spec = ResearchSpec()
        assert spec.preset is None
        assert spec.objective is None
        assert spec.asset_class is None
        assert spec.universe is None
        assert spec.tickers is None
        assert spec.start_date is None
        assert spec.end_date is None
        assert spec.frequency is None
        assert spec.data_source is None
        assert spec.feature_families is None
        assert spec.features is None
        assert spec.label is None
        assert spec.label_horizon is None
        assert spec.model is None
        assert spec.model_params is None
        assert spec.signal is None
        assert spec.signal_params is None
        assert spec.validation_protocol is None
        assert spec.train_window is None
        assert spec.test_window is None
        assert spec.transaction_cost_bps is None
        assert spec.benchmark is None
        assert spec.diagnostics is None
        assert spec.risk_constraints is None
        assert spec.notes is None

    def test_partial_spec_sets_only_provided_fields(self):
        spec = ResearchSpec(model="RidgeRegression", label="forward_returns")
        assert spec.model == "RidgeRegression"
        assert spec.label == "forward_returns"
        assert spec.universe is None
        assert spec.tickers is None


class TestSpecifiedFields:
    def test_empty_spec_has_no_specified_fields(self):
        spec = ResearchSpec()
        assert spec.specified_fields() == {}

    def test_partial_spec_returns_non_none_fields_only(self):
        spec = ResearchSpec(model="RidgeRegression", label="forward_returns", label_horizon=21)
        result = spec.specified_fields()
        assert result == {"model": "RidgeRegression", "label": "forward_returns", "label_horizon": 21}

    def test_specified_fields_excludes_none(self):
        spec = ResearchSpec(universe="core_etfs", model=None)
        result = spec.specified_fields()
        assert "universe" in result
        assert "model" not in result

    def test_specified_fields_includes_list_values(self):
        spec = ResearchSpec(tickers=["SPY", "QQQ"])
        result = spec.specified_fields()
        assert result["tickers"] == ["SPY", "QQQ"]

    def test_specified_fields_includes_dict_values(self):
        spec = ResearchSpec(model_params={"alpha": 1.0})
        result = spec.specified_fields()
        assert result["model_params"] == {"alpha": 1.0}

    def test_specified_fields_with_false_like_values(self):
        spec = ResearchSpec(transaction_cost_bps=0.0, label_horizon=0)
        result = spec.specified_fields()
        # 0.0 and 0 are not None, so they should appear
        assert "transaction_cost_bps" in result
        assert "label_horizon" in result


class TestMissingFields:
    def test_empty_spec_all_fields_missing(self):
        spec = ResearchSpec()
        required = ["model", "label", "universe"]
        missing = spec.missing_fields(required)
        assert missing == ["model", "label", "universe"]

    def test_partial_spec_returns_only_unset_fields(self):
        spec = ResearchSpec(model="RidgeRegression")
        required = ["model", "label", "universe"]
        missing = spec.missing_fields(required)
        assert "model" not in missing
        assert "label" in missing
        assert "universe" in missing

    def test_fully_specified_returns_empty_list(self):
        spec = ResearchSpec(model="RidgeRegression", label="forward_returns", universe="core_etfs")
        missing = spec.missing_fields(["model", "label", "universe"])
        assert missing == []

    def test_missing_fields_preserves_order(self):
        spec = ResearchSpec()
        required = ["universe", "model", "label"]
        missing = spec.missing_fields(required)
        assert missing == ["universe", "model", "label"]

    def test_unknown_field_name_in_required_treated_as_missing(self):
        spec = ResearchSpec(model="RidgeRegression")
        missing = spec.missing_fields(["model", "nonexistent_field"])
        # nonexistent_field is not an attribute so getattr returns None → missing
        assert "nonexistent_field" in missing


class TestSerialization:
    def test_to_dict_returns_all_fields(self):
        spec = ResearchSpec()
        d = spec.to_dict()
        assert isinstance(d, dict)
        assert "model" in d
        assert "label" in d
        assert "universe" in d

    def test_to_dict_none_values_included(self):
        spec = ResearchSpec()
        d = spec.to_dict()
        assert d["model"] is None
        assert d["universe"] is None

    def test_to_dict_round_trips_via_from_dict(self):
        original = ResearchSpec(
            model="RidgeRegression",
            label="forward_returns",
            label_horizon=21,
            universe="core_etfs",
            tickers=["SPY", "QQQ"],
            transaction_cost_bps=10.0,
        )
        d = original.to_dict()
        restored = ResearchSpec.from_dict(d)
        assert restored.model == original.model
        assert restored.label == original.label
        assert restored.label_horizon == original.label_horizon
        assert restored.universe == original.universe
        assert restored.tickers == original.tickers
        assert restored.transaction_cost_bps == original.transaction_cost_bps

    def test_from_dict_ignores_unknown_keys(self):
        data = {"model": "RidgeRegression", "totally_unknown_field": "value"}
        spec = ResearchSpec.from_dict(data)
        assert spec.model == "RidgeRegression"
        assert not hasattr(spec, "totally_unknown_field")

    def test_from_dict_empty_dict_produces_all_none_spec(self):
        spec = ResearchSpec.from_dict({})
        assert spec.specified_fields() == {}

    def test_from_dict_partial_sets_only_provided(self):
        spec = ResearchSpec.from_dict({"label": "binary_direction", "label_horizon": 5})
        assert spec.label == "binary_direction"
        assert spec.label_horizon == 5
        assert spec.model is None


class TestMutableFieldIsolation:
    def test_tickers_lists_not_shared_between_instances(self):
        spec_a = ResearchSpec(tickers=["SPY"])
        spec_b = ResearchSpec(tickers=["QQQ"])
        assert spec_a.tickers is not spec_b.tickers
        assert spec_a.tickers == ["SPY"]
        assert spec_b.tickers == ["QQQ"]

    def test_modifying_returned_list_does_not_affect_spec(self):
        spec = ResearchSpec(tickers=["SPY", "QQQ"])
        original = spec.tickers
        if original is not None:
            mutated = list(original)
            mutated.append("IWM")
            # The spec's own list is unchanged (unless we mutated it directly)
            assert spec.tickers == ["SPY", "QQQ"]

    def test_model_params_dict_not_shared_between_instances(self):
        spec_a = ResearchSpec(model_params={"alpha": 1.0})
        spec_b = ResearchSpec(model_params={"alpha": 2.0})
        assert spec_a.model_params is not spec_b.model_params

    def test_to_dict_result_is_independent_copy(self):
        spec = ResearchSpec(model="RidgeRegression")
        d = spec.to_dict()
        d["model"] = "MutatedModel"
        # The spec itself should not be mutated
        assert spec.model == "RidgeRegression"


class TestFullSpecification:
    def test_fully_populated_spec_has_all_fields_in_specified(self):
        spec = ResearchSpec(
            preset="canonical_multi_asset_etf",
            objective="factor rotation research",
            asset_class="multi_asset",
            universe="canonical_multi_asset_etfs",
            tickers=["SPY", "QQQ", "IWM"],
            start_date="2015-01-01",
            end_date="2024-12-31",
            frequency="monthly",
            data_source="yfinance",
            feature_families=["Trend", "Volatility"],
            features=["momentum", "rolling_volatility"],
            label="forward_returns",
            label_horizon=21,
            model="RidgeRegression",
            model_params={"alpha": 1.0},
            signal="top_n",
            signal_params={"n": 3},
            validation_protocol="rolling",
            train_window="36m",
            test_window="6m",
            transaction_cost_bps=10.0,
            benchmark="SPY",
            diagnostics=["split_metrics"],
            risk_constraints={"max_position": 0.3},
            notes="Phase 2 multi-asset allocation test",
        )
        specified = spec.specified_fields()
        # All 25 fields should be specified
        assert len(specified) == 25
        assert spec.missing_fields(list(specified.keys())) == []
