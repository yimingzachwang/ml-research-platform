"""Tests for src/agent/resolver/missing_fields.py"""

from __future__ import annotations

import pytest

from src.agent.intent.research_spec import ResearchSpec
from src.agent.registry.registry_loader import load_registry
from src.agent.resolver.missing_fields import (
    MissingFieldReport,
    _DEFAULT_REQUIRED_FIELDS,
    analyse_spec_completeness,
    detect_missing_fields,
)


@pytest.fixture(scope="module")
def registry():
    return load_registry()


class TestDetectMissingFields:
    def test_empty_spec_returns_all_required_fields(self):
        spec = ResearchSpec()
        missing = detect_missing_fields(spec)
        for f in _DEFAULT_REQUIRED_FIELDS:
            assert f in missing, f"Expected '{f}' in missing for empty spec"

    def test_empty_spec_includes_features_in_missing(self):
        spec = ResearchSpec()
        missing = detect_missing_fields(spec)
        assert "features" in missing

    def test_partial_spec_returns_only_remaining_missing(self):
        spec = ResearchSpec(
            model="RidgeRegression",
            label="forward_returns",
            universe="core_etfs",
        )
        missing = detect_missing_fields(spec)
        assert "model" not in missing
        assert "label" not in missing
        assert "universe" not in missing
        assert "asset_class" in missing
        assert "start_date" in missing

    def test_feature_families_satisfies_features_requirement(self):
        spec = ResearchSpec(feature_families=["Trend", "Volatility"])
        missing = detect_missing_fields(spec)
        assert "features" not in missing

    def test_features_list_satisfies_requirement(self):
        spec = ResearchSpec(features=["momentum", "rolling_volatility"])
        missing = detect_missing_fields(spec)
        assert "features" not in missing

    def test_empty_features_list_counts_as_missing(self):
        spec = ResearchSpec(features=[])
        missing = detect_missing_fields(spec)
        assert "features" in missing

    def test_custom_required_fields_override_defaults(self):
        spec = ResearchSpec(model="RidgeRegression")
        missing = detect_missing_fields(spec, required_fields=["model", "label"])
        assert "model" not in missing
        assert "label" in missing
        # Default required fields not in custom list should not appear
        assert "asset_class" not in missing

    def test_fully_specified_returns_empty_list(self):
        spec = ResearchSpec(
            asset_class="multi_asset",
            universe="core_etfs",
            start_date="2015-01-01",
            end_date="2024-12-31",
            frequency="monthly",
            features=["momentum"],
            label="forward_returns",
            model="RidgeRegression",
            signal="top_n",
            validation_protocol="rolling",
            transaction_cost_bps=10.0,
        )
        missing = detect_missing_fields(spec)
        assert missing == []

    def test_result_is_sorted(self):
        spec = ResearchSpec()
        missing = detect_missing_fields(spec)
        assert missing == sorted(missing)

    def test_none_vs_missing_field_name_distinction(self):
        spec = ResearchSpec(model=None)
        missing = detect_missing_fields(spec)
        assert "model" in missing

    def test_registry_parameter_accepted(self, registry):
        spec = ResearchSpec()
        missing = detect_missing_fields(spec, registry=registry)
        assert isinstance(missing, list)


class TestAnalyseSpecCompleteness:
    def test_returns_missing_field_report(self, registry):
        spec = ResearchSpec()
        report = analyse_spec_completeness(spec, registry)
        assert isinstance(report, MissingFieldReport)

    def test_empty_spec_completeness_is_zero(self, registry):
        spec = ResearchSpec()
        report = analyse_spec_completeness(spec, registry)
        assert report.completeness_pct == 0.0

    def test_fully_specified_spec_is_100_percent(self, registry):
        spec = ResearchSpec(
            asset_class="multi_asset",
            universe="core_etfs",
            start_date="2015-01-01",
            end_date="2024-12-31",
            frequency="monthly",
            features=["momentum"],
            label="forward_returns",
            model="RidgeRegression",
            signal="top_n",
            validation_protocol="rolling",
            transaction_cost_bps=10.0,
        )
        report = analyse_spec_completeness(spec, registry)
        assert report.completeness_pct == 100.0
        assert report.has_features is True

    def test_partial_spec_completeness_between_0_and_100(self, registry):
        spec = ResearchSpec(model="RidgeRegression", label="forward_returns")
        report = analyse_spec_completeness(spec, registry)
        assert 0.0 < report.completeness_pct < 100.0

    def test_has_features_true_when_families_set(self, registry):
        spec = ResearchSpec(feature_families=["Trend"])
        report = analyse_spec_completeness(spec, registry)
        assert report.has_features is True

    def test_missing_required_matches_detect_missing_fields(self, registry):
        spec = ResearchSpec(model="RidgeRegression")
        report = analyse_spec_completeness(spec, registry)
        direct_missing = detect_missing_fields(spec)
        assert sorted(report.missing_required) == sorted(direct_missing)

    def test_present_required_and_missing_required_are_complementary(self, registry):
        spec = ResearchSpec(
            asset_class="multi_asset",
            universe="core_etfs",
            model="RidgeRegression",
        )
        report = analyse_spec_completeness(spec, registry)
        all_rf = set(_DEFAULT_REQUIRED_FIELDS)
        present_set = set(report.present_required)
        missing_set = set(f for f in report.missing_required if f in all_rf)
        assert present_set.isdisjoint(missing_set)
        assert present_set | missing_set == all_rf

    def test_optional_fields_listed(self, registry):
        spec = ResearchSpec()
        report = analyse_spec_completeness(spec, registry)
        assert "benchmark" in report.missing_optional
        assert "notes" in report.missing_optional
