"""Tests for response_parser.py (Phase 4A).

Covers parse_intent_response() with a variety of model outputs:
  1. Valid canonical preset response → parsed
  2. Unsupported model → rejected to unrecognised_terms
  3. Malformed JSON → invalid_output
  4. Extra unsupported ResearchSpec field → warning, no crash
  5. Non-object JSON output → invalid_output
  6. Null fields → not in extracted_fields
  7. Date preservation → exact strings preserved
  8. transaction_cost_bps extraction → numeric preserved
  9. Markdown-wrapped JSON → parsed (code-fence stripping)
 10. Empty research_spec → needs_clarification
 11. Unsupported preset → rejected
 12. Unsupported label → rejected
 13. Unsupported signal → rejected
 14. Features list with mixed valid/invalid → valid kept, invalid rejected
 15. Diagnostics list with mixed → same pattern
 16. confidence preserved as float
 17. summary preserved
 18. unrecognised_terms from model output merged with rejected values
 19. research_spec missing entirely → treated as empty
 20. research_spec not a dict → invalid_output

All tests use a stub registry (MagicMock) — no disk I/O, no real registry.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.agent.intent.response_parser import parse_intent_response


# ---------------------------------------------------------------------------
# Stub registry
# ---------------------------------------------------------------------------

def _stub_registry() -> MagicMock:
    reg = MagicMock()
    reg.model_names.return_value = frozenset({"RidgeRegression", "LassoRegression"})
    reg.preset_names.return_value = frozenset({"canonical_multi_asset_etf", "single_asset_ml"})
    reg.asset_class_names.return_value = frozenset({"multi_asset", "equity_index"})
    reg.universe_names.return_value = frozenset({"canonical_multi_asset_etfs"})
    reg.label_names.return_value = frozenset({"ranking_target"})
    reg.signal_names.return_value = frozenset({"top_n", "long_short"})
    reg.validation_names.return_value = frozenset({"rolling"})
    reg.feature_names.return_value = frozenset({"mom_20", "vol_21"})
    reg.diagnostic_names.return_value = frozenset({"sharpe_ratio", "split_metrics"})
    reg.feature.side_effect = lambda name: SimpleNamespace(
        name=name, family={"mom_20": "momentum", "vol_21": "volatility"}.get(name)
    )
    return reg


def _make_output(**overrides) -> str:
    base = {
        "research_spec": {},
        "unrecognised_terms": [],
        "confidence": 0.9,
        "summary": "Parsed.",
    }
    base.update(overrides)
    return json.dumps(base)


# ---------------------------------------------------------------------------
# 1. Valid canonical preset response
# ---------------------------------------------------------------------------

class TestValidCanonicalResponse:
    def test_status_parsed(self):
        out = _make_output(research_spec={
            "preset": "canonical_multi_asset_etf",
            "start_date": "2015-01-01",
            "end_date": "2024-12-31",
            "model": "LassoRegression",
        })
        result = parse_intent_response(out, _stub_registry())
        assert result.status == "parsed"

    def test_research_spec_populated(self):
        out = _make_output(research_spec={
            "preset": "canonical_multi_asset_etf",
            "model": "LassoRegression",
            "end_date": "2024-12-31",
        })
        result = parse_intent_response(out, _stub_registry())
        assert result.research_spec is not None
        assert result.research_spec.preset == "canonical_multi_asset_etf"
        assert result.research_spec.model == "LassoRegression"

    def test_extracted_fields_populated(self):
        out = _make_output(research_spec={
            "preset": "canonical_multi_asset_etf",
            "end_date": "2024-12-31",
        })
        result = parse_intent_response(out, _stub_registry())
        assert "preset" in result.extracted_fields
        assert "end_date" in result.extracted_fields

    def test_no_registry_warnings(self):
        out = _make_output(research_spec={
            "preset": "canonical_multi_asset_etf",
            "model": "RidgeRegression",
        })
        result = parse_intent_response(out, _stub_registry())
        assert result.registry_warnings == []

    def test_raw_output_preserved(self):
        out = _make_output(research_spec={"model": "LassoRegression"})
        result = parse_intent_response(out, _stub_registry())
        assert result.raw_model_output == out


# ---------------------------------------------------------------------------
# 2. Unsupported model
# ---------------------------------------------------------------------------

class TestUnsupportedModel:
    def test_status_parsed_not_invalid(self):
        out = _make_output(research_spec={"model": "RandomForest"})
        result = parse_intent_response(out, _stub_registry())
        # Status is not "invalid_output" because the overall JSON is valid.
        # With no valid fields, it may be "needs_clarification"
        assert result.status in ("parsed", "needs_clarification")

    def test_model_not_in_research_spec(self):
        out = _make_output(research_spec={"model": "RandomForest"})
        result = parse_intent_response(out, _stub_registry())
        assert result.research_spec is None or result.research_spec.model is None

    def test_model_in_unrecognised_terms(self):
        out = _make_output(research_spec={"model": "RandomForest"})
        result = parse_intent_response(out, _stub_registry())
        assert "RandomForest" in result.unrecognised_terms

    def test_registry_warning_mentions_model(self):
        out = _make_output(research_spec={"model": "RandomForest"})
        result = parse_intent_response(out, _stub_registry())
        assert any("RandomForest" in w for w in result.registry_warnings)

    def test_model_not_in_extracted_fields(self):
        out = _make_output(research_spec={"model": "RandomForest"})
        result = parse_intent_response(out, _stub_registry())
        assert "model" not in result.extracted_fields


# ---------------------------------------------------------------------------
# 3. Malformed JSON
# ---------------------------------------------------------------------------

class TestMalformedJson:
    def test_status_invalid_output(self):
        result = parse_intent_response("this is not json!!", _stub_registry())
        assert result.status == "invalid_output"

    def test_no_research_spec(self):
        result = parse_intent_response("not json", _stub_registry())
        assert result.research_spec is None

    def test_raw_output_preserved(self):
        raw = "definitely not json {{"
        result = parse_intent_response(raw, _stub_registry())
        assert result.raw_model_output == raw

    def test_errors_populated(self):
        result = parse_intent_response("bad input", _stub_registry())
        assert result.errors


# ---------------------------------------------------------------------------
# 4. Extra unsupported ResearchSpec field
# ---------------------------------------------------------------------------

class TestExtraUnsupportedField:
    def test_no_crash(self):
        out = _make_output(research_spec={
            "model": "LassoRegression",
            "unknown_field_xyz": "some_value",
        })
        result = parse_intent_response(out, _stub_registry())
        # Should not raise
        assert result.status in ("parsed", "needs_clarification", "invalid_output")

    def test_unknown_field_not_in_extracted_fields(self):
        out = _make_output(research_spec={
            "model": "LassoRegression",
            "totally_unknown_field": "value",
        })
        result = parse_intent_response(out, _stub_registry())
        assert "totally_unknown_field" not in result.extracted_fields

    def test_warning_about_unknown_field(self):
        out = _make_output(research_spec={
            "model": "LassoRegression",
            "made_up_field": "value",
        })
        result = parse_intent_response(out, _stub_registry())
        assert any("made_up_field" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# 5. Non-object JSON output
# ---------------------------------------------------------------------------

class TestNonObjectJson:
    def test_list_output_invalid(self):
        result = parse_intent_response(json.dumps([1, 2, 3]), _stub_registry())
        assert result.status == "invalid_output"

    def test_string_output_invalid(self):
        result = parse_intent_response(json.dumps("just a string"), _stub_registry())
        assert result.status == "invalid_output"

    def test_number_output_invalid(self):
        result = parse_intent_response(json.dumps(42), _stub_registry())
        assert result.status == "invalid_output"

    def test_null_output_invalid(self):
        result = parse_intent_response(json.dumps(None), _stub_registry())
        assert result.status == "invalid_output"


# ---------------------------------------------------------------------------
# 6. Null fields
# ---------------------------------------------------------------------------

class TestNullFields:
    def test_null_model_not_in_extracted_fields(self):
        out = _make_output(research_spec={"model": None, "preset": "canonical_multi_asset_etf"})
        result = parse_intent_response(out, _stub_registry())
        assert "model" not in result.extracted_fields

    def test_null_spec_treated_as_empty(self):
        out = _make_output(research_spec=None)
        result = parse_intent_response(out, _stub_registry())
        # Null spec → same as empty spec → needs_clarification or parsed with empty
        assert result.status in ("parsed", "needs_clarification")
        assert result.extracted_fields == {}

    def test_all_null_fields_is_needs_clarification(self):
        spec = {k: None for k in [
            "preset", "model", "start_date", "end_date", "label", "signal"
        ]}
        out = _make_output(research_spec=spec)
        result = parse_intent_response(out, _stub_registry())
        assert result.status == "needs_clarification"
        assert result.extracted_fields == {}


# ---------------------------------------------------------------------------
# 7. Date preservation
# ---------------------------------------------------------------------------

class TestDatePreservation:
    def test_start_date_preserved(self):
        out = _make_output(research_spec={"start_date": "2015-01-01"})
        result = parse_intent_response(out, _stub_registry())
        assert result.extracted_fields.get("start_date") == "2015-01-01"

    def test_end_date_preserved(self):
        out = _make_output(research_spec={"end_date": "2024-12-31"})
        result = parse_intent_response(out, _stub_registry())
        assert result.extracted_fields.get("end_date") == "2024-12-31"

    def test_dates_in_research_spec(self):
        out = _make_output(research_spec={
            "start_date": "2015-01-01",
            "end_date": "2024-12-31",
        })
        result = parse_intent_response(out, _stub_registry())
        assert result.research_spec is not None
        assert result.research_spec.start_date == "2015-01-01"
        assert result.research_spec.end_date == "2024-12-31"


# ---------------------------------------------------------------------------
# 8. Transaction cost extraction
# ---------------------------------------------------------------------------

class TestTransactionCostExtraction:
    def test_float_preserved(self):
        out = _make_output(research_spec={"transaction_cost_bps": 5.0})
        result = parse_intent_response(out, _stub_registry())
        assert result.extracted_fields.get("transaction_cost_bps") == 5.0

    def test_int_preserved_as_numeric(self):
        out = _make_output(research_spec={"transaction_cost_bps": 10})
        result = parse_intent_response(out, _stub_registry())
        assert result.extracted_fields.get("transaction_cost_bps") == 10

    def test_in_research_spec(self):
        out = _make_output(research_spec={"transaction_cost_bps": 7.5})
        result = parse_intent_response(out, _stub_registry())
        assert result.research_spec is not None
        assert result.research_spec.transaction_cost_bps == 7.5


# ---------------------------------------------------------------------------
# 9. Markdown-wrapped JSON (code fence stripping)
# ---------------------------------------------------------------------------

class TestMarkdownWrappedJson:
    def test_code_fence_stripped_and_parsed(self):
        inner = json.dumps({
            "research_spec": {"model": "LassoRegression"},
            "unrecognised_terms": [],
            "confidence": 0.8,
            "summary": "Parsed.",
        })
        wrapped = f"```json\n{inner}\n```"
        result = parse_intent_response(wrapped, _stub_registry())
        assert result.status == "parsed"
        assert result.extracted_fields.get("model") == "LassoRegression"

    def test_plain_code_fence_stripped(self):
        inner = json.dumps({
            "research_spec": {"preset": "canonical_multi_asset_etf"},
            "unrecognised_terms": [],
            "confidence": 0.7,
            "summary": "ok",
        })
        wrapped = f"```\n{inner}\n```"
        result = parse_intent_response(wrapped, _stub_registry())
        assert result.status == "parsed"


# ---------------------------------------------------------------------------
# 10. Empty research_spec → needs_clarification
# ---------------------------------------------------------------------------

class TestEmptyResearchSpec:
    def test_empty_spec_needs_clarification(self):
        out = _make_output(research_spec={})
        result = parse_intent_response(out, _stub_registry())
        assert result.status == "needs_clarification"

    def test_research_spec_object_still_created(self):
        out = _make_output(research_spec={})
        result = parse_intent_response(out, _stub_registry())
        assert result.research_spec is not None


# ---------------------------------------------------------------------------
# 11–13. Unsupported preset, label, signal
# ---------------------------------------------------------------------------

class TestUnsupportedSingleValues:
    def test_unsupported_preset_rejected(self):
        out = _make_output(research_spec={"preset": "my_custom_preset_xyz"})
        result = parse_intent_response(out, _stub_registry())
        assert result.research_spec is None or result.research_spec.preset is None
        assert "my_custom_preset_xyz" in result.unrecognised_terms

    def test_unsupported_label_rejected(self):
        out = _make_output(research_spec={"label": "unknown_label"})
        result = parse_intent_response(out, _stub_registry())
        assert result.research_spec is None or result.research_spec.label is None
        assert "unknown_label" in result.unrecognised_terms

    def test_unsupported_signal_rejected(self):
        out = _make_output(research_spec={"signal": "unknown_signal"})
        result = parse_intent_response(out, _stub_registry())
        assert result.research_spec is None or result.research_spec.signal is None
        assert "unknown_signal" in result.unrecognised_terms


# ---------------------------------------------------------------------------
# 14–15. Mixed valid/invalid list fields
# ---------------------------------------------------------------------------

class TestMixedListFields:
    def test_features_valid_kept_invalid_rejected(self):
        out = _make_output(research_spec={
            "features": ["mom_20", "RandomForestFeature", "vol_21"]
        })
        result = parse_intent_response(out, _stub_registry())
        if result.extracted_fields.get("features"):
            assert "mom_20" in result.extracted_fields["features"]
            assert "RandomForestFeature" not in result.extracted_fields["features"]
        assert "RandomForestFeature" in result.unrecognised_terms

    def test_diagnostics_valid_kept_invalid_rejected(self):
        out = _make_output(research_spec={
            "diagnostics": ["sharpe_ratio", "unknown_diag"]
        })
        result = parse_intent_response(out, _stub_registry())
        if result.extracted_fields.get("diagnostics"):
            assert "sharpe_ratio" in result.extracted_fields["diagnostics"]
            assert "unknown_diag" not in result.extracted_fields["diagnostics"]
        assert "unknown_diag" in result.unrecognised_terms

    def test_all_features_invalid_field_not_set(self):
        out = _make_output(research_spec={"features": ["bad1", "bad2"]})
        result = parse_intent_response(out, _stub_registry())
        # All rejected → field should not appear
        assert result.extracted_fields.get("features") is None


# ---------------------------------------------------------------------------
# 16–17. Confidence and summary
# ---------------------------------------------------------------------------

class TestConfidenceAndSummary:
    def test_confidence_preserved(self):
        out = _make_output(
            research_spec={"model": "LassoRegression"},
            confidence=0.91,
        )
        result = parse_intent_response(out, _stub_registry())
        assert result.confidence == pytest.approx(0.91)

    def test_summary_preserved(self):
        out = _make_output(
            research_spec={"model": "LassoRegression"},
            summary="Parsed canonical ETF model comparison.",
        )
        result = parse_intent_response(out, _stub_registry())
        assert "Parsed canonical ETF" in result.summary

    def test_missing_confidence_is_none(self):
        data = {
            "research_spec": {"model": "LassoRegression"},
            "unrecognised_terms": [],
            "summary": "ok",
        }
        result = parse_intent_response(json.dumps(data), _stub_registry())
        assert result.confidence is None


# ---------------------------------------------------------------------------
# 18. unrecognised_terms from model output merged with rejected values
# ---------------------------------------------------------------------------

class TestUnrecognisedTermsMerge:
    def test_model_unrecognised_merged(self):
        out = _make_output(
            research_spec={"model": "RandomForest"},
            unrecognised_terms=["some_other_thing"],
        )
        result = parse_intent_response(out, _stub_registry())
        assert "RandomForest" in result.unrecognised_terms
        assert "some_other_thing" in result.unrecognised_terms

    def test_no_duplicates_in_unrecognised(self):
        out = _make_output(
            research_spec={"model": "RandomForest"},
            unrecognised_terms=["RandomForest"],  # same term
        )
        result = parse_intent_response(out, _stub_registry())
        assert result.unrecognised_terms.count("RandomForest") == 1


# ---------------------------------------------------------------------------
# 19. research_spec key absent
# ---------------------------------------------------------------------------

class TestResearchSpecAbsent:
    def test_absent_key_parsed_as_empty(self):
        data = {"unrecognised_terms": [], "confidence": 0.5, "summary": "ok"}
        result = parse_intent_response(json.dumps(data), _stub_registry())
        assert result.status in ("parsed", "needs_clarification")
        assert result.extracted_fields == {}

    def test_research_spec_created_even_when_absent(self):
        data = {"confidence": 0.0, "summary": "nothing"}
        result = parse_intent_response(json.dumps(data), _stub_registry())
        assert result.research_spec is not None


# ---------------------------------------------------------------------------
# 20. research_spec not a dict
# ---------------------------------------------------------------------------

class TestResearchSpecNotDict:
    def test_list_spec_invalid(self):
        data = {"research_spec": [1, 2, 3], "confidence": 0.5, "summary": ""}
        result = parse_intent_response(json.dumps(data), _stub_registry())
        assert result.status == "invalid_output"

    def test_string_spec_invalid(self):
        data = {"research_spec": "not a dict", "confidence": 0.5, "summary": ""}
        result = parse_intent_response(json.dumps(data), _stub_registry())
        assert result.status == "invalid_output"


# ---------------------------------------------------------------------------
# Governance flags
# ---------------------------------------------------------------------------

class TestGovernanceFlags:
    def test_governance_flags_always_set(self):
        out = _make_output(research_spec={"model": "LassoRegression"})
        result = parse_intent_response(out, _stub_registry())
        assert result.no_plan_created is True
        assert result.no_draft_created is True
        assert result.no_execution_performed is True

    def test_governance_in_to_dict(self):
        out = _make_output(research_spec={"model": "LassoRegression"})
        result = parse_intent_response(out, _stub_registry())
        d = result.to_dict()
        assert d["no_plan_created"] is True
        assert d["no_draft_created"] is True
        assert d["no_execution_performed"] is True
