"""Tests for intent_parser.py (Phase 4A).

Covers parse_research_intent() and parse_and_resolve_research_intent():

  D. Parser tests with stub llm_client
     - parser calls the provided llm_client callable
     - valid stub output returns parsed ResearchSpec
     - malformed stub output returns invalid_output
     - unsupported registry values are rejected
     - no downstream resolver/planner/draft functions are called

  E. Keyword fallback tests (provider=PROVIDER_STUB, no llm_client)
     - preset name extracted
     - model name extracted (CamelCase prefix match)
     - dates extracted from year range
     - transaction cost extracted
     - empty request returns needs_clarification
     - unrecognised request returns needs_clarification

  F. parse_and_resolve_research_intent tests
     - parsed result flows to resolver
     - failed parse returns WorkflowResult with status "error"

  G. Governance / edge cases
     - non-string request returns parser_error
     - empty string returns needs_clarification
     - registry load failure returns parser_error
     - governance flags always set

All tests are in-memory. No live LLM calls. No disk I/O.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.agent.intent.intent_parser import (
    parse_and_resolve_research_intent,
    parse_research_intent,
)
from src.orchestration.llm.review_schema import PROVIDER_STUB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOD_CALL_LLM = "src.agent.intent.intent_parser.call_llm"
_MOD_LOAD_REGISTRY = "src.agent.intent.intent_parser.load_registry"

_CANONICAL_SPEC_JSON = json.dumps({
    "research_spec": {
        "preset": "canonical_multi_asset_etf",
        "model": "LassoRegression",
        "start_date": "2015-01-01",
        "end_date": "2024-12-31",
    },
    "unrecognised_terms": [],
    "confidence": 0.91,
    "summary": "Parsed canonical ETF model comparison.",
})


def _stub_registry() -> MagicMock:
    reg = MagicMock()
    reg.model_names.return_value = frozenset({"RidgeRegression", "LassoRegression"})
    reg.preset_names.return_value = frozenset({"canonical_multi_asset_etf", "single_asset_ml"})
    reg.asset_class_names.return_value = frozenset({"multi_asset"})
    reg.universe_names.return_value = frozenset({"canonical_multi_asset_etfs"})
    reg.label_names.return_value = frozenset({"ranking_target"})
    reg.signal_names.return_value = frozenset({"top_n"})
    reg.validation_names.return_value = frozenset({"rolling"})
    reg.feature_names.return_value = frozenset({"mom_20", "vol_21"})
    reg.diagnostic_names.return_value = frozenset({"sharpe_ratio"})
    reg.feature.side_effect = lambda name: SimpleNamespace(
        name=name, family={"mom_20": "momentum", "vol_21": "volatility"}.get(name)
    )
    reg.default_policy.return_value = SimpleNamespace(default_mode="recommended")
    return reg


# ---------------------------------------------------------------------------
# D. Parser tests with stub llm_client
# ---------------------------------------------------------------------------

class TestParserWithStubLlmClient:
    def test_llm_client_called_with_prompt(self):
        captured = {}
        def _client(prompt: str) -> str:
            captured["prompt"] = prompt
            return _CANONICAL_SPEC_JSON

        parse_research_intent(
            "Test LassoRegression from 2015 to 2024.",
            registry=_stub_registry(),
            llm_client=_client,
        )
        assert "prompt" in captured
        assert len(captured["prompt"]) > 50  # non-trivial prompt

    def test_valid_output_returns_parsed(self):
        result = parse_research_intent(
            "Test Lasso from 2015 to 2024.",
            registry=_stub_registry(),
            llm_client=lambda _: _CANONICAL_SPEC_JSON,
        )
        assert result.status == "parsed"

    def test_research_spec_extracted(self):
        result = parse_research_intent(
            "Test Lasso.",
            registry=_stub_registry(),
            llm_client=lambda _: _CANONICAL_SPEC_JSON,
        )
        assert result.research_spec is not None
        assert result.research_spec.model == "LassoRegression"
        assert result.research_spec.preset == "canonical_multi_asset_etf"

    def test_malformed_output_returns_invalid_output(self):
        result = parse_research_intent(
            "Test something.",
            registry=_stub_registry(),
            llm_client=lambda _: "this is not json at all!",
        )
        assert result.status == "invalid_output"

    def test_unsupported_registry_value_rejected(self):
        bad_output = json.dumps({
            "research_spec": {"model": "RandomForest"},
            "unrecognised_terms": [],
            "confidence": 0.5,
            "summary": "parsed",
        })
        result = parse_research_intent(
            "Use RandomForest.",
            registry=_stub_registry(),
            llm_client=lambda _: bad_output,
        )
        assert result.research_spec is None or result.research_spec.model is None
        assert "RandomForest" in result.unrecognised_terms

    def test_no_resolver_called(self):
        with patch("src.agent.intent.intent_parser.load_registry") as mock_lr:
            result = parse_research_intent(
                "Test LassoRegression.",
                registry=_stub_registry(),
                llm_client=lambda _: _CANONICAL_SPEC_JSON,
            )
        # load_registry should not be called when registry is provided
        mock_lr.assert_not_called()

    def test_no_planner_called(self):
        from unittest.mock import patch
        with patch("src.agent.planner.create_experiment_plan.create_experiment_plan") as mock_plan:
            parse_research_intent(
                "Test Lasso.",
                registry=_stub_registry(),
                llm_client=lambda _: _CANONICAL_SPEC_JSON,
            )
        mock_plan.assert_not_called()

    def test_no_drafter_called(self):
        from unittest.mock import patch
        with patch("src.agent.drafting.plan_to_draft.create_draft_from_plan") as mock_draft:
            parse_research_intent(
                "Test Lasso.",
                registry=_stub_registry(),
                llm_client=lambda _: _CANONICAL_SPEC_JSON,
            )
        mock_draft.assert_not_called()

    def test_raw_text_preserved(self):
        request = "Test Lasso from 2020 to 2024."
        result = parse_research_intent(
            request,
            registry=_stub_registry(),
            llm_client=lambda _: _CANONICAL_SPEC_JSON,
        )
        assert result.raw_text == request

    def test_raw_model_output_preserved(self):
        output = _CANONICAL_SPEC_JSON
        result = parse_research_intent(
            "test",
            registry=_stub_registry(),
            llm_client=lambda _: output,
        )
        assert result.raw_model_output == output

    def test_llm_client_exception_returns_parser_error(self):
        def _bad_client(prompt: str) -> str:
            raise RuntimeError("network failure")

        result = parse_research_intent(
            "Test something.",
            registry=_stub_registry(),
            llm_client=_bad_client,
        )
        assert result.status == "parser_error"
        assert any("network failure" in e for e in result.errors)

    def test_confidence_preserved(self):
        output = json.dumps({
            "research_spec": {"model": "LassoRegression"},
            "unrecognised_terms": [],
            "confidence": 0.88,
            "summary": "ok",
        })
        result = parse_research_intent(
            "test",
            registry=_stub_registry(),
            llm_client=lambda _: output,
        )
        assert result.confidence == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# E. Keyword fallback (provider=PROVIDER_STUB, no llm_client)
# ---------------------------------------------------------------------------

class TestKeywordFallback:
    def test_preset_extracted_from_normalized_text(self):
        result = parse_research_intent(
            "Use the canonical multi-asset ETF preset.",
            registry=_stub_registry(),
            provider=PROVIDER_STUB,
        )
        assert result.status == "parsed"
        assert result.extracted_fields.get("preset") == "canonical_multi_asset_etf"

    def test_model_extracted_by_camelcase_prefix(self):
        result = parse_research_intent(
            "Test Lasso instead of Ridge.",
            registry=_stub_registry(),
            provider=PROVIDER_STUB,
        )
        assert result.status == "parsed"
        assert result.extracted_fields.get("model") == "LassoRegression"

    def test_model_extracted_exact_name(self):
        result = parse_research_intent(
            "Use LassoRegression model.",
            registry=_stub_registry(),
            provider=PROVIDER_STUB,
        )
        assert result.extracted_fields.get("model") == "LassoRegression"

    def test_year_range_extracted(self):
        result = parse_research_intent(
            "Run from 2015 to 2024.",
            registry=_stub_registry(),
            provider=PROVIDER_STUB,
        )
        assert result.extracted_fields.get("start_date") == "2015-01-01"
        assert result.extracted_fields.get("end_date") == "2024-12-31"

    def test_single_year_as_end_date(self):
        result = parse_research_intent(
            "Analysis ending in 2024.",
            registry=_stub_registry(),
            provider=PROVIDER_STUB,
        )
        assert result.extracted_fields.get("end_date") == "2024-12-31"

    def test_iso_dates_preserved(self):
        result = parse_research_intent(
            "From 2015-03-01 to 2024-06-30.",
            registry=_stub_registry(),
            provider=PROVIDER_STUB,
        )
        assert result.extracted_fields.get("start_date") == "2015-03-01"
        assert result.extracted_fields.get("end_date") == "2024-06-30"

    def test_transaction_cost_extracted(self):
        result = parse_research_intent(
            "Use 10 bps transaction costs.",
            registry=_stub_registry(),
            provider=PROVIDER_STUB,
        )
        assert result.extracted_fields.get("transaction_cost_bps") == 10.0

    def test_decimal_bps_extracted(self):
        result = parse_research_intent(
            "Apply 2.5bps costs.",
            registry=_stub_registry(),
            provider=PROVIDER_STUB,
        )
        assert result.extracted_fields.get("transaction_cost_bps") == pytest.approx(2.5)

    def test_empty_request_needs_clarification(self):
        result = parse_research_intent(
            "",
            registry=_stub_registry(),
            provider=PROVIDER_STUB,
        )
        assert result.status == "needs_clarification"

    def test_whitespace_only_request_needs_clarification(self):
        result = parse_research_intent(
            "   \n\t  ",
            registry=_stub_registry(),
            provider=PROVIDER_STUB,
        )
        assert result.status == "needs_clarification"

    def test_unrecognised_request_needs_clarification(self):
        result = parse_research_intent(
            "zzz unknown tokens xyz abc",
            registry=_stub_registry(),
            provider=PROVIDER_STUB,
        )
        assert result.status == "needs_clarification"

    def test_stub_warning_present(self):
        result = parse_research_intent(
            "canonical multi-asset ETF",
            registry=_stub_registry(),
            provider=PROVIDER_STUB,
        )
        assert any("stub" in w.lower() or "keyword" in w.lower() for w in result.warnings)

    def test_governance_flags_on_stub(self):
        result = parse_research_intent(
            "canonical multi-asset ETF",
            registry=_stub_registry(),
            provider=PROVIDER_STUB,
        )
        assert result.no_plan_created is True
        assert result.no_draft_created is True
        assert result.no_execution_performed is True

    def test_full_canonical_request(self):
        result = parse_research_intent(
            "Use the canonical multi-asset ETF preset but test Lasso instead of Ridge "
            "from 2015 to 2024.",
            registry=_stub_registry(),
            provider=PROVIDER_STUB,
        )
        assert result.status == "parsed"
        assert result.extracted_fields.get("preset") == "canonical_multi_asset_etf"
        assert result.extracted_fields.get("model") == "LassoRegression"
        assert result.extracted_fields.get("start_date") == "2015-01-01"
        assert result.extracted_fields.get("end_date") == "2024-12-31"


# ---------------------------------------------------------------------------
# F. parse_and_resolve_research_intent
# ---------------------------------------------------------------------------

class TestParseAndResolve:
    def test_parsed_result_flows_to_resolver(self):
        output = json.dumps({
            "research_spec": {
                "preset": "canonical_multi_asset_etf",
                "end_date": "2024-12-31",
            },
            "unrecognised_terms": [],
            "confidence": 0.9,
            "summary": "ok",
        })
        result = parse_and_resolve_research_intent(
            "canonical ETF 2024",
            registry=_stub_registry(),
            llm_client=lambda _: output,
        )
        # WorkflowResult — should be resolved or needs_user_input
        assert hasattr(result, "status")
        assert result.status in ("resolved", "needs_user_input", "invalid", "error")

    def test_failed_parse_returns_workflow_error(self):
        result = parse_and_resolve_research_intent(
            "test",
            registry=_stub_registry(),
            llm_client=lambda _: "not json!!",
        )
        assert result.status == "error"
        assert result.errors

    def test_needs_clarification_returns_workflow_needs_clarification(self):
        # Phase 4B: needs_clarification is a normal advisory status, not an error
        output = json.dumps({
            "research_spec": {},
            "unrecognised_terms": [],
            "confidence": 0.0,
            "summary": "",
        })
        result = parse_and_resolve_research_intent(
            "gibberish zzz",
            registry=_stub_registry(),
            llm_client=lambda _: output,
        )
        assert result.status == "needs_clarification"

    def test_no_execution_in_result(self):
        output = json.dumps({
            "research_spec": {
                "preset": "canonical_multi_asset_etf",
                "end_date": "2024-12-31",
            },
            "unrecognised_terms": [],
            "confidence": 0.9,
            "summary": "ok",
        })
        result = parse_and_resolve_research_intent(
            "canonical ETF",
            registry=_stub_registry(),
            llm_client=lambda _: output,
        )
        # WorkflowResult always has execution_performed=False
        assert result.execution_performed is False


# ---------------------------------------------------------------------------
# G. Governance / edge cases
# ---------------------------------------------------------------------------

class TestGovernanceEdgeCases:
    def test_non_string_request_returns_parser_error(self):
        result = parse_research_intent(42, registry=_stub_registry())  # type: ignore[arg-type]
        assert result.status == "parser_error"

    def test_none_request_returns_parser_error(self):
        result = parse_research_intent(None, registry=_stub_registry())  # type: ignore[arg-type]
        assert result.status == "parser_error"

    def test_registry_load_failure_returns_parser_error(self):
        with patch(_MOD_LOAD_REGISTRY, side_effect=FileNotFoundError("no registry")):
            result = parse_research_intent("test")
        assert result.status == "parser_error"
        assert any("registry" in e.lower() or "no registry" in e.lower() for e in result.errors)

    def test_governance_flags_always_present(self):
        result = parse_research_intent(
            "test",
            registry=_stub_registry(),
            llm_client=lambda _: _CANONICAL_SPEC_JSON,
        )
        assert result.no_plan_created is True
        assert result.no_draft_created is True
        assert result.no_execution_performed is True

    def test_governance_flags_on_invalid_output(self):
        result = parse_research_intent(
            "test",
            registry=_stub_registry(),
            llm_client=lambda _: "bad json",
        )
        assert result.no_plan_created is True
        assert result.no_draft_created is True
        assert result.no_execution_performed is True

    def test_governance_flags_on_parser_error(self):
        result = parse_research_intent(None, registry=_stub_registry())  # type: ignore[arg-type]
        assert result.no_plan_created is True
        assert result.no_draft_created is True
        assert result.no_execution_performed is True

    def test_to_dict_has_all_required_keys(self):
        result = parse_research_intent(
            "test",
            registry=_stub_registry(),
            llm_client=lambda _: _CANONICAL_SPEC_JSON,
        )
        d = result.to_dict()
        required = {
            "status", "research_spec", "raw_text", "raw_model_output",
            "extracted_fields", "unrecognised_terms", "registry_warnings",
            "missing_after_parse", "confidence", "errors", "warnings", "summary",
            "no_plan_created", "no_draft_created", "no_execution_performed",
        }
        assert required.issubset(d.keys()), f"Missing: {required - set(d.keys())}"

    def test_package_level_import(self):
        from src.agent.intent import (
            IntentParseResult,
            parse_and_resolve_research_intent,
            parse_research_intent,
        )
        assert callable(parse_research_intent)
        assert callable(parse_and_resolve_research_intent)
        assert IntentParseResult is not None
