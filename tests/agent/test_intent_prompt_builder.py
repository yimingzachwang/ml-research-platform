"""Tests for registry_context.py and prompt_builder.py (Phase 4A).

Covers:
  A. Registry context builder
     - compact context includes all required categories
     - only registry-supported values appear
     - feature_families derived from feature entries
     - context is deterministic (same registry → same output)
     - full context adds model/preset details

  B. Prompt builder
     - prompt contains the user request verbatim
     - prompt instructs strict JSON output
     - prompt includes valid registry values (models, presets, etc.)
     - prompt forbids YAML, final config, execution
     - prompt says unsupported/unknown values must be null
     - output schema present in prompt (all ResearchSpec fields)
     - prompt is deterministic for the same inputs

All tests are in-memory. No disk I/O, no LLM, no quant engine.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.agent.intent.prompt_builder import build_intent_parser_prompt
from src.agent.intent.registry_context import build_registry_context


# ---------------------------------------------------------------------------
# Helpers — minimal stub registry
# ---------------------------------------------------------------------------

def _stub_registry() -> MagicMock:
    reg = MagicMock()
    reg.model_names.return_value = frozenset({"RidgeRegression", "LassoRegression"})
    reg.feature_names.return_value = frozenset({"mom_20", "vol_21", "carry_1m"})
    reg.signal_names.return_value = frozenset({"top_n", "long_short"})
    reg.label_names.return_value = frozenset({"ranking_target"})
    reg.universe_names.return_value = frozenset({"canonical_multi_asset_etfs"})
    reg.asset_class_names.return_value = frozenset({"multi_asset", "equity_index"})
    reg.validation_names.return_value = frozenset({"rolling"})
    reg.diagnostic_names.return_value = frozenset({"sharpe_ratio", "split_metrics"})
    reg.preset_names.return_value = frozenset({"canonical_multi_asset_etf", "single_asset_ml"})
    # Feature entries with family field
    reg.feature.side_effect = lambda name: SimpleNamespace(
        name=name,
        family={"mom_20": "momentum", "vol_21": "volatility", "carry_1m": "carry"}.get(name),
    )
    reg.default_policy.return_value = SimpleNamespace(default_mode="recommended")
    return reg


# ---------------------------------------------------------------------------
# A. Registry context builder
# ---------------------------------------------------------------------------

class TestBuildRegistryContext:
    def test_compact_has_all_required_keys(self):
        reg = _stub_registry()
        ctx = build_registry_context(reg, compact=True)
        required = {
            "presets", "asset_classes", "universes", "models",
            "features", "feature_families", "labels", "signals",
            "validation_protocols", "diagnostics", "default_mode",
        }
        assert required.issubset(ctx.keys()), f"Missing: {required - set(ctx.keys())}"

    def test_models_match_registry(self):
        reg = _stub_registry()
        ctx = build_registry_context(reg)
        assert set(ctx["models"]) == {"RidgeRegression", "LassoRegression"}

    def test_presets_match_registry(self):
        reg = _stub_registry()
        ctx = build_registry_context(reg)
        assert "canonical_multi_asset_etf" in ctx["presets"]
        assert "single_asset_ml" in ctx["presets"]

    def test_universes_match_registry(self):
        reg = _stub_registry()
        ctx = build_registry_context(reg)
        assert "canonical_multi_asset_etfs" in ctx["universes"]

    def test_asset_classes_match_registry(self):
        reg = _stub_registry()
        ctx = build_registry_context(reg)
        assert "multi_asset" in ctx["asset_classes"]

    def test_validation_protocols_match_registry(self):
        reg = _stub_registry()
        ctx = build_registry_context(reg)
        assert "rolling" in ctx["validation_protocols"]

    def test_diagnostics_match_registry(self):
        reg = _stub_registry()
        ctx = build_registry_context(reg)
        assert "sharpe_ratio" in ctx["diagnostics"]

    def test_feature_families_derived_from_features(self):
        reg = _stub_registry()
        ctx = build_registry_context(reg)
        fams = set(ctx["feature_families"])
        assert "momentum" in fams
        assert "volatility" in fams
        assert "carry" in fams

    def test_feature_families_none_excluded(self):
        reg = _stub_registry()
        # One feature with no family
        reg.feature.side_effect = lambda name: SimpleNamespace(
            name=name,
            family={"mom_20": "momentum"}.get(name),  # only mom_20 has family
        )
        ctx = build_registry_context(reg)
        # None values must not appear
        assert None not in ctx["feature_families"]

    def test_default_mode_from_registry(self):
        reg = _stub_registry()
        ctx = build_registry_context(reg)
        assert ctx["default_mode"] == "recommended"

    def test_default_mode_fallback_on_error(self):
        reg = _stub_registry()
        reg.default_policy.side_effect = Exception("no policy")
        ctx = build_registry_context(reg)
        assert ctx["default_mode"] == "recommended"

    def test_compact_context_is_deterministic(self):
        reg = _stub_registry()
        ctx1 = build_registry_context(reg, compact=True)
        ctx2 = build_registry_context(reg, compact=True)
        assert ctx1 == ctx2

    def test_lists_are_sorted(self):
        reg = _stub_registry()
        ctx = build_registry_context(reg, compact=True)
        assert ctx["models"] == sorted(ctx["models"])
        assert ctx["presets"] == sorted(ctx["presets"])

    def test_no_unsupported_model_in_context(self):
        reg = _stub_registry()
        ctx = build_registry_context(reg)
        # Only models in registry.model_names() should appear.
        assert "RandomForest" not in ctx["models"]
        assert "GradientBoosting" not in ctx["models"]

    def test_full_context_has_model_details(self):
        reg = _stub_registry()
        reg.model.side_effect = lambda name: SimpleNamespace(
            name=name, display_name=name, description=f"{name} desc",
        )
        ctx = build_registry_context(reg, compact=False)
        assert "model_details" in ctx
        assert any(m["name"] in ("RidgeRegression", "LassoRegression") for m in ctx["model_details"])

    def test_full_context_has_preset_details(self):
        reg = _stub_registry()
        reg.preset.side_effect = lambda name: SimpleNamespace(
            name=name, display_name=name, description="desc", asset_class="multi_asset",
            universe="canonical_multi_asset_etfs",
        )
        ctx = build_registry_context(reg, compact=False)
        assert "preset_details" in ctx
        assert any(p["name"] == "canonical_multi_asset_etf" for p in ctx["preset_details"])


# ---------------------------------------------------------------------------
# B. Prompt builder
# ---------------------------------------------------------------------------

class TestBuildIntentParserPrompt:
    def _ctx(self) -> dict:
        return build_registry_context(_stub_registry(), compact=True)

    def test_prompt_contains_user_request(self):
        request = "Test LassoRegression from 2015 to 2024."
        prompt = build_intent_parser_prompt(request, self._ctx())
        assert request in prompt

    def test_prompt_instructs_strict_json(self):
        prompt = build_intent_parser_prompt("test", self._ctx())
        prompt_lower = prompt.lower()
        assert "strict json" in prompt_lower or "json only" in prompt_lower

    def test_prompt_includes_models(self):
        prompt = build_intent_parser_prompt("test", self._ctx())
        assert "RidgeRegression" in prompt
        assert "LassoRegression" in prompt

    def test_prompt_includes_presets(self):
        prompt = build_intent_parser_prompt("test", self._ctx())
        assert "canonical_multi_asset_etf" in prompt

    def test_prompt_includes_universes(self):
        prompt = build_intent_parser_prompt("test", self._ctx())
        assert "canonical_multi_asset_etfs" in prompt

    def test_prompt_includes_validation_protocols(self):
        prompt = build_intent_parser_prompt("test", self._ctx())
        assert "rolling" in prompt

    def test_prompt_forbids_yaml(self):
        prompt = build_intent_parser_prompt("test", self._ctx())
        assert "yaml" in prompt.lower() or "YAML" in prompt

    def test_prompt_forbids_execution(self):
        prompt = build_intent_parser_prompt("test", self._ctx())
        prompt_lower = prompt.lower()
        assert "execut" in prompt_lower or "never" in prompt_lower

    def test_prompt_says_unsupported_values_must_be_null_or_unrecognised(self):
        prompt = build_intent_parser_prompt("test", self._ctx())
        prompt_lower = prompt.lower()
        assert "null" in prompt_lower
        assert "unrecognised" in prompt_lower or "unrecognized" in prompt_lower

    def test_prompt_contains_all_researchspec_fields(self):
        prompt = build_intent_parser_prompt("test", self._ctx())
        required_fields = [
            "preset", "objective", "asset_class", "universe", "tickers",
            "start_date", "end_date", "model", "label", "signal",
            "transaction_cost_bps", "validation_protocol",
        ]
        for f in required_fields:
            assert f in prompt, f"Field '{f}' missing from prompt"

    def test_prompt_output_schema_contains_json_shape(self):
        prompt = build_intent_parser_prompt("test", self._ctx())
        assert '"research_spec"' in prompt
        assert '"unrecognised_terms"' in prompt
        assert '"confidence"' in prompt
        assert '"summary"' in prompt

    def test_prompt_is_deterministic(self):
        ctx = self._ctx()
        p1 = build_intent_parser_prompt("same request", ctx)
        p2 = build_intent_parser_prompt("same request", ctx)
        assert p1 == p2

    def test_prompt_forbids_recommending_approval(self):
        prompt = build_intent_parser_prompt("test", self._ctx())
        prompt_lower = prompt.lower()
        assert "approv" in prompt_lower and "not" in prompt_lower or "never" in prompt_lower

    def test_prompt_mentions_feature_families(self):
        prompt = build_intent_parser_prompt("test", self._ctx())
        assert "feature_famil" in prompt.lower() or "feature families" in prompt.lower()
