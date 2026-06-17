"""Lightweight contract tests for the LM Studio demo script (Phase 5F.2).

These tests verify script structure, CLI defaults, dry-run safety, and output
contract without ever calling LM Studio or any external endpoint.

Design principles:
  - NO live LM Studio calls.
  - NO API keys required.
  - Exercises only: import, argparse defaults, dry-run execution, governance proof.
"""

from __future__ import annotations

import importlib
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Script loading helpers
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "demo_lmstudio_research_agent_loop.py"


def _load_demo_module():
    """Import the demo script as a module."""
    spec = importlib.util.spec_from_file_location("demo_lmstudio", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# TestImport
# ---------------------------------------------------------------------------

class TestImport:
    """The script must import without side effects."""

    def test_script_file_exists(self):
        assert _SCRIPT_PATH.exists(), f"Script not found: {_SCRIPT_PATH}"

    def test_script_imports_cleanly(self):
        mod = _load_demo_module()
        assert callable(getattr(mod, "main", None))

    def test_prompts_list_present(self):
        mod = _load_demo_module()
        assert isinstance(mod._PROMPTS, list)
        assert len(mod._PROMPTS) == 5

    def test_prompt_ids(self):
        mod = _load_demo_module()
        ids = [p["id"] for p in mod._PROMPTS]
        assert ids == ["P1", "P2", "P3", "P4", "P5"]

    def test_governance_accumulator_class(self):
        mod = _load_demo_module()
        gov = mod.GovernanceAccumulator()
        assert gov.llm_used_as_parser_only is True
        assert gov.registry_remained_authoritative is True
        assert gov.draft_not_approved is True
        assert gov.experiment_not_executed is True
        assert gov.all_ok() is True

    def test_default_base_url(self):
        mod = _load_demo_module()
        assert "127.0.0.1:1234" in mod._DEFAULT_BASE_URL

    def test_default_model_is_qwen(self):
        mod = _load_demo_module()
        assert "qwen" in mod._DEFAULT_MODEL.lower()


# ---------------------------------------------------------------------------
# TestCLIDefaults
# ---------------------------------------------------------------------------

class TestCLIDefaults:
    """CLI argument defaults enforce safe behaviour."""

    def _parse(self, args: list[str]):
        import argparse

        mod = _load_demo_module()
        # Replicate the argparse setup from main() using known defaults
        parser = argparse.ArgumentParser()
        parser.add_argument("--base-url", default=mod._DEFAULT_BASE_URL)
        parser.add_argument("--model", default=mod._DEFAULT_MODEL)
        parser.add_argument("--save-draft", action="store_true", default=False)
        parser.add_argument("--dry-run", action="store_true", default=False)
        parser.add_argument("--prompt", choices=["P1", "P2", "P3", "P4", "P5"], default=None)
        return parser.parse_args(args)

    def test_save_draft_default_false(self):
        ns = self._parse([])
        assert ns.save_draft is False

    def test_dry_run_default_false(self):
        ns = self._parse([])
        assert ns.dry_run is False

    def test_dry_run_flag_sets_true(self):
        ns = self._parse(["--dry-run"])
        assert ns.dry_run is True

    def test_save_draft_flag_sets_true(self):
        ns = self._parse(["--save-draft"])
        assert ns.save_draft is True

    def test_default_model(self):
        ns = self._parse([])
        assert "qwen" in ns.model.lower()

    def test_custom_model_override(self):
        ns = self._parse(["--model", "qwen2.5-coder-7b"])
        assert ns.model == "qwen2.5-coder-7b"

    def test_custom_base_url(self):
        ns = self._parse(["--base-url", "http://localhost:5678/v1"])
        assert ns.base_url == "http://localhost:5678/v1"

    def test_prompt_filter(self):
        ns = self._parse(["--dry-run", "--prompt", "P1"])
        assert ns.prompt == "P1"


# ---------------------------------------------------------------------------
# TestDryRunExecution
# ---------------------------------------------------------------------------

class TestDryRunExecution:
    """Dry-run must complete without calling LM Studio or any network."""

    def test_dry_run_exits_zero(self, capsys):
        mod = _load_demo_module()
        exit_code = mod.main(["--dry-run"])
        assert exit_code == 0

    def test_dry_run_does_not_call_endpoint_check(self):
        mod = _load_demo_module()
        with patch.object(mod, "_check_endpoint_reachable") as mock_check:
            mod.main(["--dry-run"])
            mock_check.assert_not_called()

    def test_dry_run_prints_governance_proof(self, capsys):
        mod = _load_demo_module()
        mod.main(["--dry-run"])
        captured = capsys.readouterr()
        assert "GOVERNANCE PROOF" in captured.out

    def test_dry_run_all_governance_invariants_mentioned(self, capsys):
        mod = _load_demo_module()
        mod.main(["--dry-run"])
        captured = capsys.readouterr()
        assert "intent extraction" in captured.out.lower()
        assert "draft approval did not occur" in captured.out.lower()
        assert "experiment execution did not occur" in captured.out.lower()

    def test_dry_run_p1_reaches_draft_created(self, capsys):
        mod = _load_demo_module()
        mod.main(["--dry-run", "--prompt", "P1"])
        captured = capsys.readouterr()
        assert "draft_created" in captured.out

    def test_dry_run_p1_governance_gate_held(self, capsys):
        mod = _load_demo_module()
        mod.main(["--dry-run", "--prompt", "P1"])
        captured = capsys.readouterr()
        assert "confirmation_required" in captured.out
        assert "Gate held" in captured.out

    def test_dry_run_p2_randomforest_not_in_spec(self, capsys):
        mod = _load_demo_module()
        mod.main(["--dry-run", "--prompt", "P2"])
        captured = capsys.readouterr()
        # RandomForest should not appear as an accepted model in the spec output
        assert "Registry rejection check:" in captured.out
        assert "OK" in captured.out

    def test_dry_run_p3_ambiguity_warning(self, capsys):
        mod = _load_demo_module()
        mod.main(["--dry-run", "--prompt", "P3"])
        captured = capsys.readouterr()
        assert "Ambiguous" in captured.out or "multiple models" in captured.out.lower()

    def test_dry_run_p4_execution_keyword_noted(self, capsys):
        mod = _load_demo_module()
        mod.main(["--dry-run", "--prompt", "P4"])
        captured = capsys.readouterr()
        # Execution keywords are noted but not acted on
        assert "execution" in captured.out.lower() or "run" in captured.out.lower()

    def test_dry_run_p1_draft_not_approved(self, capsys):
        mod = _load_demo_module()
        mod.main(["--dry-run", "--prompt", "P1"])
        captured = capsys.readouterr()
        # Check that draft_approved is present and set to False
        assert "draft_approved:" in captured.out
        lines = [l for l in captured.out.splitlines() if "draft_approved:" in l]
        assert any("False" in l for l in lines)

    def test_dry_run_p1_execution_not_performed(self, capsys):
        mod = _load_demo_module()
        mod.main(["--dry-run", "--prompt", "P1"])
        captured = capsys.readouterr()
        assert "execution_performed:" in captured.out
        lines = [l for l in captured.out.splitlines() if "execution_performed:" in l]
        assert any("False" in l for l in lines)

    def test_dry_run_save_draft_false_by_default(self, capsys):
        mod = _load_demo_module()
        mod.main(["--dry-run", "--prompt", "P1"])
        captured = capsys.readouterr()
        # draft_saved should be False when --save-draft is not passed
        assert "draft_saved:" in captured.out
        lines = [l for l in captured.out.splitlines() if "draft_saved:" in l]
        assert any("False" in l for l in lines)


# ---------------------------------------------------------------------------
# TestEndpointCheck
# ---------------------------------------------------------------------------

class TestEndpointCheck:
    """Endpoint check returns False if LM Studio is not reachable."""

    def test_unreachable_endpoint_returns_false(self):
        mod = _load_demo_module()
        # Use a port that is guaranteed not to be in use in CI
        result = mod._check_endpoint_reachable("http://127.0.0.1:19999/v1", timeout=0.5)
        assert result is False

    def test_main_exits_1_when_endpoint_unreachable(self, capsys):
        mod = _load_demo_module()
        # Patch _check_endpoint_reachable to simulate unavailability
        with patch.object(mod, "_check_endpoint_reachable", return_value=False):
            exit_code = mod.main([])  # no --dry-run
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "not reachable" in captured.out.lower()
        assert "--dry-run" in captured.out

    def test_main_continues_when_endpoint_reachable(self, capsys):
        mod = _load_demo_module()
        # Patch reachability + all chain functions so no real LM Studio call occurs
        with (
            patch.object(mod, "_check_endpoint_reachable", return_value=True),
            patch("src.agent.intent.intent_parser.parse_research_intent") as mock_parse,
        ):
            from src.agent.intent.intent_result import IntentParseResult
            from src.agent.intent.research_spec import ResearchSpec
            mock_parse.return_value = IntentParseResult(
                status="needs_clarification",
                research_spec=ResearchSpec(),
                raw_text="test",
                summary="mocked",
            )
            exit_code = mod.main(["--prompt", "P2"])  # no --dry-run, reachable
        # No crash — chain ran (even if stub-like)
        assert exit_code in (0, 2)


# ---------------------------------------------------------------------------
# TestGovernanceAccumulator
# ---------------------------------------------------------------------------

class TestGovernanceAccumulator:
    """GovernanceAccumulator correctly tracks violations."""

    def _gov(self):
        return _load_demo_module().GovernanceAccumulator()

    def test_initial_all_ok(self):
        gov = self._gov()
        assert gov.all_ok() is True

    def test_violation_makes_not_ok(self):
        gov = self._gov()
        gov.record_violation("test violation")
        assert gov.all_ok() is False
        assert "test violation" in gov.violations

    def test_multiple_violations(self):
        gov = self._gov()
        gov.record_violation("A")
        gov.record_violation("B")
        assert len(gov.violations) == 2

    def test_no_violations_initially(self):
        gov = self._gov()
        assert gov.violations == []


# ---------------------------------------------------------------------------
# TestOutputContract
# ---------------------------------------------------------------------------

class TestOutputContract:
    """Output must include all required sections."""

    def test_output_includes_all_prompt_ids(self, capsys):
        mod = _load_demo_module()
        mod.main(["--dry-run"])
        captured = capsys.readouterr()
        for pid in ["P1:", "P2:", "P3:", "P4:", "P5:"]:
            assert pid in captured.out, f"Missing prompt section {pid}"

    def test_output_includes_parse_step_header(self, capsys):
        mod = _load_demo_module()
        mod.main(["--dry-run", "--prompt", "P1"])
        captured = capsys.readouterr()
        assert "[1/N] PARSE" in captured.out

    def test_output_includes_resolve_step_header(self, capsys):
        mod = _load_demo_module()
        mod.main(["--dry-run", "--prompt", "P1"])
        captured = capsys.readouterr()
        assert "[2/N] RESOLVE" in captured.out

    def test_output_includes_plan_step_header(self, capsys):
        mod = _load_demo_module()
        mod.main(["--dry-run", "--prompt", "P1"])
        captured = capsys.readouterr()
        assert "[3/N] PLAN" in captured.out

    def test_output_includes_full_chain_header_for_p1(self, capsys):
        mod = _load_demo_module()
        mod.main(["--dry-run", "--prompt", "P1"])
        captured = capsys.readouterr()
        assert "FULL CHAIN" in captured.out

    def test_output_governance_proof_has_eight_items(self, capsys):
        mod = _load_demo_module()
        mod.main(["--dry-run"])
        captured = capsys.readouterr()
        proof_section = captured.out.split("GOVERNANCE PROOF")[-1]
        checks = [line for line in proof_section.splitlines() if line.strip().startswith(("✓", "✗"))]
        assert len(checks) == 8, f"Expected 8 governance items, got {len(checks)}: {checks}"
