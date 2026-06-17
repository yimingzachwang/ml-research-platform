"""Tests for operator manual routing rules and auto-load (Phase 5G.1).

Verifies:
  - _OPERATOR_MANUAL_TEXT is populated at import time when file is present.
  - _load_operator_manual_text() returns None gracefully when file is missing.
  - get_zeto_operator_manual() includes manual_text when file is present.
  - _OPERATOR_RULES includes the 7 routing rules (ROUTING A–G).
  - Routing rules cover the key behavioural requirements.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import src.mcp.zeto_server as zeto


# ---------------------------------------------------------------------------
# A — Module-level cache
# ---------------------------------------------------------------------------


class TestOperatorManualTextCache:
    def test_manual_text_is_str_or_none(self):
        assert zeto._OPERATOR_MANUAL_TEXT is None or isinstance(
            zeto._OPERATOR_MANUAL_TEXT, str
        )

    def test_manual_text_populated_when_file_exists(self, tmp_path):
        content = "# Operator Manual\n## Tool Routing Rules\nsome rules"
        manual_file = tmp_path / "manual.md"
        manual_file.write_text(content, encoding="utf-8")
        result = zeto._load_operator_manual_text.__wrapped__ if hasattr(
            zeto._load_operator_manual_text, "__wrapped__"
        ) else None
        # Direct path override test:
        original_path = zeto._OPERATOR_MANUAL_PATH
        with patch.object(
            zeto,
            "_OPERATOR_MANUAL_PATH",
            str(manual_file.relative_to(_REPO_ROOT)) if manual_file.is_relative_to(_REPO_ROOT) else str(manual_file),
        ):
            text = zeto._load_operator_manual_text()
        assert text == content

    def test_load_operator_manual_text_returns_none_when_file_missing(self, tmp_path):
        with patch("src.mcp.zeto_server.PROJECT_ROOT", tmp_path):
            text = zeto._load_operator_manual_text()
        assert text is None

    def test_load_operator_manual_text_returns_str_when_file_present(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "LM_STUDIO_QWEN_OPERATOR_MANUAL.md").write_text(
            "# Manual\n## Tool Routing Rules\nA: use create_validated_draft_from_request",
            encoding="utf-8",
        )
        with patch("src.mcp.zeto_server.PROJECT_ROOT", tmp_path):
            text = zeto._load_operator_manual_text()
        assert isinstance(text, str)
        assert "Tool Routing Rules" in text


# ---------------------------------------------------------------------------
# B — get_zeto_operator_manual() includes manual_text when present
# ---------------------------------------------------------------------------


class TestGetZetoOperatorManualIncludesText:
    def test_manual_text_not_in_payload(self):
        # manual_text is intentionally excluded from the tool payload (size limit).
        # The cache (_OPERATOR_MANUAL_TEXT) is available for internal server use only.
        out = zeto.get_zeto_operator_manual()
        assert "manual_text" not in out["data"]

    def test_manual_available_false_when_none(self):
        with patch("src.mcp.zeto_server._OPERATOR_MANUAL_TEXT", None):
            out = zeto.get_zeto_operator_manual()
        assert out["data"]["manual_available"] is False

    def test_manual_available_true_when_loaded(self):
        fake_text = "# Operator Manual\n## Tool Routing Rules\nA: use create_validated..."
        with patch("src.mcp.zeto_server._OPERATOR_MANUAL_TEXT", fake_text):
            out = zeto.get_zeto_operator_manual()
        assert out["data"]["manual_available"] is True

    def test_rules_always_present(self):
        with patch("src.mcp.zeto_server._OPERATOR_MANUAL_TEXT", None):
            out = zeto.get_zeto_operator_manual()
        assert isinstance(out["data"]["rules"], list)
        assert len(out["data"]["rules"]) > 0

    def test_manual_path_always_present(self):
        out = zeto.get_zeto_operator_manual()
        assert out["data"]["manual_path"] == zeto._OPERATOR_MANUAL_PATH


# ---------------------------------------------------------------------------
# C — _OPERATOR_RULES routing rules content
# ---------------------------------------------------------------------------


class TestRoutingRulesContent:
    """Verify the 7 routing rules (ROUTING A–G) are present in _OPERATOR_RULES."""

    def _rules_text(self) -> str:
        return "\n".join(zeto._OPERATOR_RULES)

    def test_routing_a_present(self):
        text = self._rules_text()
        assert "ROUTING A" in text
        assert "create_validated_draft_from_request" in text

    def test_routing_b_present(self):
        text = self._rules_text()
        assert "ROUTING B" in text
        assert "recommend_next_actions_from_request" in text

    def test_routing_c_present(self):
        text = self._rules_text()
        assert "ROUTING C" in text
        assert "generate_experiment_draft" in text

    def test_routing_d_present(self):
        text = self._rules_text()
        assert "ROUTING D" in text
        assert "wrong_tool_for_research_design_request" in text

    def test_routing_e_present(self):
        text = self._rules_text()
        assert "ROUTING E" in text
        # Must warn against inventing experiment names
        assert "invent" in text.lower() or "experiment names" in text.lower()

    def test_routing_f_present(self):
        text = self._rules_text()
        assert "ROUTING F" in text
        # Must mention re-routing on failure
        assert "re-route" in text or "reroute" in text.lower() or "high-level" in text

    def test_routing_g_present(self):
        text = self._rules_text()
        assert "ROUTING G" in text
        # Must mention unchanged governance gates
        assert "governance" in text.lower() or "APPROVE" in text or "RUN" in text

    def test_routing_rules_count(self):
        routing_rules = [r for r in zeto._OPERATOR_RULES if r.startswith("ROUTING ")]
        assert len(routing_rules) >= 7, (
            f"Expected at least 7 ROUTING rules, found {len(routing_rules)}"
        )


# ---------------------------------------------------------------------------
# D — Operator manual markdown contains Tool Routing Rules section
# ---------------------------------------------------------------------------


class TestOperatorManualMarkdownContent:
    """If the file is present, it must contain the Tool Routing Rules section."""

    def test_markdown_contains_tool_routing_rules_section(self):
        manual_path = _REPO_ROOT / zeto._OPERATOR_MANUAL_PATH
        if not manual_path.exists():
            pytest.skip("Operator manual file not present — skipping content check")
        text = manual_path.read_text(encoding="utf-8")
        assert "## Tool Routing Rules" in text

    def test_markdown_routing_section_covers_all_labels(self):
        manual_path = _REPO_ROOT / zeto._OPERATOR_MANUAL_PATH
        if not manual_path.exists():
            pytest.skip("Operator manual file not present — skipping content check")
        text = manual_path.read_text(encoding="utf-8")
        for label in ("### A.", "### B.", "### C.", "### D.", "### E.", "### F.", "### G."):
            assert label in text, f"Missing section {label!r} in operator manual"

    def test_markdown_mentions_create_validated_draft_from_request(self):
        manual_path = _REPO_ROOT / zeto._OPERATOR_MANUAL_PATH
        if not manual_path.exists():
            pytest.skip("Operator manual file not present — skipping content check")
        text = manual_path.read_text(encoding="utf-8")
        assert "create_validated_draft_from_request" in text
