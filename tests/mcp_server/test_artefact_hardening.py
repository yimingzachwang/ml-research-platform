"""Artefact-hardening tests for Zeto incomplete-metrics protection.

Verifies that missing or malformed experiment metrics are caught cleanly at
every stage of the governed research workflow — get_experiment_metrics,
compare_experiment_metrics, review_post_run_result, verify_experiment_artifacts,
memory indexing, and the operator manual stopping rules.

No disk I/O, no quant engine, no LLM; all API calls are patched or use real
in-memory artefacts written to tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.mcp import zeto_server as zeto
import src.orchestration.api.research_api as api

_MCP_MOD = "src.mcp.zeto_server._api"
_CONTRACT_KEYS = {"ok", "stage", "display", "data", "next_suggested_action"}


# ---------------------------------------------------------------------------
# Helpers — build minimal on-disk artefacts for tmp_path experiments
# ---------------------------------------------------------------------------

def _write_complete_experiment(base: Path, name: str, sharpe: float = 0.5) -> None:
    d = base / name
    (d / "diagnostics").mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"experiment_name": name}))
    (d / "metrics.json").write_text(json.dumps({
        "sharpe_ratio": sharpe,
        "mean_oos_sharpe": -0.2,
        "max_drawdown": -0.35,
        "max_drawdown_pct": -35.0,
    }))
    (d / "diagnostics" / "split_metrics.json").write_text(json.dumps({
        "summary": {"n_splits": 6, "mean_sharpe": -0.2, "std_sharpe": 0.4,
                    "hit_rate_positive_sharpe": 0.5, "mean_annualized_return": 0.03,
                    "worst_max_drawdown": -0.35},
        "splits": [{"sharpe_ratio": -0.2 + i * 0.05} for i in range(6)],
    }))


def _write_incomplete_experiment(base: Path, name: str) -> None:
    """Experiment with metrics.json that has sharpe_ratio=null."""
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"experiment_name": name}))
    (d / "metrics.json").write_text(json.dumps({
        "sharpe_ratio": None,
        "mean_oos_sharpe": None,
    }))


def _write_missing_metrics_experiment(base: Path, name: str) -> None:
    """Experiment whose metrics.json contains no numeric values at all."""
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"experiment_name": name}))
    (d / "metrics.json").write_text(json.dumps({}))


# ---------------------------------------------------------------------------
# 1. get_experiment_metrics handles missing metrics.json cleanly
# ---------------------------------------------------------------------------


def test_get_experiment_metrics_handles_missing_metrics_cleanly(tmp_path):
    """get_experiment_metrics returns ok=false for a nonexistent experiment."""
    result = api.get_experiment_metrics("nonexistent_exp_xyz", base=tmp_path)
    assert result["status"] == "not_found"
    assert result["missing_metrics"]

    out = zeto.get_experiment_metrics.__wrapped__("nonexistent_exp_xyz") \
        if hasattr(zeto.get_experiment_metrics, "__wrapped__") \
        else None
    # Via MCP handler
    with patch(f"{_MCP_MOD}.get_experiment_metrics",
               return_value={"status": "not_found", "experiment_name": "x",
                             "metrics": {}, "missing_metrics": []}):
        out = zeto.get_experiment_metrics("x")
    assert out["ok"] is False
    assert out["stage"] == "experiment_metrics_unavailable"
    assert _CONTRACT_KEYS.issubset(out)


# ---------------------------------------------------------------------------
# 2. get_experiment_metrics handles None numeric values cleanly
# ---------------------------------------------------------------------------


def test_get_experiment_metrics_handles_none_numeric_values(tmp_path):
    """MCP handler returns ok=false with incomplete stage when sharpe_ratio is None."""
    incomplete_result = {
        "status": "ok",
        "experiment_name": "exp_v13",
        "metrics": {
            "sharpe_ratio": None,
            "mean_oos_sharpe": None,
            "max_drawdown_pct": None,
        },
        "missing_metrics": ["sharpe_ratio", "mean_oos_sharpe"],
        "failure_modes": [],
        "metrics_path": "/some/path/metrics.json",
        "report_path": None,
        "plots_dir": None,
    }
    with patch(f"{_MCP_MOD}.get_experiment_metrics", return_value=incomplete_result):
        out = zeto.get_experiment_metrics("exp_v13")

    assert out["ok"] is False, (
        "get_experiment_metrics must return ok=false when sharpe_ratio is None"
    )
    assert out["stage"] == "experiment_metrics_incomplete"
    assert out["next_suggested_action"] == "stop_and_report_to_user"
    assert "invent" in out["display"].lower() or "absent" in out["display"].lower()
    assert _CONTRACT_KEYS.issubset(out)


def test_get_experiment_metrics_none_sharpe_does_not_crash(tmp_path):
    """API get_experiment_metrics never raises when metrics are None."""
    _write_incomplete_experiment(tmp_path, "exp_v13")
    try:
        result = api.get_experiment_metrics("exp_v13", base=tmp_path)
    except TypeError:
        pytest.fail("get_experiment_metrics must not raise TypeError on None metrics")
    assert result["status"] == "ok"
    assert "sharpe_ratio" in result["missing_metrics"]


# ---------------------------------------------------------------------------
# 3. compare_experiment_metrics refuses incomplete candidate metrics
# ---------------------------------------------------------------------------


def test_compare_refuses_incomplete_candidate_metrics(tmp_path):
    """compare_experiment_metrics returns incomplete_metrics when candidate sharpe is None."""
    _write_complete_experiment(tmp_path, "exp_base", sharpe=0.5)
    _write_incomplete_experiment(tmp_path, "exp_v13_incomplete")

    result = api.compare_experiment_metrics(
        "exp_base", "exp_v13_incomplete",
        base=tmp_path,
        persist_evidence=False,
    )
    assert result["status"] == "incomplete_metrics", (
        "compare_experiment_metrics must return incomplete_metrics "
        "when candidate sharpe_ratio is None"
    )
    assert "exp_v13_incomplete" in result["incomplete_experiments"]


def test_compare_refuses_incomplete_base_metrics(tmp_path):
    """compare_experiment_metrics also refuses when BASE has missing core metrics."""
    _write_incomplete_experiment(tmp_path, "exp_base_broken")
    _write_complete_experiment(tmp_path, "exp_v2", sharpe=0.6)

    result = api.compare_experiment_metrics(
        "exp_base_broken", "exp_v2",
        base=tmp_path,
        persist_evidence=False,
    )
    assert result["status"] == "incomplete_metrics"
    assert "exp_base_broken" in result["incomplete_experiments"]


# ---------------------------------------------------------------------------
# 4. compare_experiment_metrics does not persist evidence when metrics incomplete
# ---------------------------------------------------------------------------


def test_compare_does_not_persist_evidence_when_incomplete(tmp_path):
    """No comparison_evidence.json must be written when metrics are incomplete."""
    _write_complete_experiment(tmp_path, "exp_base", sharpe=0.5)
    _write_incomplete_experiment(tmp_path, "exp_v13_bad")
    comp_base = tmp_path / "results" / "comparisons"

    result = api.compare_experiment_metrics(
        "exp_base", "exp_v13_bad",
        base=tmp_path,
        comparisons_base=comp_base,
        persist_evidence=True,
    )
    assert result["status"] == "incomplete_metrics"
    # No evidence directory or file should have been created
    assert not comp_base.exists() or not any(comp_base.rglob("comparison_evidence.json")), (
        "No comparison_evidence.json must be written when metrics are incomplete"
    )


def test_compare_mcp_handler_returns_ok_false_for_incomplete(tmp_path):
    """MCP compare_experiment_metrics returns ok=false on incomplete_metrics status."""
    incomplete_result = {
        "status": "incomplete_metrics",
        "base_experiment": "exp_base",
        "candidate_experiment": "exp_v13",
        "incomplete_experiments": ["exp_v13"],
        "details": "exp_v13 missing sharpe_ratio",
    }
    with patch(f"{_MCP_MOD}.compare_experiment_metrics", return_value=incomplete_result):
        out = zeto.compare_experiment_metrics("exp_base", "exp_v13")

    assert out["ok"] is False
    assert out["stage"] == "experiment_metrics_incomplete"
    assert out["next_suggested_action"] == "stop_and_report_to_user"
    assert "invent" in out["display"].lower() or "refused" in out["display"].lower()
    assert _CONTRACT_KEYS.issubset(out)


# ---------------------------------------------------------------------------
# 5. review_post_run_result returns clean blocked envelope on None metrics
# ---------------------------------------------------------------------------


def test_review_post_run_result_blocked_on_none_metrics():
    """review_post_run_result returns blocked envelope when sharpe_ratio is None."""
    incomplete_metrics = {
        "status": "ok",
        "experiment_name": "exp_v13",
        "metrics": {"sharpe_ratio": None, "mean_oos_sharpe": None},
        "missing_metrics": ["sharpe_ratio", "mean_oos_sharpe"],
        "failure_modes": [],
    }
    with patch(f"{_MCP_MOD}.get_experiment_metrics", return_value=incomplete_metrics):
        out = zeto.review_post_run_result("exp_v13")

    assert out["ok"] is False, (
        "review_post_run_result must return ok=false when core metrics are None"
    )
    assert "blocked_incomplete_metrics" in out["stage"], (
        f"Expected stage containing 'blocked_incomplete_metrics', got: {out['stage']}"
    )
    assert out["next_suggested_action"] == "stop_and_report_to_user"
    assert "invent" in out["display"].lower() or "incomplete" in out["display"].lower()
    assert _CONTRACT_KEYS.issubset(out)


def test_review_post_run_result_blocked_does_not_raise_type_error():
    """review_post_run_result must never raise TypeError on None metrics."""
    incomplete_metrics = {
        "status": "ok",
        "experiment_name": "exp_v13",
        "metrics": {"sharpe_ratio": None},
        "missing_metrics": ["sharpe_ratio"],
        "failure_modes": [],
    }
    with patch(f"{_MCP_MOD}.get_experiment_metrics", return_value=incomplete_metrics):
        try:
            out = zeto.review_post_run_result("exp_v13")
        except TypeError as exc:
            pytest.fail(
                f"review_post_run_result must not raise TypeError on None metrics: {exc}"
            )
    assert out["ok"] is False


def test_review_pre_run_also_blocked_on_none_metrics():
    """run_experiment_review (pre-run path) is also blocked on None metrics."""
    incomplete_metrics = {
        "status": "ok",
        "experiment_name": "exp_v13",
        "metrics": {"sharpe_ratio": None},
        "missing_metrics": ["sharpe_ratio"],
        "failure_modes": [],
    }
    with (
        patch(f"{_MCP_MOD}.get_experiment_metrics", return_value=incomplete_metrics),
        patch("src.mcp.zeto_server._experiment_config_exists", return_value=True),
    ):
        out = zeto.run_experiment_review("exp_v13")

    assert out["ok"] is False
    assert "blocked_incomplete_metrics" in out["stage"]


# ---------------------------------------------------------------------------
# 6. verify_experiment_artifacts reports incomplete artefacts
# ---------------------------------------------------------------------------


def test_verify_experiment_artifacts_reports_complete(tmp_path):
    """verify_experiment_artifacts returns artefact_status=complete for full artefacts."""
    _write_complete_experiment(tmp_path, "exp_ok")
    result = api.verify_experiment_artifacts("exp_ok", base=tmp_path)
    assert result["status"] == "ok"
    assert result["artefact_status"] == "complete"
    assert result["artefact_root_exists"] is True
    assert result["metrics_path_exists"] is True
    assert result["metrics_parseable"] is True
    assert result["required_metrics_present"] is True
    assert result["missing_metrics"] == []


def test_verify_experiment_artifacts_reports_incomplete_missing_root(tmp_path):
    """verify_experiment_artifacts returns incomplete when artefact root is absent."""
    result = api.verify_experiment_artifacts("nonexistent", base=tmp_path)
    assert result["artefact_status"] == "incomplete"
    assert result["artefact_root_exists"] is False


def test_verify_experiment_artifacts_reports_incomplete_none_metrics(tmp_path):
    """verify_experiment_artifacts returns incomplete when sharpe_ratio is None."""
    _write_incomplete_experiment(tmp_path, "exp_bad")
    result = api.verify_experiment_artifacts("exp_bad", base=tmp_path)
    assert result["artefact_status"] == "incomplete"
    assert "sharpe_ratio" in result["missing_metrics"]


def test_verify_mcp_tool_returns_ok_false_when_incomplete():
    """MCP verify_experiment_artifacts returns ok=false on incomplete status."""
    incomplete_verify = {
        "status": "ok",
        "experiment_name": "exp_v13",
        "artefact_root_exists": True,
        "metrics_path_exists": True,
        "metrics_parseable": True,
        "required_metrics_present": False,
        "report_path_exists": False,
        "plots_dir_exists": False,
        "missing_metrics": ["sharpe_ratio"],
        "artefact_status": "incomplete",
    }
    with patch(f"{_MCP_MOD}.verify_experiment_artifacts", return_value=incomplete_verify):
        out = zeto.verify_experiment_artifacts("exp_v13")

    assert out["ok"] is False
    assert out["stage"] == "artefact_verification_incomplete"
    assert out["next_suggested_action"] == "stop_and_report_to_user"
    assert "incomplete" in out["display"].lower()
    assert "invent" in out["display"].lower() or "do not" in out["display"].lower()
    assert _CONTRACT_KEYS.issubset(out)


def test_verify_mcp_tool_returns_ok_true_when_complete():
    """MCP verify_experiment_artifacts returns ok=true and suggests review when complete."""
    complete_verify = {
        "status": "ok",
        "experiment_name": "exp_v13",
        "artefact_root_exists": True,
        "metrics_path_exists": True,
        "metrics_parseable": True,
        "required_metrics_present": True,
        "report_path_exists": True,
        "plots_dir_exists": True,
        "missing_metrics": [],
        "artefact_status": "complete",
    }
    with patch(f"{_MCP_MOD}.verify_experiment_artifacts", return_value=complete_verify):
        out = zeto.verify_experiment_artifacts("exp_v13")

    assert out["ok"] is True
    assert out["stage"] == "artefact_verification_complete"
    assert out["next_suggested_action"] == "review_post_run_result"
    assert _CONTRACT_KEYS.issubset(out)


# ---------------------------------------------------------------------------
# 7. Operator manual contains no-invention metric failure rules
# ---------------------------------------------------------------------------


def test_operator_manual_contains_incomplete_metrics_stopping_rules():
    """Operator manual must contain rules about stopping on incomplete metrics."""
    blob = json.dumps(zeto.get_zeto_operator_manual()).lower()
    assert "incomplete" in blob, "Manual must mention incomplete artefacts"
    assert "verify_experiment_artifacts" in blob, (
        "Manual must reference verify_experiment_artifacts"
    )
    assert "stop" in blob


def test_operator_manual_contains_no_invention_metric_rules():
    """Operator manual must instruct never to invent hypothetical metrics."""
    blob = json.dumps(zeto.get_zeto_operator_manual()).lower()
    assert "invent" in blob or "hypothetical" in blob, (
        "Manual must prohibit inventing metrics"
    )


# ---------------------------------------------------------------------------
# 8. No approval/render/execute side effects from metrics/verify tools
# ---------------------------------------------------------------------------


def test_no_side_effects_from_metrics_and_verify_tools():
    """Metric and verify tools must not call approve, render, or execute."""
    complete_metrics = {
        "status": "ok", "experiment_name": "exp",
        "metrics": {"sharpe_ratio": 0.5, "mean_oos_sharpe": -0.2},
        "missing_metrics": [], "failure_modes": [],
    }
    compare_ok = {
        "status": "ok", "base_experiment": "a", "candidate_experiment": "b",
        "base_sharpe": 0.4, "candidate_sharpe": 0.5,
        "delta_sharpe": 0.1, "base_mean_oos_sharpe": -0.3,
        "candidate_mean_oos_sharpe": -0.2, "delta_mean_oos_sharpe": 0.1,
        "base_max_drawdown": None, "candidate_max_drawdown": None,
        "delta_max_drawdown_pct": None, "base_failure_modes": [],
        "candidate_failure_modes": [], "conclusion": "ok",
        "evidence_path": None,
    }
    verify_ok = {
        "status": "ok", "experiment_name": "exp",
        "artefact_root_exists": True, "metrics_path_exists": True,
        "metrics_parseable": True, "required_metrics_present": True,
        "report_path_exists": True, "plots_dir_exists": True,
        "missing_metrics": [], "artefact_status": "complete",
    }
    with (
        patch(f"{_MCP_MOD}.get_experiment_metrics", return_value=complete_metrics),
        patch(f"{_MCP_MOD}.compare_experiment_metrics", return_value=compare_ok),
        patch(f"{_MCP_MOD}.verify_experiment_artifacts", return_value=verify_ok),
        patch(f"{_MCP_MOD}.approve_experiment_draft") as m_approve,
        patch(f"{_MCP_MOD}.render_draft_to_yaml") as m_render,
        patch(f"{_MCP_MOD}.execute_approved_config") as m_exec,
    ):
        zeto.get_experiment_metrics("exp")
        zeto.compare_experiment_metrics("a", "b")
        zeto.verify_experiment_artifacts("exp")

    m_approve.assert_not_called()
    m_render.assert_not_called()
    m_exec.assert_not_called()


# ---------------------------------------------------------------------------
# 9. No memory evidence created from incomplete metrics
# ---------------------------------------------------------------------------


def test_no_memory_evidence_from_incomplete_metrics(tmp_path):
    """compare_experiment_metrics must not write comparison_evidence when metrics incomplete."""
    _write_complete_experiment(tmp_path, "base_exp", sharpe=0.5)
    _write_incomplete_experiment(tmp_path, "bad_exp")
    comp_root = tmp_path / "results" / "comparisons"

    api.compare_experiment_metrics(
        "base_exp", "bad_exp",
        base=tmp_path,
        comparisons_base=comp_root,
        persist_evidence=True,
    )
    evidence_files = list(comp_root.rglob("comparison_evidence.json")) if comp_root.exists() else []
    assert not evidence_files, (
        "No comparison_evidence.json must be created when metrics are incomplete — "
        "incomplete metrics must not enter the evidence memory"
    )


def test_memory_evidence_written_only_when_both_experiments_complete(tmp_path):
    """comparison_evidence IS written when both experiments have complete metrics."""
    _write_complete_experiment(tmp_path, "base_exp", sharpe=0.5)
    _write_complete_experiment(tmp_path, "cand_exp", sharpe=0.6)
    comp_root = tmp_path / "results" / "comparisons"

    result = api.compare_experiment_metrics(
        "base_exp", "cand_exp",
        base=tmp_path,
        comparisons_base=comp_root,
        persist_evidence=True,
    )
    assert result["status"] == "ok"
    evidence_files = list(comp_root.rglob("comparison_evidence.json"))
    assert evidence_files, (
        "comparison_evidence.json must be written when both experiments have complete metrics"
    )


# ---------------------------------------------------------------------------
# 10. execute_approved_config suggests verify_experiment_artifacts
# ---------------------------------------------------------------------------


def test_execute_approved_config_suggests_verify_after_success():
    """Successful execution must suggest verify_experiment_artifacts as next step."""
    from src.orchestration.api.research_api import ExecutionResult

    stub = ExecutionResult(
        config_path="configs/exp.yaml",
        experiment_name="exp_v13",
        success=True,
        artefact_root="results/experiments/exp_v13",
        report_path=None,
        error=None,
        command_hint="python run.py",
    )
    with patch(f"{_MCP_MOD}.execute_approved_config", return_value=stub):
        out = zeto.execute_approved_config("configs/exp.yaml", confirmation="RUN")

    assert out["ok"] is True
    assert out["stage"] == "execution_completed"
    assert out["next_suggested_action"] == "verify_experiment_artifacts", (
        "After execution, the model must call verify_experiment_artifacts "
        "to confirm artefact completeness before reviewing"
    )
