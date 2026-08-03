"""Tests for src/experiments/execution_profile.py — StageRecorder & friends."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pytest
from src.experiments.execution_profile import (
    ExecutionTrace,
    StageExecution,
    StageRecorder,
    execution_trace_jsonl_path,
    jsonl_sink,
    write_execution_trace,
)

# ---------------------------------------------------------------------------
# Successful stage
# ---------------------------------------------------------------------------


def test_successful_stage_records_completed_status():
    recorder = StageRecorder()
    with recorder.stage("Load market data"):
        pass
    assert len(recorder.stages) == 1
    assert recorder.stages[0].status == "completed"


def test_successful_stage_has_valid_timestamps_and_duration():
    recorder = StageRecorder()
    with recorder.stage("Load market data"):
        pass
    stage = recorder.stages[0]
    # Must be parseable ISO-8601.
    started = datetime.fromisoformat(stage.started_at)
    ended = datetime.fromisoformat(stage.ended_at)
    assert ended >= started
    assert stage.duration_s is not None
    assert stage.duration_s >= 0.0


def test_successful_stage_records_metadata():
    recorder = StageRecorder()
    with recorder.stage("Load market data", metadata={"ticker_count": 5}):
        pass
    assert recorder.stages[0].metadata == {"ticker_count": 5}


def test_running_event_emitted_before_completed_event():
    events = []
    recorder = StageRecorder(sink=events.append)
    with recorder.stage("Load market data"):
        assert events[-1]["status"] == "running"
        assert events[-1]["ended_at"] is None
    assert len(events) == 2
    assert events[0]["status"] == "running"
    assert events[1]["status"] == "completed"


# ---------------------------------------------------------------------------
# Failed stage
# ---------------------------------------------------------------------------


def test_failed_stage_reraises_original_exception():
    recorder = StageRecorder()
    with pytest.raises(ValueError, match="boom"):
        with recorder.stage("Backtesting"):
            raise ValueError("boom")


def test_failed_stage_records_failed_status_and_error():
    recorder = StageRecorder()
    with pytest.raises(ValueError):
        with recorder.stage("Backtesting"):
            raise ValueError("boom")
    stage = recorder.stages[0]
    assert stage.status == "failed"
    assert stage.ended_at is not None
    assert stage.duration_s is not None
    assert "boom" in stage.error
    assert "ValueError" in stage.error


def test_failed_stage_emits_running_then_failed():
    events = []
    recorder = StageRecorder(sink=events.append)
    with pytest.raises(RuntimeError):
        with recorder.stage("Backtesting"):
            raise RuntimeError("nope")
    assert [e["status"] for e in events] == ["running", "failed"]


def test_stage_after_failure_is_not_recorded():
    """A stage raising must not prevent the recorder itself from being usable
    for stages that already ran before it — no partial/placeholder entries
    are invented for stages that never started."""
    recorder = StageRecorder()
    with pytest.raises(ValueError):
        with recorder.stage("Load market data"):
            pass
        with recorder.stage("Backtesting"):
            raise ValueError("boom")
    assert [s.name for s in recorder.stages] == ["Load market data", "Backtesting"]
    assert recorder.stages[0].status == "completed"
    assert recorder.stages[1].status == "failed"


# ---------------------------------------------------------------------------
# Metadata updates mid-stage
# ---------------------------------------------------------------------------


def test_update_metadata_during_stage_is_persisted():
    recorder = StageRecorder()
    with recorder.stage("Load market data", metadata={"ticker_count": 2}) as stage:
        stage.update_metadata(row_count=500)
    assert recorder.stages[0].metadata == {"ticker_count": 2, "row_count": 500}


def test_metadata_is_json_serializable():
    recorder = StageRecorder()
    with recorder.stage("Walk-forward validation") as stage:
        stage.update_metadata(n_splits=4, start="2020-01-01", cost_bps=5.0)
    json.dumps(recorder.stages[0].to_dict())  # must not raise


# ---------------------------------------------------------------------------
# Sink failure must never interrupt the pipeline
# ---------------------------------------------------------------------------


def _broken_sink(_event):
    raise OSError("disk full")


def test_sink_failure_does_not_interrupt_execution(caplog):
    recorder = StageRecorder(sink=_broken_sink)
    with caplog.at_level(logging.WARNING):
        with recorder.stage("Load market data"):
            pass
    assert recorder.stages[0].status == "completed"


def test_sink_failure_is_logged(caplog):
    recorder = StageRecorder(sink=_broken_sink)
    with caplog.at_level(logging.WARNING):
        with recorder.stage("Load market data"):
            pass
    assert any("sink failed" in rec.message for rec in caplog.records)


def test_sink_failure_on_failed_stage_still_reraises_original_error():
    recorder = StageRecorder(sink=_broken_sink)
    with pytest.raises(ValueError, match="boom"):
        with recorder.stage("Backtesting"):
            raise ValueError("boom")


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_stages_remain_in_execution_order():
    recorder = StageRecorder()
    names = ["Load configuration", "Load market data", "Backtesting", "Generate figures"]
    for name in names:
        with recorder.stage(name):
            pass
    assert [s.name for s in recorder.stages] == names


# ---------------------------------------------------------------------------
# JSONL sink
# ---------------------------------------------------------------------------


def test_jsonl_sink_writes_one_parseable_line_per_event(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    sink = jsonl_sink(path)
    recorder = StageRecorder(sink=sink)
    with recorder.stage("Load market data"):
        pass
    with recorder.stage("Backtesting"):
        pass

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 4  # running+completed per stage, 2 stages
    for line in lines:
        json.loads(line)  # must not raise


def test_jsonl_sink_shows_running_then_completed_per_stage(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    recorder = StageRecorder(sink=jsonl_sink(path))
    with recorder.stage("Load market data"):
        pass
    events = [json.loads(line) for line in path.read_text().strip().split("\n")]
    assert events[0]["name"] == "Load market data"
    assert events[0]["status"] == "running"
    assert events[1]["name"] == "Load market data"
    assert events[1]["status"] == "completed"


def test_jsonl_sink_reset_true_truncates_existing_file(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    path.write_text("stale line from a previous run\n")
    sink = jsonl_sink(path, reset=True)
    recorder = StageRecorder(sink=sink)
    with recorder.stage("Load market data"):
        pass
    content = path.read_text()
    assert "stale line" not in content


def test_jsonl_sink_reset_false_appends_to_existing_file(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    first = jsonl_sink(path, reset=True)
    r1 = StageRecorder(sink=first)
    with r1.stage("Load market data"):
        pass

    second = jsonl_sink(path, reset=False)
    r2 = StageRecorder(sink=second, existing_stages=r1.stages)
    with r2.stage("Generate report"):
        pass

    lines = path.read_text().strip().split("\n")
    names = {json.loads(line)["name"] for line in lines}
    assert names == {"Load market data", "Generate report"}


def test_execution_trace_jsonl_path_uses_experiment_name(tmp_path: Path):
    p = execution_trace_jsonl_path("my_experiment", base_dir=tmp_path)
    assert p == tmp_path / "my_experiment.jsonl"


# ---------------------------------------------------------------------------
# finalize() / ExecutionTrace / write_execution_trace()
# ---------------------------------------------------------------------------


def test_finalize_reports_completed_status_when_all_stages_ok():
    recorder = StageRecorder()
    with recorder.stage("Load market data"):
        pass
    with recorder.stage("Backtesting"):
        pass
    trace = recorder.finalize("my_experiment")
    assert isinstance(trace, ExecutionTrace)
    assert trace.status == "completed"
    assert trace.experiment_name == "my_experiment"
    assert len(trace.stages) == 2


def test_finalize_reports_failed_status_when_any_stage_failed():
    recorder = StageRecorder()
    with recorder.stage("Load market data"):
        pass
    with pytest.raises(ValueError):
        with recorder.stage("Backtesting"):
            raise ValueError("boom")
    trace = recorder.finalize("my_experiment")
    assert trace.status == "failed"


def test_finalize_total_duration_is_sum_of_stage_durations():
    recorder = StageRecorder()
    with recorder.stage("Load market data"):
        pass
    with recorder.stage("Backtesting"):
        pass
    trace = recorder.finalize("my_experiment")
    expected = sum(s.duration_s for s in recorder.stages)
    assert trace.total_duration_s == pytest.approx(expected)


def test_write_execution_trace_persists_json(tmp_path: Path):
    recorder = StageRecorder()
    with recorder.stage("Load market data", metadata={"ticker_count": 3}):
        pass
    trace = recorder.finalize("my_experiment")
    dest = write_execution_trace(tmp_path, trace)
    assert dest == tmp_path / "execution_trace.json"
    data = json.loads(dest.read_text())
    assert data["experiment_name"] == "my_experiment"
    assert data["status"] == "completed"
    assert len(data["stages"]) == 1
    assert data["stages"][0]["metadata"] == {"ticker_count": 3}


def test_stage_execution_to_dict_is_json_serializable():
    se = StageExecution(
        name="x", status="completed", started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:01+00:00", duration_s=1.0, metadata={"a": 1},
    )
    json.dumps(se.to_dict())  # must not raise
