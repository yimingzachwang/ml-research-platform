"""Execution progress & profiling — observability only.

Records the running / completed / failed lifecycle of the named stages that
already exist inside ``run_experiment_from_config`` / ``_run_ml_experiment`` /
``run_and_report``. This module adds no behaviour to the pipeline it
instruments:

- A stage that raises is recorded as ``failed`` and the original exception is
  re-raised unchanged — nothing here catches, replaces, or reinterprets an
  execution error.
- A broken sink (e.g. a filesystem write failure) is logged and ignored —
  observability must never become a new execution dependency.

Two persistence shapes are produced from the same ``StageExecution`` records:
a live, flushed-per-event JSONL sidecar (``jsonl_sink``) for watching a run in
progress, and a finalized ``ExecutionTrace`` written once as durable
``execution_trace.json`` alongside the rest of an experiment's artefacts.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

StageStatus = Literal["running", "completed", "failed"]

_ERROR_TRUNCATE = 300


@dataclass
class StageExecution:
    """One pipeline stage's observed lifecycle.

    ``ended_at``/``duration_s`` stay ``None`` while ``status == "running"``.
    ``metadata`` must stay small and JSON-serializable — concise operational
    facts already available at the stage boundary (ticker counts, fold
    counts, ...), never large arrays, dataframes, or raw predictions.
    """

    name: str
    status: StageStatus
    started_at: str
    ended_at: str | None = None
    duration_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionTrace:
    """Finalized, durable record of one execute_approved_config run."""

    experiment_name: str
    started_at: str
    ended_at: str
    total_duration_s: float
    status: Literal["completed", "failed"]
    stages: list[StageExecution]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_duration_s": self.total_duration_s,
            "status": self.status,
            "stages": [s.to_dict() for s in self.stages],
        }


class StageHandle:
    """Yielded by ``StageRecorder.stage()`` — lets the wrapped block attach
    metadata that's only known partway through the stage (e.g. a row count
    read after data loads)."""

    def __init__(self, execution: StageExecution) -> None:
        self._execution = execution

    def update_metadata(self, **kwargs: Any) -> None:
        self._execution.metadata.update(kwargs)


class StageRecorder:
    """Records named pipeline stages in execution order with monotonic-clock
    durations and UTC ISO-8601 timestamps. Fully synchronous — no threads, no
    async, no queue. Never changes control flow (see module docstring).
    """

    def __init__(
        self,
        sink: Callable[[dict[str, Any]], None] | None = None,
        existing_stages: list[StageExecution] | None = None,
    ) -> None:
        self._sink = sink
        self.stages: list[StageExecution] = list(existing_stages or [])

    @contextmanager
    def stage(
        self, name: str, metadata: dict[str, Any] | None = None
    ) -> Iterator[StageHandle]:
        started_wall = datetime.now(UTC)
        started_mono = time.monotonic()
        execution = StageExecution(
            name=name,
            status="running",
            started_at=started_wall.isoformat(),
            metadata=dict(metadata or {}),
        )
        self.stages.append(execution)
        self._emit(execution)

        handle = StageHandle(execution)
        try:
            yield handle
        except BaseException as exc:
            execution.status = "failed"
            execution.ended_at = datetime.now(UTC).isoformat()
            execution.duration_s = round(time.monotonic() - started_mono, 3)
            execution.error = f"{type(exc).__name__}: {exc}"[:_ERROR_TRUNCATE]
            self._emit(execution)
            raise
        else:
            execution.status = "completed"
            execution.ended_at = datetime.now(UTC).isoformat()
            execution.duration_s = round(time.monotonic() - started_mono, 3)
            self._emit(execution)

    def _emit(self, execution: StageExecution) -> None:
        if self._sink is None:
            return
        try:
            self._sink(execution.to_dict())
        except Exception:
            logger.warning(
                "execution trace sink failed while recording stage %r — "
                "continuing the experiment without live progress persistence "
                "for this event",
                execution.name,
                exc_info=True,
            )

    def finalize(self, experiment_name: str) -> ExecutionTrace:
        """Build the durable, overall trace summary from stages recorded so far."""
        if self.stages:
            started_at = self.stages[0].started_at
            ended_at = self.stages[-1].ended_at or datetime.now(UTC).isoformat()
        else:
            now = datetime.now(UTC).isoformat()
            started_at = now
            ended_at = now
        total_duration_s = round(sum(s.duration_s or 0.0 for s in self.stages), 3)
        status: Literal["completed", "failed"] = (
            "failed" if any(s.status == "failed" for s in self.stages) else "completed"
        )
        return ExecutionTrace(
            experiment_name=experiment_name,
            started_at=started_at,
            ended_at=ended_at,
            total_duration_s=total_duration_s,
            status=status,
            stages=list(self.stages),
        )


def jsonl_sink(path: str | Path, reset: bool = True) -> Callable[[dict[str, Any]], None]:
    """Return a sink that appends one flushed JSON line per event to ``path``.

    ``reset=True`` (default) truncates any existing file first, so each new
    run for a given experiment name starts a fresh live trace — consistent
    with this system executing one config at a time (no queue, no
    parallelism). Pass ``reset=False`` to keep appending to a trace an
    earlier stage of the same run already started (e.g. report generation
    appending to the trace ``run_experiment_from_config`` began).
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if reset:
        resolved.write_text("", encoding="utf-8")

    def _write(event: dict[str, Any]) -> None:
        with resolved.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
            f.flush()

    return _write


def execution_trace_jsonl_path(
    experiment_name: str, base_dir: str | Path = Path("results/execution_trace")
) -> Path:
    """Live sidecar path for a given experiment name — reset at the start of
    each new run for that name."""
    return Path(base_dir) / f"{experiment_name}.jsonl"


def write_execution_trace(out_path: str | Path, trace: ExecutionTrace) -> Path:
    """Persist the finalized trace alongside an experiment's own artefacts."""
    dest = Path(out_path) / "execution_trace.json"
    dest.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")
    return dest
