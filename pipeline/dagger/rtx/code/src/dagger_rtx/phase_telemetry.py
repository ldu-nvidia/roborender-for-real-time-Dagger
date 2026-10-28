"""Low-volume wall-clock phase boundaries for the native vector DAgger loop.

This stream is deliberately separate from per-route capacity telemetry.  It
emits only vector-level scalar metadata: never arrays, payloads, tokens, or
per-lane records.  A JSONL event is append-only and ``progress.json`` is the
atomically replaced latest snapshot.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

from .contracts import ROUND_BETAS, canonical_json, require_identifier
from .durable import atomic_write_json


PHASE_BOUNDARY_SCHEMA = "roborender.dagger.rtx.phase-boundary.v1"
PHASE_EVENTS = frozenset({"start", "end", "progress", "error"})
_MAX_COUNTS = 64
_MAX_COUNT_STRING = 256


class PhaseTelemetryError(RuntimeError):
    """The phase stream is malformed or cannot preserve its invariants."""


def _utc_from_ns(value: int) -> str:
    return (
        datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _small_counts(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Accept only a bounded flat scalar map so telemetry cannot copy tensors."""

    if value is None:
        return {}
    if not isinstance(value, Mapping) or len(value) > _MAX_COUNTS:
        raise PhaseTelemetryError("counts must be a small mapping")
    result: dict[str, Any] = {}
    for raw_name, item in value.items():
        if not isinstance(raw_name, str) or not raw_name or len(raw_name) > 96:
            raise PhaseTelemetryError("count names must be short non-empty strings")
        if isinstance(item, float) and not math.isfinite(item):
            raise PhaseTelemetryError(f"count {raw_name} must be finite")
        if isinstance(item, str) and len(item) > _MAX_COUNT_STRING:
            raise PhaseTelemetryError(f"count {raw_name} string is too long")
        if item is not None and not isinstance(item, (bool, int, float, str)):
            raise PhaseTelemetryError(
                f"count {raw_name} must be a JSON scalar; arrays/objects are forbidden"
            )
        result[raw_name] = item
    return result


class PhaseBoundaryTelemetry:
    """Append vector-level start/end/error events and atomically publish latest state.

    Durations always use ``monotonic_ns``. UTC timestamps are presentation and
    correlation fields only. JSONL appends are intentionally not fsynced; the
    atomic progress snapshot is the durable latest view. Work is vector-level,
    never per lane.
    """

    def __init__(
        self,
        journal_path: Path,
        progress_path: Path,
        *,
        run_id: str,
        round_id: int,
        total_vector_steps: int,
        batch_size: int,
        clock_ns=time.time_ns,
        monotonic_ns=time.perf_counter_ns,
        fsync_events: bool = False,
        event_observer: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.run_id = require_identifier(run_id, "run_id")
        if round_id not in ROUND_BETAS:
            raise PhaseTelemetryError("round_id must be in [1,4]")
        if (
            isinstance(total_vector_steps, bool)
            or not isinstance(total_vector_steps, int)
            or total_vector_steps < 1
        ):
            raise PhaseTelemetryError("total_vector_steps must be positive")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise PhaseTelemetryError("batch_size must be positive")
        if type(fsync_events) is not bool:
            raise PhaseTelemetryError("fsync_events must be bool")
        self.round_id = round_id
        self.total_vector_steps = total_vector_steps
        self.batch_size = batch_size
        self.journal_path = Path(journal_path)
        self.progress_path = Path(progress_path)
        self.clock_ns = clock_ns
        self.monotonic_ns = monotonic_ns
        self.fsync_events = fsync_events
        self.event_observer = event_observer
        self._lock = threading.RLock()
        self._active: dict[tuple[int, str], dict[str, Any]] = {}
        self._completed: set[tuple[int, str]] = set()
        self._attempts: Counter[tuple[int, str]] = Counter()
        self._completion_counts: Counter[str] = Counter()
        self._sequence = 0
        # Native step identities are window-local (0..80), while a persistent
        # production process needs one monotonic telemetry/W&B axis across all
        # windows.  The production driver changes this only between windows.
        self._vector_step_offset = 0
        self.journal_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.progress_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._restore_completed()
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
        self._fd = os.open(self.journal_path, flags, 0o600)
        os.chmod(self.journal_path, 0o600)

    def _restore_completed(self) -> None:
        if not self.journal_path.exists():
            return
        with self.journal_path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PhaseTelemetryError(
                        f"corrupt phase telemetry line {line_no}: {self.journal_path}"
                    ) from exc
                if row.get("schema") != PHASE_BOUNDARY_SCHEMA:
                    raise PhaseTelemetryError("phase telemetry schema mismatch")
                if row.get("run_id") != self.run_id or row.get("round_id") != self.round_id:
                    raise PhaseTelemetryError("phase telemetry run identity mismatch")
                self._sequence = max(self._sequence, int(row.get("sequence", 0)))
                token = (int(row["vector_step"]), str(row["phase"]))
                if row.get("event") == "end" and token not in self._completed:
                    self._completed.add(token)
                    self._completion_counts[token[1]] += 1

    def close(self) -> None:
        with self._lock:
            fd = getattr(self, "_fd", -1)
            if fd >= 0:
                os.close(fd)
                self._fd = -1

    def __enter__(self) -> "PhaseBoundaryTelemetry":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _validate_common(self, vector_step: int, phase: str, completed: int, total: int) -> None:
        if (
            isinstance(vector_step, bool)
            or not isinstance(vector_step, int)
            or not 0 <= vector_step < self.total_vector_steps
        ):
            raise PhaseTelemetryError("vector_step is outside the configured window")
        require_identifier(phase, "phase", 96)
        if (
            isinstance(completed, bool)
            or not isinstance(completed, int)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or total < 1
            or not 0 <= completed <= total
        ):
            raise PhaseTelemetryError("completed/total must satisfy 0 <= completed <= total")

    def set_vector_step_offset(self, offset: int) -> None:
        """Map subsequent window-local step indices onto a global run axis.

        Changing scope while a phase is active would split a start/end pair, so
        it is rejected.  Existing single-window callers retain the zero default.
        """

        with self._lock:
            if (
                isinstance(offset, bool)
                or not isinstance(offset, int)
                or offset < 0
                or offset >= self.total_vector_steps
            ):
                raise PhaseTelemetryError("vector-step offset is outside the configured run")
            if self._active:
                raise PhaseTelemetryError("cannot change vector-step offset with active phases")
            self._vector_step_offset = offset

    def _absolute_vector_step(self, vector_step: int) -> int:
        if isinstance(vector_step, bool) or not isinstance(vector_step, int) or vector_step < 0:
            raise PhaseTelemetryError("vector_step must be a non-negative integer")
        return self._vector_step_offset + vector_step

    def _append_locked(self, row: dict[str, Any]) -> None:
        if self._fd < 0:
            raise PhaseTelemetryError("phase telemetry is closed")
        data = canonical_json(row) + b"\n"
        offset = 0
        while offset < len(data):
            written = os.write(self._fd, data[offset:])
            if written <= 0:
                raise PhaseTelemetryError("short phase telemetry write")
            offset += written
        if self.fsync_events:
            os.fsync(self._fd)
        snapshot = {
            **row,
            "batch_size": self.batch_size,
            "total_vector_steps": self.total_vector_steps,
            "completed_phase_instances": len(self._completed),
            "phase_completion_counts": dict(sorted(self._completion_counts.items())),
            "updated_at_utc": row["finished_at_utc"] or row["started_at_utc"],
        }
        atomic_write_json(self.progress_path, snapshot)
        if self.event_observer is not None:
            self.event_observer(row)

    def _row(
        self,
        *,
        vector_step: int,
        phase: str,
        event: str,
        started_ns: int,
        finished_ns: int | None,
        elapsed_ns: int | None,
        completed: int,
        total: int,
        status: str,
        counts: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if event not in PHASE_EVENTS:
            raise PhaseTelemetryError(f"event must be one of {sorted(PHASE_EVENTS)}")
        self._validate_common(vector_step, phase, completed, total)
        require_identifier(status, "status", 96)
        self._sequence += 1
        return {
            "schema": PHASE_BOUNDARY_SCHEMA,
            "sequence": self._sequence,
            "run_id": self.run_id,
            "round_id": self.round_id,
            "vector_step": vector_step,
            "phase": phase,
            "event": event,
            "started_at_utc": _utc_from_ns(started_ns),
            "finished_at_utc": _utc_from_ns(finished_ns) if finished_ns is not None else None,
            "elapsed_ms": round(elapsed_ns / 1_000_000, 6) if elapsed_ns is not None else None,
            "completed": completed,
            "total": total,
            "status": status,
            "counts": _small_counts(counts),
        }

    def start(
        self,
        vector_step: int,
        phase: str,
        *,
        total: int,
        counts: Mapping[str, Any] | None = None,
    ) -> bool:
        """Start one phase attempt; return False when it was already completed."""

        with self._lock:
            vector_step = self._absolute_vector_step(vector_step)
            self._validate_common(vector_step, phase, 0, total)
            token = (vector_step, phase)
            now_wall = int(self.clock_ns())
            now_mono = int(self.monotonic_ns())
            if token in self._completed:
                replay_counts = {**_small_counts(counts), "replayed": True}
                row = self._row(
                    vector_step=vector_step,
                    phase=phase,
                    event="progress",
                    started_ns=now_wall,
                    finished_ns=now_wall,
                    elapsed_ns=0,
                    completed=total,
                    total=total,
                    status="already_complete",
                    counts=replay_counts,
                )
                self._append_locked(row)
                return False
            if token in self._active:
                raise PhaseTelemetryError(f"phase {phase} is already active at step {vector_step}")
            self._attempts[token] += 1
            attempt = self._attempts[token]
            self._active[token] = {
                "wall_ns": now_wall,
                "monotonic_ns": now_mono,
                "total": total,
                "attempt": attempt,
            }
            row = self._row(
                vector_step=vector_step,
                phase=phase,
                event="start",
                started_ns=now_wall,
                finished_ns=None,
                elapsed_ns=None,
                completed=0,
                total=total,
                status="running",
                counts={**_small_counts(counts), "attempt": attempt},
            )
            self._append_locked(row)
            return True

    def progress(
        self,
        vector_step: int,
        phase: str,
        *,
        completed: int,
        total: int,
        status: str = "running",
        counts: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            vector_step = self._absolute_vector_step(vector_step)
            self._validate_common(vector_step, phase, completed, total)
            token = (vector_step, phase)
            active = self._active.get(token)
            now_wall = int(self.clock_ns())
            now_mono = int(self.monotonic_ns())
            started_wall = int(active["wall_ns"]) if active else now_wall
            elapsed = now_mono - int(active["monotonic_ns"]) if active else 0
            attempt = int(active["attempt"]) if active else int(self._attempts[token])
            row = self._row(
                vector_step=vector_step,
                phase=phase,
                event="progress",
                started_ns=started_wall,
                finished_ns=None,
                elapsed_ns=elapsed,
                completed=completed,
                total=total,
                status=status,
                counts={**_small_counts(counts), "attempt": attempt},
            )
            self._append_locked(row)

    def end(
        self,
        vector_step: int,
        phase: str,
        *,
        completed: int | None = None,
        total: int,
        status: str = "complete",
        counts: Mapping[str, Any] | None = None,
    ) -> bool:
        with self._lock:
            vector_step = self._absolute_vector_step(vector_step)
            token = (vector_step, phase)
            if token in self._completed:
                now = int(self.clock_ns())
                row = self._row(
                    vector_step=vector_step,
                    phase=phase,
                    event="progress",
                    started_ns=now,
                    finished_ns=now,
                    elapsed_ns=0,
                    completed=total,
                    total=total,
                    status="already_complete",
                    counts={**_small_counts(counts), "replayed": True},
                )
                self._append_locked(row)
                return False
            active = self._active.pop(token, None)
            if active is None:
                raise PhaseTelemetryError(f"phase {phase} was not started at step {vector_step}")
            if int(active["total"]) != total:
                raise PhaseTelemetryError("phase total changed between start and end")
            now_wall = int(self.clock_ns())
            elapsed = int(self.monotonic_ns()) - int(active["monotonic_ns"])
            done = total if completed is None else completed
            self._validate_common(vector_step, phase, done, total)
            row = self._row(
                vector_step=vector_step,
                phase=phase,
                event="end",
                started_ns=int(active["wall_ns"]),
                finished_ns=now_wall,
                elapsed_ns=elapsed,
                completed=done,
                total=total,
                status=status,
                counts={**_small_counts(counts), "attempt": int(active["attempt"])},
            )
            self._completed.add(token)
            self._completion_counts[phase] += 1
            self._append_locked(row)
            return True

    def error(
        self,
        vector_step: int,
        phase: str,
        exc: BaseException,
        *,
        completed: int,
        total: int,
        counts: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            vector_step = self._absolute_vector_step(vector_step)
            token = (vector_step, phase)
            active = self._active.pop(token, None)
            now_wall = int(self.clock_ns())
            now_mono = int(self.monotonic_ns())
            started_wall = int(active["wall_ns"]) if active else now_wall
            elapsed = now_mono - int(active["monotonic_ns"]) if active else 0
            attempt = int(active["attempt"]) if active else int(self._attempts[token])
            row = self._row(
                vector_step=vector_step,
                phase=phase,
                event="error",
                started_ns=started_wall,
                finished_ns=now_wall,
                elapsed_ns=elapsed,
                completed=completed,
                total=total,
                status="error",
                counts={
                    **_small_counts(counts),
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:_MAX_COUNT_STRING],
                },
            )
            self._append_locked(row)
