"""Low-overhead wall-clock phase telemetry for distributed DAgger.

The journal is observational only: failures to report never alter the DAgger
transaction, and records contain identifiers/timings rather than tensors.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class PhaseTelemetry:
    """Append-only start/end/error events measured with a monotonic clock."""

    def __init__(self, path: Path, *, latest_path: Path | None = None):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.latest_path = latest_path.resolve() if latest_path is not None else None
        if self.latest_path is not None:
            self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(
            self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC, 0o600
        )
        self._lock = threading.Lock()

    def emit(
        self,
        *,
        run_id: str,
        round_id: int,
        phase: str,
        event: str,
        status: str,
        completed: int,
        total: int,
        vector_step_index: int | None = None,
        actor_id: str | None = None,
        elapsed_ms: float | None = None,
        started_at_utc: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if event not in {"start", "end", "progress", "error"}:
            raise ValueError(f"invalid phase event: {event}")
        if completed < 0 or total < 0 or completed > total:
            raise ValueError("invalid phase progress")
        now = _utc_now()
        record: dict[str, Any] = {
            "schema": "roborender.dagger.phase-timing.v1",
            "run_id": run_id,
            "round_id": round_id,
            "vector_step_index": vector_step_index,
            "actor_id": actor_id,
            "phase": phase,
            "event": event,
            "status": status,
            "completed": completed,
            "total": total,
            "started_at_utc": started_at_utc,
            "finished_at_utc": now if event in {"end", "error"} else None,
            "elapsed_ms": elapsed_ms,
            "recorded_at_utc": now,
            "details": details or {},
        }
        payload = (json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) + "\n").encode("utf-8")
        with self._lock:
            offset = 0
            while offset < len(payload):
                offset += os.write(self._fd, payload[offset:])
            if self.latest_path is not None:
                temporary = self.latest_path.with_suffix(self.latest_path.suffix + ".part")
                temporary.write_bytes(payload)
                os.replace(temporary, self.latest_path)
        return record

    @contextmanager
    def span(
        self,
        *,
        run_id: str,
        round_id: int,
        phase: str,
        total: int = 1,
        vector_step_index: int | None = None,
        actor_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        started_at = _utc_now()
        started = time.perf_counter()
        mutable_details = dict(details or {})
        common = {
            "run_id": run_id,
            "round_id": round_id,
            "phase": phase,
            "total": total,
            "vector_step_index": vector_step_index,
            "actor_id": actor_id,
        }
        self.emit(
            **common, event="start", status="running", completed=0,
            started_at_utc=started_at, details=mutable_details,
        )
        try:
            yield mutable_details
        except BaseException as exc:
            self.emit(
                **common, event="error", status="failed", completed=0,
                started_at_utc=started_at,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                details={**mutable_details, "error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
        else:
            self.emit(
                **common, event="end", status="completed", completed=total,
                started_at_utc=started_at,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                details=mutable_details,
            )

    def close(self) -> None:
        with self._lock:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
