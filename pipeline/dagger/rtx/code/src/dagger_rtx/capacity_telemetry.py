"""Append-only, credential-free telemetry for bounded capacity probes."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Protocol

from .durable import DurableJournal


CAPACITY_TELEMETRY_SCHEMA = "roborender.dagger.rtx.capacity-telemetry.v1"
RUN_TELEMETRY_SCHEMA = "roborender.dagger.rtx.run-telemetry.v1"


class EventSink(Protocol):
    def record(self, event: str, **detail: Any) -> None: ...


class AppendOnlyTelemetry:
    """Thread-safe JSONL telemetry suitable for concurrent HTTP fanout.

    Events deliberately contain route identities and measurements only.  The
    bearer token and full image/control payloads are never copied into this
    diagnostic stream.
    """

    def __init__(
        self,
        path: Path,
        *,
        clock_ns=time.time_ns,
        schema: str = CAPACITY_TELEMETRY_SCHEMA,
    ) -> None:
        self.path = Path(path)
        self.journal = DurableJournal(self.path)
        self.clock_ns = clock_ns
        self.schema = schema
        self._lock = threading.Lock()

    def record(self, event: str, **detail: Any) -> None:
        row = {
            "schema": self.schema,
            "time_ns": int(self.clock_ns()),
            "event": event,
            **detail,
        }
        # DurableJournal opens with O_APPEND, while this lock also keeps one
        # process's concurrently generated records in a deterministic order.
        with self._lock:
            self.journal.append(row)

    def read(self) -> list[dict[str, Any]]:
        with self._lock:
            return self.journal.read()


def record_if_present(sink: EventSink | None, event: str, **detail: Any) -> None:
    if sink is not None:
        sink.record(event, **detail)
