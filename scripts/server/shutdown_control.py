#!/usr/bin/env python3
"""Cooperative, file-backed shutdown control for persistent render servers."""

from __future__ import annotations

import os
import hashlib
import json
import stat
import threading
import tempfile
from datetime import datetime, timezone
from pathlib import Path


MAX_STOP_POLL_SECONDS = 2.0
DEFAULT_STOP_POLL_SECONDS = 1.0


class StopController:
    """Watch an optional stop sentinel without coupling the server to Slurm."""

    def __init__(
        self, path: Path | None, ack_path: Path | None, poll_seconds: float
    ) -> None:
        if not 0.0 < poll_seconds <= MAX_STOP_POLL_SECONDS:
            raise ValueError(
                "IMAGE3F_STOP_POLL_S must be greater than zero and at most "
                f"{MAX_STOP_POLL_SECONDS:g} seconds"
            )
        self.path = path
        self.ack_path = ack_path
        self.poll_seconds = poll_seconds
        self.error: BaseException | None = None
        if self.path is not None and self.ack_path is None:
            raise ValueError(
                "IMAGE3F_STOP_ACK_FILE is required when IMAGE3F_STOP_FILE is set"
            )

    @classmethod
    def from_environment(cls) -> "StopController":
        raw_path = os.environ.get("IMAGE3F_STOP_FILE", "").strip()
        raw_ack_path = os.environ.get("IMAGE3F_STOP_ACK_FILE", "").strip()
        raw_poll = os.environ.get(
            "IMAGE3F_STOP_POLL_S", str(DEFAULT_STOP_POLL_SECONDS)
        )
        return cls(
            Path(raw_path) if raw_path else None,
            Path(raw_ack_path) if raw_ack_path else None,
            float(raw_poll),
        )

    def requested(self) -> bool:
        if self.path is None:
            return False
        try:
            mode = self.path.stat(follow_symlinks=False).st_mode
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"Image3F stop sentinel must not be a symlink: {self.path}")
        if not stat.S_ISREG(mode):
            raise RuntimeError(
                f"Image3F stop sentinel must be a regular file: {self.path}"
            )
        return True

    def watch(self, stopped: threading.Event) -> None:
        """Set ``stopped`` within one configured polling interval."""

        try:
            while not stopped.is_set():
                if self.requested():
                    stopped.set()
                    return
                stopped.wait(self.poll_seconds)
        except BaseException as exc:
            self.error = exc
            stopped.set()

    def acknowledge(self, server_id: str) -> dict[str, object]:
        """Atomically attest that this server observed the exact STOP file."""

        if self.path is None or self.ack_path is None:
            raise RuntimeError("cannot acknowledge a disabled Image3F stop controller")
        if not self.requested():
            raise RuntimeError("cannot acknowledge a missing Image3F stop sentinel")
        sentinel = self.path.read_bytes()
        payload: dict[str, object] = {
            "schema": "fast_roborender.image3f_stop_ack.v1",
            "status": "STOP_observed",
            "server_id": server_id,
            "pid": os.getpid(),
            "stop_sentinel_path": str(self.path.resolve(strict=True)),
            "stop_sentinel_sha256": hashlib.sha256(sentinel).hexdigest(),
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
        destination = self.ack_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"Image3F stop acknowledgement already exists: {destination}")
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return payload
