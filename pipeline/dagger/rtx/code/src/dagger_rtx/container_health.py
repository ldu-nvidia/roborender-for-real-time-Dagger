"""Read-only headless actor health; no CUDA, network, stepping or recovery.

Docker's inherited Isaac check looks for an interactive Kit AppReady log.
This check instead reads small native phase/driver snapshots. Docker's start
period covers initialization; absent progress must not masquerade as ready.
An unhealthy result is diagnostic only: it never restarts the simulator.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import time

MAX_SNAPSHOT_BYTES = 64 * 1024
PROGRESS_MAX_AGE_S = 120
CHECKPOINT_WAIT_MAX_AGE_S = 600
FAILURE_PHASES = {"CRASH_UNCERTAIN", "FAILED", "ABANDONED"}
WAIT_PHASES = {"WAITING_FOR_CHECKPOINT", "WAITING_FOR_CHECKPOINT_ADVANCE"}


def _read(path: Path, run_id: str, now: float):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_SNAPSHOT_BYTES:
        raise ValueError("unsafe health snapshot")
    age = now - info.st_mtime
    if age < -5:
        raise ValueError("health snapshot is from the future")
    with path.open("rb") as stream:
        raw = stream.read(MAX_SNAPSHOT_BYTES + 1)
    if len(raw) > MAX_SNAPSHOT_BYTES:
        raise ValueError("oversized health snapshot")
    row = json.loads(raw)
    # The production driver stores run identity inside its immutable spec.
    identity = row.get("run_id", row.get("spec", {}).get("run_id"))
    if identity != run_id:
        raise ValueError("health snapshot run mismatch")
    return row, max(age, 0.0)


def check(root: Path, run_id: str, *, now=None) -> dict:
    if not run_id or not root.is_absolute():
        return {"ok": False, "reason": "missing_actor_identity"}
    now = time.time() if now is None else now
    try:
        drivers = [value for name in ("qualification-state.json", "production-driver-state.json", "beta-round-state.json")
                   if (value := _read(root / name, run_id, now)) is not None]
        if any(row.get("phase") in FAILURE_PHASES or row.get("uncertain_reason")
               for row, _ in drivers):
            return {"ok": False, "reason": "native_driver_failed_closed"}
        progress = _read(root / "progress.json", run_id, now)
        if progress is not None:
            row, age = progress
            if row.get("schema") != "roborender.dagger.rtx.phase-boundary.v1":
                raise ValueError("unexpected progress schema")
            if row.get("event") == "error" or row.get("status") in {"failed", "error"}:
                return {"ok": False, "reason": "native_phase_error"}
            if age <= PROGRESS_MAX_AGE_S:
                return {"ok": True, "phase": row.get("phase"), "progress_age_s": round(age, 3)}
        for row, age in drivers:
            if row.get("phase") in WAIT_PHASES and age <= CHECKPOINT_WAIT_MAX_AGE_S:
                return {"ok": True, "phase": row["phase"], "wait_age_s": round(age, 3)}
        return {"ok": False, "reason": "initializing_or_stale_native_progress"}
    except (OSError, ValueError, TypeError, AttributeError):
        return {"ok": False, "reason": "invalid_native_health_snapshot"}


def main() -> int:
    result = check(Path(os.environ.get("DAGGER_RUNTIME_ROOT", ".")),
                   os.environ.get("DAGGER_RUN_ID", ""))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
