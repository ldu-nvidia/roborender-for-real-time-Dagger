#!/usr/bin/env python3
"""Start the production evaluation W&B bridge as a detached process."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.round_dagger import start_student_quality_wandb_bridge as process  # noqa: E402


BRIDGE = ROOT / "scripts/round_dagger/evaluation_wandb_bridge.py"
PID_SCHEMA = "fast_roborender.round_dagger.evaluation_bridge_process.v1"
READY_SCHEMA = "fast_roborender.round_dagger.evaluation_wandb_ready.v1"
PID_KEYS = {
    "schema",
    "pid",
    "start_ticks",
    "entrypoint",
    "campaign_root",
    "state_dir",
    "driver_status",
}
StarterError = process.StarterError


def _existing_process(pidfile: Path) -> tuple[int, int] | None:
    if not pidfile.exists() and not pidfile.is_symlink():
        return None
    process._regular(pidfile, "evaluation bridge pidfile")
    try:
        document = json.loads(pidfile.read_text(encoding="utf-8"))
        if set(document) != PID_KEYS or document["schema"] != PID_SCHEMA:
            raise ValueError
        pid = document["pid"]
        start_ticks = document["start_ticks"]
        for key in ("entrypoint", "campaign_root", "state_dir", "driver_status"):
            if not isinstance(document[key], str) or not document[key]:
                raise ValueError
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise StarterError("evaluation bridge pidfile is malformed") from None
    if type(pid) is int and type(start_ticks) is int:
        if process._process_start_ticks(pid) == start_ticks:
            return pid, start_ticks
    return None


def start_bridge(
    *,
    python: Path,
    campaign_root: Path,
    state_dir: Path,
    driver_status: Path,
    project: str,
    entity: str,
    expected_email: str,
    poll_seconds: float,
    pidfile: Path,
    logfile: Path,
    ready_timeout_seconds: float = 45.0,
) -> dict[str, object]:
    """Validate and detach one lifecycle-aware production evaluation bridge."""

    process._executable_entrypoint(python, "evaluation bridge Python")
    process._regular(BRIDGE, "evaluation bridge entrypoint")
    process._directory(campaign_root, "campaign root")
    process._regular(driver_status, "production driver status")
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise StarterError("production evaluation bridge poll interval must be positive")
    if not project or not entity or not expected_email:
        raise StarterError("W&B project, entity, and expected email are required")
    if not math.isfinite(ready_timeout_seconds) or ready_timeout_seconds <= 0:
        raise StarterError("evaluation bridge ready timeout must be positive")
    if _existing_process(pidfile) is not None:
        raise StarterError(
            "an evaluation bridge recorded by this pidfile is already running"
        )
    try:
        state_dir.relative_to(campaign_root)
    except ValueError:
        pass
    else:
        raise StarterError("evaluation bridge state must be outside the campaign root")
    state_dir.mkdir(parents=True, exist_ok=True)
    if state_dir.is_symlink() or not state_dir.is_dir():
        raise StarterError("evaluation bridge state directory is unsafe")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    if logfile.exists() or logfile.is_symlink():
        process._regular(logfile, "evaluation bridge logfile")

    ready_marker = state_dir / "EVALUATION_WANDB_READY.json"
    if ready_marker.exists() or ready_marker.is_symlink():
        process._regular(ready_marker, "evaluation bridge readiness marker")
        ready_marker.unlink()
        directory = os.open(state_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    command = [
        str(python),
        str(BRIDGE),
        "--campaign-root",
        str(campaign_root),
        "--state-dir",
        str(state_dir),
        "--project",
        project,
        "--entity",
        entity,
        "--expected-email",
        expected_email,
        "--authorization-scope",
        "team",
        "--driver-status",
        str(driver_status),
        "--poll-seconds",
        str(poll_seconds),
        "--ready-marker",
        str(ready_marker),
    ]
    descriptor = os.open(
        logfile,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_APPEND
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(descriptor)
    try:
        deadline = time.monotonic() + ready_timeout_seconds
        ready_document: dict[str, object] | None = None
        while time.monotonic() < deadline:
            return_code = child.poll()
            if return_code is not None:
                raise StarterError(
                    f"evaluation bridge exited during startup with status {return_code}"
                )
            try:
                candidate = json.loads(ready_marker.read_text(encoding="utf-8"))
            except FileNotFoundError:
                candidate = None
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise StarterError(
                    "evaluation bridge readiness marker is invalid"
                ) from exc
            if isinstance(candidate, dict):
                if (
                    candidate.get("schema") != READY_SCHEMA
                    or candidate.get("process_id") != child.pid
                    or not isinstance(candidate.get("wandb_run_id"), str)
                ):
                    raise StarterError(
                        "evaluation bridge readiness marker does not match child"
                    )
                ready_document = candidate
                break
            time.sleep(0.1)
        if ready_document is None:
            raise StarterError("evaluation bridge readiness timed out")
        start_ticks = process._process_start_ticks(child.pid)
        if start_ticks is None:
            raise StarterError("cannot bind detached evaluation bridge process identity")
        process._atomic_json(
            pidfile,
            {
                "schema": PID_SCHEMA,
                "pid": child.pid,
                "start_ticks": start_ticks,
                "entrypoint": str(BRIDGE),
                "campaign_root": str(campaign_root),
                "state_dir": str(state_dir),
                "driver_status": str(driver_status),
            },
        )
        return {
            "status": "STARTED",
            "pid": child.pid,
            "start_ticks": start_ticks,
            "pidfile": str(pidfile),
            "logfile": str(logfile),
            "campaign_root": str(campaign_root),
            "driver_status": str(driver_status),
            "poll_seconds": poll_seconds,
            "ready_marker": str(ready_marker),
            "wandb_run_id": ready_document["wandb_run_id"],
            "ready": True,
            "lifecycle_aware": True,
        }
    except BaseException:
        process._terminate_child(child)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--driver-status", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--expected-email", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--pidfile", type=Path, required=True)
    parser.add_argument("--logfile", type=Path, required=True)
    parser.add_argument("--ready-timeout-seconds", type=float, default=45.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = start_bridge(
            python=process._absolute(arguments.python, "evaluation bridge Python"),
            campaign_root=process._absolute(arguments.campaign_root, "campaign root"),
            state_dir=process._absolute(arguments.state_dir, "evaluation bridge state"),
            driver_status=process._absolute(
                arguments.driver_status, "production driver status"
            ),
            project=arguments.project,
            entity=arguments.entity,
            expected_email=arguments.expected_email,
            poll_seconds=arguments.poll_seconds,
            pidfile=process._absolute(arguments.pidfile, "evaluation bridge pidfile"),
            logfile=process._absolute(arguments.logfile, "evaluation bridge logfile"),
            ready_timeout_seconds=arguments.ready_timeout_seconds,
        )
    except StarterError as exc:
        print(f"evaluation bridge starter failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
