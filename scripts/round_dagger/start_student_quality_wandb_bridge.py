#!/usr/bin/env python3
"""Start the production student-quality W&B bridge as a detached process.

The starter deliberately requires the production driver's durable status file
and always passes it to the bridge.  This prevents a deployment from creating a
quality dashboard that cannot distinguish an active campaign from a failed one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BRIDGE = ROOT / "scripts/round_dagger/student_quality_wandb_bridge.py"
PID_SCHEMA = "fast_roborender.round_dagger.student_quality_bridge_process.v1"
READY_SCHEMA = "fast_roborender.round_dagger.student_quality_wandb_ready.v1"
PID_KEYS = {
    "schema",
    "pid",
    "start_ticks",
    "entrypoint",
    "campaign_root",
    "state_dir",
    "driver_status",
}


class StarterError(RuntimeError):
    """A detached quality bridge could not be started safely."""


def _absolute(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    if not path.is_absolute():
        raise StarterError(f"{label} must be absolute")
    return path


def _regular(path: Path, label: str) -> Path:
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise StarterError(f"{label} is missing") from exc
    if not stat.S_ISREG(info.st_mode):
        raise StarterError(f"{label} must be a regular non-symlink file")
    return path


def _executable_entrypoint(path: Path, label: str) -> Path:
    """Accept a conventional interpreter symlink only when its target is safe."""

    if not path.exists() or not os.access(path, os.X_OK):
        raise StarterError(f"{label} is missing or not executable")
    try:
        resolved = path.resolve(strict=True)
        info = os.lstat(resolved)
    except OSError as exc:
        raise StarterError(f"{label} cannot be resolved") from exc
    if not stat.S_ISREG(info.st_mode) or not os.access(resolved, os.X_OK):
        raise StarterError(f"{label} must resolve to a regular executable")
    return path


def _directory(path: Path, label: str) -> Path:
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise StarterError(f"{label} is missing") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise StarterError(f"{label} must be a non-symlink directory")
    return path


def _process_start_ticks(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        if fields[2] == "Z":
            return None
        return int(fields[21])
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def _atomic_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as output:
            json.dump(document, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _existing_process(pidfile: Path) -> tuple[int, int] | None:
    if not pidfile.exists() and not pidfile.is_symlink():
        return None
    _regular(pidfile, "quality bridge pidfile")
    try:
        document = json.loads(pidfile.read_text(encoding="utf-8"))
        if set(document) != PID_KEYS:
            raise ValueError
        if document["schema"] != PID_SCHEMA:
            raise ValueError
        pid = document["pid"]
        start_ticks = document["start_ticks"]
        for key in ("entrypoint", "campaign_root", "state_dir", "driver_status"):
            if not isinstance(document[key], str) or not document[key]:
                raise ValueError
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        raise StarterError("quality bridge pidfile is malformed") from None
    if type(pid) is int and type(start_ticks) is int:
        if _process_start_ticks(pid) == start_ticks:
            return pid, start_ticks
    return None


def _terminate_child(child: subprocess.Popen[bytes]) -> None:
    """Reap the exact child created by a failed startup transaction."""

    try:
        running = child.poll() is None
    except Exception:
        running = True
    if running:
        try:
            child.terminate()
        except (OSError, ProcessLookupError):
            pass
    try:
        child.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            child.kill()
        except (OSError, ProcessLookupError):
            pass
        try:
            child.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            pass
    except OSError:
        pass


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
    """Validate and detach one lifecycle-aware production quality bridge."""

    _executable_entrypoint(python, "quality bridge Python")
    _regular(BRIDGE, "quality bridge entrypoint")
    _directory(campaign_root, "campaign root")
    _regular(driver_status, "production driver status")
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise StarterError("production quality bridge poll interval must be positive")
    if not project or not entity or not expected_email:
        raise StarterError("W&B project, entity, and expected email are required")
    if not math.isfinite(ready_timeout_seconds) or ready_timeout_seconds <= 0:
        raise StarterError("quality bridge ready timeout must be positive")
    if _existing_process(pidfile) is not None:
        raise StarterError("a quality bridge recorded by this pidfile is already running")
    try:
        state_dir.relative_to(campaign_root)
    except ValueError:
        pass
    else:
        raise StarterError("quality bridge state must be outside the campaign root")
    state_dir.mkdir(parents=True, exist_ok=True)
    if state_dir.is_symlink() or not state_dir.is_dir():
        raise StarterError("quality bridge state directory is unsafe")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    if logfile.exists() or logfile.is_symlink():
        _regular(logfile, "quality bridge logfile")

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
    ]
    ready_marker = state_dir / "STUDENT_QUALITY_WANDB_READY.json"
    if ready_marker.exists() or ready_marker.is_symlink():
        _regular(ready_marker, "quality bridge readiness marker")
        ready_marker.unlink()
        directory = os.open(state_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    command.extend(("--ready-marker", str(ready_marker)))
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
                    f"quality bridge exited during startup with status {return_code}"
                )
            try:
                candidate = json.loads(ready_marker.read_text(encoding="utf-8"))
            except FileNotFoundError:
                candidate = None
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise StarterError("quality bridge readiness marker is invalid") from exc
            if isinstance(candidate, dict):
                if (
                    candidate.get("schema") != READY_SCHEMA
                    or candidate.get("process_id") != child.pid
                    or not isinstance(candidate.get("wandb_run_id"), str)
                ):
                    raise StarterError(
                        "quality bridge readiness marker does not match child"
                    )
                ready_document = candidate
                break
            time.sleep(0.1)
        if ready_document is None:
            raise StarterError("quality bridge readiness timed out")
        start_ticks = _process_start_ticks(child.pid)
        if start_ticks is None:
            raise StarterError("cannot bind detached quality bridge process identity")
        _atomic_json(
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
        _terminate_child(child)
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
            python=_absolute(arguments.python, "quality bridge Python"),
            campaign_root=_absolute(arguments.campaign_root, "campaign root"),
            state_dir=_absolute(arguments.state_dir, "quality bridge state"),
            driver_status=_absolute(arguments.driver_status, "production driver status"),
            project=arguments.project,
            entity=arguments.entity,
            expected_email=arguments.expected_email,
            poll_seconds=arguments.poll_seconds,
            pidfile=_absolute(arguments.pidfile, "quality bridge pidfile"),
            logfile=_absolute(arguments.logfile, "quality bridge logfile"),
            ready_timeout_seconds=arguments.ready_timeout_seconds,
        )
    except StarterError as exc:
        print(f"quality bridge starter failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
