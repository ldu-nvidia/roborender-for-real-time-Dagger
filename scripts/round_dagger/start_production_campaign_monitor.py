#!/usr/bin/env python3
"""Detach the read-only campaign monitor from an operator terminal safely."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MONITOR = ROOT / "scripts/round_dagger/monitor_production_campaign.py"
_SENSITIVE_RE = re.compile(
    r"(?i)(api[_-]?key|password|secret|access[_-]?token|identity[_-]?token)"
)


class StarterError(RuntimeError):
    pass


def _absolute(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    if not path.is_absolute():
        raise StarterError(f"{label} must be absolute")
    return path


def _process_start_ticks(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def _atomic_pidfile(path: Path, pid: int, start_ticks: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            {"schema": "fast_roborender.round_dagger.monitor_process.v1", "pid": pid, "start_ticks": start_ticks},
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _existing_monitor(pidfile: Path) -> tuple[int, int] | None:
    try:
        info = os.lstat(pidfile)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise StarterError("monitor pidfile is unsafe")
    try:
        document = json.loads(pidfile.read_text(encoding="utf-8"))
        pid = document["pid"]
        ticks = document["start_ticks"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StarterError("existing monitor pidfile is invalid") from exc
    if type(pid) is int and type(ticks) is int and _process_start_ticks(pid) == ticks:
        return pid, ticks
    return None


def start_monitor(
    *,
    python: Path,
    pidfile: Path,
    logfile: Path,
    monitor_arguments: list[str],
) -> dict[str, object]:
    if _existing_monitor(pidfile) is not None:
        raise StarterError("a monitor recorded by this pidfile is already running")
    try:
        info = os.lstat(python)
    except FileNotFoundError as exc:
        raise StarterError(f"monitor Python is missing: {python}") from exc
    if not stat.S_ISREG(info.st_mode) or not os.access(python, os.X_OK):
        raise StarterError("monitor Python must be an executable regular file")
    forbidden = {"--once", "--test-max-polls", "--test-poll-interval-seconds"}
    if forbidden.intersection(monitor_arguments):
        raise StarterError("durable monitor cannot use bounded/test polling flags")
    if any(_SENSITIVE_RE.search(argument) for argument in monitor_arguments):
        raise StarterError("monitor argv appears to contain a credential")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    try:
        log_info = os.lstat(logfile)
    except FileNotFoundError:
        log_info = None
    if log_info is not None and not stat.S_ISREG(log_info.st_mode):
        raise StarterError("monitor logfile is unsafe")
    log_descriptor = os.open(
        logfile,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_APPEND
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        child = subprocess.Popen(
            [str(python), str(MONITOR), *monitor_arguments],
            stdin=subprocess.DEVNULL,
            stdout=log_descriptor,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(log_descriptor)
    # Detect immediate argument/configuration failures without keeping the
    # starter attached to the long-lived observer.
    try:
        return_code = child.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        return_code = None
    if return_code is not None:
        raise StarterError(f"monitor exited during startup with status {return_code}")
    start_ticks = _process_start_ticks(child.pid)
    if start_ticks is None:
        raise StarterError("cannot bind detached monitor process identity")
    _atomic_pidfile(pidfile, child.pid, start_ticks)
    return {
        "status": "STARTED",
        "pid": child.pid,
        "start_ticks": start_ticks,
        "pidfile": str(pidfile),
        "logfile": str(logfile),
        "monitor": str(MONITOR),
        "interval_seconds": 1800,
        "read_only": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--pidfile", type=Path, required=True)
    parser.add_argument("--logfile", type=Path, required=True)
    parser.add_argument("monitor_arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    monitor_arguments = list(arguments.monitor_arguments)
    if monitor_arguments[:1] == ["--"]:
        monitor_arguments = monitor_arguments[1:]
    if not monitor_arguments:
        print("monitor starter failed: monitor arguments are required after --", file=sys.stderr)
        return 2
    try:
        report = start_monitor(
            python=_absolute(arguments.python, "monitor Python"),
            pidfile=_absolute(arguments.pidfile, "pidfile"),
            logfile=_absolute(arguments.logfile, "logfile"),
            monitor_arguments=monitor_arguments,
        )
    except StarterError as exc:
        print(f"monitor starter failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
