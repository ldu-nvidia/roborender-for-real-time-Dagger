#!/usr/bin/env python3
"""Detach the read-only campaign monitor from an operator terminal safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.round_dagger import start_evaluation_wandb_bridge as evaluation_starter  # noqa: E402
from scripts.round_dagger import start_student_quality_wandb_bridge as quality_starter  # noqa: E402


MONITOR = ROOT / "scripts/round_dagger/monitor_production_campaign.py"
MONITOR_PID_SCHEMA = "fast_roborender.round_dagger.monitor_process.v2"
MONITOR_READY_SCHEMA = (
    "fast_roborender.round_dagger.production_monitor_ready.v1"
)
_SENSITIVE_RE = re.compile(
    r"(?i)(api[_-]?key|password|secret|access[_-]?token|identity[_-]?token)"
)


StarterError = quality_starter.StarterError


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


def _atomic_pidfile(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(document, sort_keys=True)
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
        if not isinstance(document, dict) or document.get("schema") not in {
            "fast_roborender.round_dagger.monitor_process.v1",
            MONITOR_PID_SCHEMA,
        }:
            raise ValueError
        pid = document["pid"]
        ticks = document["start_ticks"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StarterError("existing monitor pidfile is invalid") from exc
    if type(pid) is int and type(ticks) is int and _process_start_ticks(pid) == ticks:
        return pid, ticks
    return None


def _single_argument(arguments: list[str], name: str) -> str:
    values = [
        arguments[index + 1]
        for index, value in enumerate(arguments[:-1])
        if value == name
    ]
    if len(values) != 1 or not values[0]:
        raise StarterError(f"monitor requires exactly one {name}")
    if not Path(values[0]).is_absolute():
        raise StarterError(f"monitor {name} must be absolute")
    return values[0]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ready_marker(
    path: Path,
    *,
    child_pid: int,
    campaign_root: str,
    output_root: str,
    driver_status: str,
    learner_backend: str,
) -> dict[str, object] | None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise StarterError("monitor readiness marker is unsafe")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StarterError("monitor readiness marker is invalid") from exc
    expected = {
        "schema": MONITOR_READY_SCHEMA,
        "process_id": child_pid,
        "campaign_root": campaign_root,
        "output_root": output_root,
        "driver_status": driver_status,
        "learner_backend": learner_backend,
    }
    if not isinstance(document, dict) or any(
        document.get(key) != value for key, value in expected.items()
    ):
        raise StarterError("monitor readiness marker does not match child")
    report_path = document.get("report_path")
    report_sha = document.get("report_sha256")
    ready_ns = document.get("ready_unix_ns")
    if (
        report_path != str(Path(output_root) / "latest.json")
        or not isinstance(report_sha, str)
        or len(report_sha) != 64
        or type(ready_ns) is not int
    ):
        raise StarterError("monitor readiness marker binding is malformed")
    report_file = Path(report_path)
    try:
        if _sha256_file(report_file) != report_sha:
            raise StarterError("monitor readiness report changed")
        report = json.loads(report_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StarterError("monitor readiness report is invalid") from exc
    progress = report.get("progress_health") if isinstance(report, dict) else None
    learner = report.get("learner_control_plane") if isinstance(report, dict) else None
    wandb = report.get("wandb_remote") if isinstance(report, dict) else None
    if not (
        isinstance(progress, dict)
        and progress.get("production_observability_healthy") is True
        and progress.get("required_processes_healthy") is True
        and isinstance(learner, dict)
        and learner.get("backend") == learner_backend
        and learner.get("healthy") is True
        and isinstance(wandb, dict)
        and wandb.get("reachable") is True
        and wandb.get("authorized") is True
    ):
        raise StarterError("monitor first report is not production healthy")
    return document


def _terminate_child(child: subprocess.Popen[bytes]) -> None:
    """Reap the exact monitor child created by a failed startup transaction."""

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


def start_monitor(
    *,
    python: Path,
    pidfile: Path,
    logfile: Path,
    monitor_arguments: list[str],
    ready_timeout_seconds: float = 180.0,
) -> dict[str, object]:
    if _existing_monitor(pidfile) is not None:
        raise StarterError("a monitor recorded by this pidfile is already running")
    quality_starter._executable_entrypoint(python, "monitor Python")
    if not math.isfinite(ready_timeout_seconds) or ready_timeout_seconds <= 0:
        raise StarterError("monitor readiness timeout must be positive")
    forbidden = {"--once", "--test-max-polls", "--test-poll-interval-seconds"}
    if forbidden.intersection(monitor_arguments):
        raise StarterError("durable monitor cannot use bounded/test polling flags")
    if any(_SENSITIVE_RE.search(argument) for argument in monitor_arguments):
        raise StarterError("monitor argv appears to contain a credential")
    campaign_root = _single_argument(monitor_arguments, "--campaign-root")
    output_root = _single_argument(monitor_arguments, "--output-root")
    driver_status = _single_argument(monitor_arguments, "--expected-driver-status")
    learner_backend_values = [
        monitor_arguments[index + 1]
        for index, value in enumerate(monitor_arguments[:-1])
        if value == "--learner-backend"
    ]
    if len(learner_backend_values) != 1 or learner_backend_values[0] not in {
        "oci_nrt",
        "lepton_job_v1",
    }:
        raise StarterError("monitor requires exactly one supported learner backend")
    learner_backend = learner_backend_values[0]
    ready_marker = Path(output_root) / "MONITOR_READY.json"
    if ready_marker.exists() or ready_marker.is_symlink():
        if not stat.S_ISREG(os.lstat(ready_marker).st_mode):
            raise StarterError("monitor readiness marker is unsafe")
        ready_marker.unlink()
    if "--ready-marker" in monitor_arguments:
        raise StarterError("monitor readiness marker is owned by the starter")
    command_arguments = [
        *monitor_arguments,
        "--ready-marker",
        str(ready_marker),
    ]
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
            [str(python), str(MONITOR), *command_arguments],
            stdin=subprocess.DEVNULL,
            stdout=log_descriptor,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(log_descriptor)
    try:
        deadline = time.monotonic() + ready_timeout_seconds
        ready_document: dict[str, object] | None = None
        while time.monotonic() < deadline:
            return_code = child.poll()
            if return_code is not None:
                raise StarterError(
                    f"monitor exited during startup with status {return_code}"
                )
            ready_document = _read_ready_marker(
                ready_marker,
                child_pid=child.pid,
                campaign_root=campaign_root,
                output_root=output_root,
                driver_status=driver_status,
                learner_backend=learner_backend,
            )
            if ready_document is not None:
                break
            time.sleep(0.1)
        if ready_document is None:
            raise StarterError("monitor first healthy report timed out")
        start_ticks = _process_start_ticks(child.pid)
        if start_ticks is None:
            raise StarterError("cannot bind detached monitor process identity")
        _atomic_pidfile(
            pidfile,
            {
                "schema": MONITOR_PID_SCHEMA,
                "pid": child.pid,
                "start_ticks": start_ticks,
                "entrypoint": str(MONITOR),
                "campaign_root": campaign_root,
                "output_root": output_root,
                "driver_status": driver_status,
                "learner_backend": learner_backend,
                "ready_marker": str(ready_marker),
            },
        )
        return {
            "status": "STARTED",
            "pid": child.pid,
            "start_ticks": start_ticks,
            "pidfile": str(pidfile),
            "logfile": str(logfile),
            "monitor": str(MONITOR),
            "campaign_root": campaign_root,
            "output_root": output_root,
            "driver_status": driver_status,
            "learner_backend": learner_backend,
            "ready_marker": str(ready_marker),
            "ready": True,
            "interval_seconds": 1800,
            "read_only": True,
        }
    except BaseException:
        _terminate_child(child)
        raise


def _stop_bound_process(
    report: dict[str, object], *, graceful_timeout_seconds: float = 15.0
) -> None:
    """Gracefully drain only the exact child created by this transaction."""

    pid = report.get("pid")
    ticks = report.get("start_ticks")
    if type(pid) is int and type(ticks) is int and _process_start_ticks(pid) == ticks:
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + graceful_timeout_seconds
        while time.monotonic() < deadline:
            if _process_start_ticks(pid) != ticks:
                return
            time.sleep(0.1)
        if _process_start_ticks(pid) == ticks:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if _process_start_ticks(pid) != ticks:
                    return
                time.sleep(0.1)


def start_production_observers(
    *,
    python: Path,
    pidfile: Path,
    logfile: Path,
    monitor_arguments: list[str],
    quality_campaign_root: Path,
    quality_state_dir: Path,
    quality_driver_status: Path,
    quality_project: str,
    quality_entity: str,
    quality_expected_email: str,
    quality_poll_seconds: float,
    quality_pidfile: Path,
    quality_logfile: Path,
    quality_ready_timeout_seconds: float,
    evaluation_campaign_root: Path,
    evaluation_state_dir: Path,
    evaluation_driver_status: Path,
    evaluation_project: str,
    evaluation_entity: str,
    evaluation_expected_email: str,
    evaluation_poll_seconds: float,
    evaluation_pidfile: Path,
    evaluation_logfile: Path,
    evaluation_ready_timeout_seconds: float,
    monitor_ready_timeout_seconds: float = 180.0,
) -> dict[str, object]:
    """Start both required W&B bridges, await readiness, then their monitor."""

    if evaluation_campaign_root != quality_campaign_root:
        raise StarterError("quality and evaluation campaign roots must match")
    if evaluation_driver_status != quality_driver_status:
        raise StarterError("quality and evaluation driver status paths must match")

    quality = quality_starter.start_bridge(
        python=python,
        campaign_root=quality_campaign_root,
        state_dir=quality_state_dir,
        driver_status=quality_driver_status,
        project=quality_project,
        entity=quality_entity,
        expected_email=quality_expected_email,
        poll_seconds=quality_poll_seconds,
        pidfile=quality_pidfile,
        logfile=quality_logfile,
        ready_timeout_seconds=quality_ready_timeout_seconds,
    )
    try:
        evaluation = evaluation_starter.start_bridge(
            python=python,
            campaign_root=evaluation_campaign_root,
            state_dir=evaluation_state_dir,
            driver_status=evaluation_driver_status,
            project=evaluation_project,
            entity=evaluation_entity,
            expected_email=evaluation_expected_email,
            poll_seconds=evaluation_poll_seconds,
            pidfile=evaluation_pidfile,
            logfile=evaluation_logfile,
            ready_timeout_seconds=evaluation_ready_timeout_seconds,
        )
    except BaseException:
        _stop_bound_process(quality)
        raise
    monitored_arguments = list(monitor_arguments)
    expected_driver_pair = (
        "--expected-driver-status",
        str(quality_driver_status),
    )
    existing_driver_values = [
        monitored_arguments[index + 1]
        for index, value in enumerate(monitored_arguments[:-1])
        if value == "--expected-driver-status"
    ]
    if existing_driver_values and existing_driver_values != [str(quality_driver_status)]:
        _stop_bound_process(evaluation)
        _stop_bound_process(quality)
        raise StarterError("monitor driver status differs from both bridges")
    if not existing_driver_values:
        monitored_arguments.extend(expected_driver_pair)
    for required_pidfile in (quality_pidfile, evaluation_pidfile):
        pair = ("--required-process-pidfile", str(required_pidfile))
        if not any(
            tuple(monitored_arguments[index : index + 2]) == pair
            for index in range(len(monitored_arguments) - 1)
        ):
            monitored_arguments.extend(pair)
    try:
        monitor = start_monitor(
            python=python,
            pidfile=pidfile,
            logfile=logfile,
            monitor_arguments=monitored_arguments,
            ready_timeout_seconds=monitor_ready_timeout_seconds,
        )
    except BaseException:
        _stop_bound_process(evaluation)
        _stop_bound_process(quality)
        raise
    return {
        "status": "STARTED",
        "read_only": True,
        "monitor": monitor,
        "student_quality_bridge": quality,
        "evaluation_bridge": evaluation,
        "quality_bridge_ready": True,
        "quality_bridge_monitored": True,
        "evaluation_bridge_ready": True,
        "evaluation_bridge_monitored": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--pidfile", type=Path, required=True)
    parser.add_argument("--logfile", type=Path, required=True)
    parser.add_argument("--quality-campaign-root", type=Path, required=True)
    parser.add_argument("--quality-state-dir", type=Path, required=True)
    parser.add_argument("--quality-driver-status", type=Path, required=True)
    parser.add_argument("--quality-project", required=True)
    parser.add_argument("--quality-entity", required=True)
    parser.add_argument("--quality-expected-email", required=True)
    parser.add_argument("--quality-poll-seconds", type=float, default=60.0)
    parser.add_argument("--quality-pidfile", type=Path, required=True)
    parser.add_argument("--quality-logfile", type=Path, required=True)
    parser.add_argument("--quality-ready-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--evaluation-campaign-root", type=Path, required=True)
    parser.add_argument("--evaluation-state-dir", type=Path, required=True)
    parser.add_argument("--evaluation-driver-status", type=Path, required=True)
    parser.add_argument("--evaluation-project", required=True)
    parser.add_argument("--evaluation-entity", required=True)
    parser.add_argument("--evaluation-expected-email", required=True)
    parser.add_argument("--evaluation-poll-seconds", type=float, default=60.0)
    parser.add_argument("--evaluation-pidfile", type=Path, required=True)
    parser.add_argument("--evaluation-logfile", type=Path, required=True)
    parser.add_argument(
        "--evaluation-ready-timeout-seconds", type=float, default=45.0
    )
    parser.add_argument("--monitor-ready-timeout-seconds", type=float, default=180.0)
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
        report = start_production_observers(
            python=_absolute(arguments.python, "monitor Python"),
            pidfile=_absolute(arguments.pidfile, "pidfile"),
            logfile=_absolute(arguments.logfile, "logfile"),
            monitor_arguments=monitor_arguments,
            quality_campaign_root=_absolute(
                arguments.quality_campaign_root, "quality campaign root"
            ),
            quality_state_dir=_absolute(arguments.quality_state_dir, "quality state"),
            quality_driver_status=_absolute(
                arguments.quality_driver_status, "quality driver status"
            ),
            quality_project=arguments.quality_project,
            quality_entity=arguments.quality_entity,
            quality_expected_email=arguments.quality_expected_email,
            quality_poll_seconds=arguments.quality_poll_seconds,
            quality_pidfile=_absolute(arguments.quality_pidfile, "quality pidfile"),
            quality_logfile=_absolute(arguments.quality_logfile, "quality logfile"),
            quality_ready_timeout_seconds=arguments.quality_ready_timeout_seconds,
            evaluation_campaign_root=_absolute(
                arguments.evaluation_campaign_root, "evaluation campaign root"
            ),
            evaluation_state_dir=_absolute(
                arguments.evaluation_state_dir, "evaluation state"
            ),
            evaluation_driver_status=_absolute(
                arguments.evaluation_driver_status, "evaluation driver status"
            ),
            evaluation_project=arguments.evaluation_project,
            evaluation_entity=arguments.evaluation_entity,
            evaluation_expected_email=arguments.evaluation_expected_email,
            evaluation_poll_seconds=arguments.evaluation_poll_seconds,
            evaluation_pidfile=_absolute(
                arguments.evaluation_pidfile, "evaluation pidfile"
            ),
            evaluation_logfile=_absolute(
                arguments.evaluation_logfile, "evaluation logfile"
            ),
            evaluation_ready_timeout_seconds=(
                arguments.evaluation_ready_timeout_seconds
            ),
            monitor_ready_timeout_seconds=arguments.monitor_ready_timeout_seconds,
        )
    except StarterError as exc:
        print(f"monitor starter failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
