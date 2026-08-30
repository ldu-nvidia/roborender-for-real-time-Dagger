"""Read-only 30-minute health monitor for the production DAgger campaign.

Campaign files, processes, GPUs, and the remote OCI-NRT Slurm control plane are
observed only.  Reports are written to a separate operator-owned output root;
the monitor never invokes a campaign command, signals a process, or submits,
cancels, holds, or releases a job.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from round_dagger import artifacts, state


MONITOR_REPORT_SCHEMA = "fast_roborender.round_dagger.production_health_report.v1"
MONITOR_INTERVAL_SECONDS = 1800.0
GPU_BURST_SAMPLES = 12
GPU_BURST_INTERVAL_SECONDS = 5.0
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}")


class ProductionMonitorError(RuntimeError):
    """A safe read-only observation could not be performed."""


def _absolute(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    if not path.is_absolute():
        raise ProductionMonitorError(f"{label} must be absolute")
    return path


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ProductionMonitorError(f"{label} must be a lowercase SHA-256")
    return value


def _regular_immutable(path: Path, expected_sha: str, label: str) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise ProductionMonitorError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
        raise ProductionMonitorError(f"{label} must be immutable and non-symlink: {path}")
    observed = artifacts.sha256_file(path)
    if observed != _sha(expected_sha, f"{label} SHA-256"):
        raise ProductionMonitorError(f"{label} SHA-256 mismatch")


def _outside_campaign(output_root: Path, campaign_root: Path) -> None:
    try:
        state._check_path_chain(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        state._check_path_chain(output_root)
    except Exception as exc:
        raise ProductionMonitorError(f"unsafe monitor output root: {exc}") from exc
    output_root = output_root.resolve(strict=True)
    campaign_root = campaign_root.resolve(strict=True)
    try:
        output_root.relative_to(campaign_root)
    except ValueError:
        return
    raise ProductionMonitorError(
        "monitor output root must be outside the campaign; monitoring cannot mutate run state"
    )


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
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
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


class MonitorLock:
    def __init__(self, output_root: Path):
        self.path = output_root / ".monitor.lock"
        self._descriptor: int | None = None

    def __enter__(self) -> "MonitorLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            info = os.lstat(self.path)
        except FileNotFoundError:
            info = None
        if info is not None and not stat.S_ISREG(info.st_mode):
            raise ProductionMonitorError("unsafe monitor lock path")
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise ProductionMonitorError("another monitor owns this output root") from exc
        self._descriptor = descriptor
        return self

    def __exit__(self, *_: object) -> None:
        if self._descriptor is not None:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            os.close(self._descriptor)
            self._descriptor = None


def _coordinator(campaign_root: Path) -> dict[str, Any]:
    try:
        campaign = state.load_campaign(campaign_root)
        coordinator = state.load_state(campaign_root)
    except Exception as exc:
        raise ProductionMonitorError(f"durable campaign state is invalid: {exc}") from exc
    count = campaign.get("macro_round_count")
    if count != 5:
        raise ProductionMonitorError("campaign is not the bounded five-round production run")
    return {
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": artifacts.canonical_json_sha256(campaign),
        "phase": coordinator["phase"],
        "next_macro_round": coordinator["next_macro_round"],
        "open_macro_round": coordinator["open_macro_round"],
        "macro_round_count": count,
        "completed": coordinator["phase"] == "READY" and coordinator["next_macro_round"] == count,
        "revision": coordinator["revision"],
        "active_policy_version": coordinator["active_policy"]["version"],
        "active_policy_checkpoint_sha256": coordinator["active_policy"]["checkpoint_sha256"],
    }


def _process_matches(patterns: Sequence[str]) -> dict[str, Any]:
    counts = {pattern: 0 for pattern in patterns}
    pids: dict[str, list[int]] = {pattern: [] for pattern in patterns}
    # This monitor's argv contains every configured pattern.  Its detached
    # starter may still be its ancestor during the first poll.  Neither is a
    # campaign workload, so exclude the complete monitor ancestry.
    excluded_pids: set[int] = set()
    ancestor = os.getpid()
    while ancestor > 1 and ancestor not in excluded_pids:
        excluded_pids.add(ancestor)
        try:
            status_lines = Path(f"/proc/{ancestor}/status").read_text(
                encoding="utf-8"
            ).splitlines()
            parent_line = next(
                line for line in status_lines if line.startswith("PPid:")
            )
            ancestor = int(parent_line.split()[1])
        except (FileNotFoundError, PermissionError, StopIteration, OSError, ValueError):
            break
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        if int(entry.name) in excluded_pids:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace")
        for pattern in patterns:
            if pattern in command:
                counts[pattern] += 1
                pids[pattern].append(int(entry.name))
    # Deliberately omit process argv: it is unnecessary for health and could
    # disclose an unrelated process credential.
    return {
        pattern: {"count": counts[pattern], "pids": sorted(pids[pattern])}
        for pattern in patterns
    }


def _process_start_ticks(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        if fields[2] == "Z":
            return None
        return int(fields[21])
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def _process_argv(pid: int) -> tuple[str, ...] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    if not raw:
        return None
    return tuple(
        item.decode("utf-8", errors="replace")
        for item in raw.split(b"\0")
        if item
    )


def _has_argument_pair(argv: Sequence[str], name: str, value: str) -> bool:
    return any(
        argv[index] == name and argv[index + 1] == value
        for index in range(len(argv) - 1)
    )


def _bound_process_matches(
    pidfiles: Sequence[Path], *, campaign_root: Path
) -> dict[str, dict[str, Any]]:
    """Check exact campaign process identities, never a cross-campaign substring."""

    reports: dict[str, dict[str, Any]] = {}
    expected_keys = {
        "schema",
        "pid",
        "start_ticks",
        "entrypoint",
        "campaign_root",
        "state_dir",
        "driver_status",
    }
    expected_schemas = {
        "fast_roborender.round_dagger.student_quality_bridge_process.v1",
        "fast_roborender.round_dagger.evaluation_bridge_process.v1",
    }
    for pidfile in pidfiles:
        key = str(pidfile)
        report: dict[str, Any] = {"healthy": False, "pid": None}
        reports[key] = report
        try:
            info = os.lstat(pidfile)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("unsafe_pidfile")
            document = artifacts.load_strict_json(pidfile)
            if not isinstance(document, Mapping) or set(document) != expected_keys:
                raise ValueError("invalid_pidfile_contract")
            if document.get("schema") not in expected_schemas:
                raise ValueError("invalid_pidfile_schema")
            pid = document.get("pid")
            start_ticks = document.get("start_ticks")
            entrypoint = document.get("entrypoint")
            bound_campaign = document.get("campaign_root")
            state_dir = document.get("state_dir")
            driver_status = document.get("driver_status")
            if type(pid) is not int or type(start_ticks) is not int:
                raise ValueError("invalid_process_identity")
            if not all(
                isinstance(value, str) and Path(value).is_absolute()
                for value in (entrypoint, bound_campaign, state_dir, driver_status)
            ):
                raise ValueError("invalid_process_binding")
            report["pid"] = pid
            if bound_campaign != str(campaign_root):
                raise ValueError("campaign_mismatch")
            if _process_start_ticks(pid) != start_ticks:
                raise ValueError("process_identity_not_live")
            argv = _process_argv(pid)
            if argv is None or entrypoint not in argv:
                raise ValueError("entrypoint_mismatch")
            for name, value in (
                ("--campaign-root", bound_campaign),
                ("--state-dir", state_dir),
                ("--driver-status", driver_status),
            ):
                if not _has_argument_pair(argv, name, value):
                    raise ValueError("argument_binding_mismatch")
            report.update(
                {
                    "healthy": True,
                    "start_ticks": start_ticks,
                    "campaign_bound": True,
                }
            )
        except FileNotFoundError:
            report["error"] = "pidfile_missing"
        except artifacts.ArtifactError:
            report["error"] = "invalid_pidfile_json"
        except OSError:
            report["error"] = "pidfile_io_error"
        except (TypeError, ValueError) as exc:
            # Only stable reason codes are exposed; process argv is never copied
            # into telemetry because unrelated arguments could contain secrets.
            report["error"] = str(exc) or type(exc).__name__
    return reports


def _tree_snapshot(root: Path, *, maximum_files: int = 200_000) -> dict[str, Any]:
    result: dict[str, Any] = {
        "root": str(root),
        "exists": False,
        "file_count": 0,
        "total_bytes": 0,
        "latest_mtime_ns": None,
        "newest_relative_path": None,
        "truncated": False,
    }
    try:
        info = os.lstat(root)
    except FileNotFoundError:
        return result
    if not stat.S_ISDIR(info.st_mode):
        result["error"] = "not_a_non_symlink_directory"
        return result
    result["exists"] = True
    newest: tuple[int, str] | None = None
    for directory, names, filenames in os.walk(root, followlinks=False):
        names[:] = [
            name
            for name in names
            if not os.path.islink(os.path.join(directory, name))
        ]
        for filename in filenames:
            path = Path(directory) / filename
            try:
                child = os.lstat(path)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(child.st_mode):
                continue
            result["file_count"] += 1
            result["total_bytes"] += child.st_size
            relative = path.relative_to(root).as_posix()
            item = (child.st_mtime_ns, relative)
            if newest is None or item > newest:
                newest = item
            if result["file_count"] >= maximum_files:
                result["truncated"] = True
                break
        if result["truncated"]:
            break
    if newest is not None:
        result["latest_mtime_ns"], result["newest_relative_path"] = newest
    return result


def _movement(current: Mapping[str, Any], previous: Mapping[str, Any] | None) -> dict[str, Any]:
    if previous is None:
        return {"baseline": True, "file_count_delta": None, "byte_delta": None, "latest_advanced": None}
    return {
        "baseline": False,
        "file_count_delta": current["file_count"] - previous.get("file_count", 0),
        "byte_delta": current["total_bytes"] - previous.get("total_bytes", 0),
        "latest_advanced": (
            current.get("latest_mtime_ns") is not None
            and current.get("latest_mtime_ns") > (previous.get("latest_mtime_ns") or 0)
        ),
    }


def _movement_progress(entries: Sequence[Mapping[str, Any]]) -> bool | None:
    """Return True/False after a baseline exists, otherwise None."""

    observed = False
    progressed = False
    for entry in entries:
        movement = entry.get("movement")
        if not isinstance(movement, Mapping) or movement.get("baseline") is True:
            continue
        observed = True
        progressed = progressed or bool(
            movement.get("latest_advanced")
            or (movement.get("file_count_delta") or 0) > 0
            or (movement.get("byte_delta") or 0) > 0
        )
    return progressed if observed else None


def _stale_poll_count(
    *,
    previous: Mapping[str, Any] | None,
    key: str,
    expected_now: bool,
    progressed: bool | None,
    coordinator: Mapping[str, Any],
) -> int:
    if not expected_now or progressed is None or progressed:
        return 0
    prior_campaign = previous.get("campaign", {}) if isinstance(previous, Mapping) else {}
    same_frontier = (
        prior_campaign.get("phase") == coordinator.get("phase")
        and prior_campaign.get("next_macro_round") == coordinator.get("next_macro_round")
        and prior_campaign.get("revision") == coordinator.get("revision")
    )
    if not same_frontier:
        return 1
    prior_health = previous.get("progress_health", {})
    prior_count = prior_health.get(key, 0) if isinstance(prior_health, Mapping) else 0
    return (prior_count if type(prior_count) is int and prior_count >= 0 else 0) + 1


def _gpu_health() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {
            "available": False,
            "error": "nvidia-smi_not_found",
            "gpus": [],
            "compute_processes": [],
        }
    command = [
        executable,
        "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "error": type(exc).__name__,
            "gpus": [],
            "compute_processes": [],
        }
    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            continue
        try:
            gpus.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "uuid": fields[2],
                    "memory_total_mib": int(fields[3]),
                    "memory_used_mib": int(fields[4]),
                    "memory_free_mib": int(fields[3]) - int(fields[4]),
                    "memory_headroom_ok": int(fields[3]) - int(fields[4]) >= 4096,
                    "utilization_percent": int(fields[5]),
                    "temperature_c": int(fields[6]),
                    "temperature_ok": int(fields[6]) < 85,
                }
            )
        except ValueError:
            continue
    process_command = [
        executable,
        "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    processes: list[dict[str, Any]] = []
    process_error: str | None = None
    try:
        process_result = subprocess.run(
            process_command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        for line in process_result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",", maxsplit=3)]
            if len(fields) != 4:
                continue
            try:
                processes.append(
                    {
                        "pid": int(fields[0]),
                        "gpu_uuid": fields[1],
                        # nvidia-smi's process basename is safe to expose; the
                        # full argv is intentionally never collected.
                        "process_name": fields[2],
                        "used_memory_mib": int(fields[3]),
                    }
                )
            except ValueError:
                continue
    except (OSError, subprocess.SubprocessError) as exc:
        process_error = type(exc).__name__
    return {
        "available": True,
        "error": None,
        "gpus": gpus,
        "compute_processes": processes,
        "compute_process_query_error": process_error,
    }


def _gpu_health_burst() -> dict[str, Any]:
    """Sample one read-only 60-second GPU window for each 30-minute report."""

    samples: list[dict[str, Any]] = []
    for _ in range(GPU_BURST_SAMPLES):
        samples.append(_gpu_health())
        time.sleep(GPU_BURST_INTERVAL_SECONDS)
    by_uuid: dict[str, dict[str, Any]] = {}
    processes: dict[tuple[int, str, str], dict[str, Any]] = {}
    for sample_index, sample in enumerate(samples):
        for device in sample.get("gpus", []):
            item = by_uuid.setdefault(
                device["uuid"],
                {
                    **device,
                    "utilization_samples_percent": [],
                    "temperature_samples_c": [],
                    "used_memory_samples_mib": [],
                    "free_memory_samples_mib": [],
                },
            )
            item["utilization_samples_percent"].append(device["utilization_percent"])
            item["temperature_samples_c"].append(device["temperature_c"])
            item["used_memory_samples_mib"].append(device["memory_used_mib"])
            item["free_memory_samples_mib"].append(device["memory_free_mib"])
        for process in sample.get("compute_processes", []):
            key = (process["pid"], process["gpu_uuid"], process["process_name"])
            item = processes.setdefault(
                key,
                {
                    **process,
                    "sample_indices": [],
                    "maximum_used_memory_mib": 0,
                    "classification": (
                        "known_foreign_tesseract"
                        if "tesseract" in process["process_name"].lower()
                        else "unattributed_compute_process"
                    ),
                },
            )
            item["sample_indices"].append(sample_index)
            item["maximum_used_memory_mib"] = max(
                item["maximum_used_memory_mib"], process["used_memory_mib"]
            )
    devices: list[dict[str, Any]] = []
    severity = "healthy"
    alerts: list[str] = []
    rank = {"healthy": 0, "warning": 1, "critical": 2, "emergency": 3}

    def elevate(level: str, message: str) -> None:
        nonlocal severity
        if rank[level] > rank[severity]:
            severity = level
        alerts.append(f"{level}: {message}")

    uuid_to_index: dict[str, int] = {}
    for item in by_uuid.values():
        uuid_to_index[item["uuid"]] = item["index"]
        item["utilization_mean_percent"] = round(
            sum(item["utilization_samples_percent"])
            / len(item["utilization_samples_percent"]),
            2,
        )
        item["utilization_max_percent"] = max(item["utilization_samples_percent"])
        item["temperature_max_c"] = max(item["temperature_samples_c"])
        item["memory_used_max_mib"] = max(item["used_memory_samples_mib"])
        item["memory_free_min_mib"] = min(item["free_memory_samples_mib"])
        if item["memory_free_min_mib"] <= 15_000:
            elevate("emergency", f"GPU {item['index']} free VRAM <=15000 MiB")
        elif item["memory_used_max_mib"] >= 70_000 or item["memory_free_min_mib"] <= 25_000:
            elevate("critical", f"GPU {item['index']} VRAM crossed critical threshold")
        elif item["memory_used_max_mib"] >= 55_000 or item["memory_free_min_mib"] <= 40_000:
            elevate("warning", f"GPU {item['index']} VRAM crossed warning threshold")
        if item["temperature_max_c"] >= 87:
            elevate("critical", f"GPU {item['index']} temperature >=87 C")
        elif item["temperature_max_c"] >= 83:
            elevate("warning", f"GPU {item['index']} temperature >=83 C")
        devices.append(item)
    process_inventory = sorted(processes.values(), key=lambda item: (item["gpu_uuid"], item["pid"]))
    for process in process_inventory:
        gpu_index = uuid_to_index.get(process["gpu_uuid"])
        process["gpu_index"] = gpu_index
        if process["classification"] == "known_foreign_tesseract":
            if process["maximum_used_memory_mib"] > 10_240:
                elevate("critical", "known tesseract process exceeds 10240 MiB")
            elif gpu_index != 0 or process["maximum_used_memory_mib"] > 6_144:
                elevate("warning", "known tesseract process moved/restarted or exceeds 6144 MiB")
        else:
            elevate(
                "warning",
                f"unattributed GPU compute process pid={process['pid']} gpu={gpu_index}",
            )
    return {
        "available": any(sample.get("available") for sample in samples),
        "sample_count": len(samples),
        "sample_interval_seconds": GPU_BURST_INTERVAL_SECONDS,
        "window_seconds": GPU_BURST_INTERVAL_SECONDS * len(samples),
        "severity": severity,
        "alerts": alerts,
        "gpus": sorted(devices, key=lambda item: item["index"]),
        "compute_processes": process_inventory,
        "read_only": True,
    }


def _disk_health(paths: Sequence[Path]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        try:
            usage = shutil.disk_usage(path)
            reports.append(
                {
                    "path": str(path),
                    "total_bytes": usage.total,
                    "used_bytes": usage.used,
                    "free_bytes": usage.free,
                    "free_percent": round(100.0 * usage.free / usage.total, 3),
                    "error": None,
                }
            )
        except OSError as exc:
            reports.append({"path": str(path), "error": type(exc).__name__})
    return reports


def _remote_command(host: str, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    remote = shlex.join(list(arguments))
    return subprocess.run(
        [
            "ssh",
            "-oBatchMode=yes",
            "-oConnectTimeout=10",
            "-oServerAliveInterval=10",
            "-oServerAliveCountMax=1",
            host,
            remote,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _parse_slurm(output: str, prefix: str, *, fields: Sequence[str]) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for line in output.splitlines():
        values = line.rstrip().split("|")
        if len(values) < len(fields):
            continue
        item = dict(zip(fields, values, strict=False))
        if item.get("job_name", "").startswith(prefix):
            jobs.append(item)
    return jobs


def _slurm_health(host: str, user: str, job_prefix: str) -> dict[str, Any]:
    if _TOKEN_RE.fullmatch(host) is None or _TOKEN_RE.fullmatch(user) is None:
        raise ProductionMonitorError("unsafe OCI-NRT SSH host or Slurm user")
    if not job_prefix or len(job_prefix) > 128 or "|" in job_prefix or "\n" in job_prefix:
        raise ProductionMonitorError("unsafe Slurm job-name prefix")
    report: dict[str, Any] = {
        "ssh_host": host,
        "slurm_user": user,
        "job_name_prefix": job_prefix,
        "reachable": False,
        "active_jobs": [],
        "recent_jobs": [],
        "error": None,
    }
    try:
        active = _remote_command(
            host,
            ["squeue", "--noheader", "--user", user, "--format", "%i|%T|%M|%R|%j"],
        )
        recent = _remote_command(
            host,
            [
                "sacct",
                "-X",
                "--noheader",
                "--parsable2",
                "--starttime",
                "now-2days",
                "--user",
                user,
                "--format",
                "JobIDRaw,State,Elapsed,ExitCode,JobName",
            ],
        )
        report["reachable"] = True
        report["active_jobs"] = _parse_slurm(
            active.stdout,
            job_prefix,
            fields=("job_id", "state", "elapsed", "reason_or_node", "job_name"),
        )
        report["recent_jobs"] = _parse_slurm(
            recent.stdout,
            job_prefix,
            fields=("job_id", "state", "elapsed", "exit_code", "job_name"),
        )[-100:]
    except (OSError, subprocess.SubprocessError) as exc:
        report["error"] = type(exc).__name__
    return report


def _viewer_value(viewer: object, key: str) -> object:
    if isinstance(viewer, Mapping):
        return viewer.get(key)
    return getattr(viewer, key, None)


def _wandb_remote_health(
    *,
    entity: str,
    project: str,
    expected_email: str,
    netrc_path: Path,
    campaign_id: str,
    campaign_sha256: str | None = None,
    campaign_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Read recent immutable-campaign-bound W&B runs without mutating them.

    ``campaign_id`` remains supported by itself for non-production callers.
    Production callers provide both immutable digests.  A remote run must then
    carry at least one matching digest, and every digest it carries must match.
    This keeps legacy actor telemetry (contract-bound) and the newer quality
    and evaluation runs (campaign-bound) in one view without mixing a stale
    campaign incarnation that reused the same human-readable campaign ID.
    """

    for value, label in (
        (entity, "W&B entity"),
        (project, "W&B project"),
        (expected_email, "W&B expected email"),
    ):
        if _TOKEN_RE.fullmatch(value) is None:
            raise ProductionMonitorError(f"unsafe {label}")
    expected_campaign_sha = (
        _sha(campaign_sha256, "W&B campaign SHA-256")
        if campaign_sha256 is not None
        else None
    )
    expected_contract_sha = (
        _sha(campaign_contract_sha256, "W&B campaign contract SHA-256")
        if campaign_contract_sha256 is not None
        else None
    )
    try:
        info = os.lstat(netrc_path)
    except FileNotFoundError as exc:
        raise ProductionMonitorError(f"W&B netrc is missing: {netrc_path}") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        raise ProductionMonitorError("W&B netrc must be a private non-symlink regular file")
    report: dict[str, Any] = {
        "reachable": False,
        "authorized": False,
        "entity": entity,
        "project": project,
        "expected_email": expected_email,
        "viewer_email": None,
        "campaign_sha256": expected_campaign_sha,
        "campaign_contract_sha256": expected_contract_sha,
        "campaign_run_count": 0,
        "immutable_binding_counts": {},
        "ignored_run_counts": {
            "other_campaign_id": 0,
            "immutable_mismatch": 0,
            "missing_immutable_binding": 0,
        },
        "runs": [],
        "error": None,
        "read_only": True,
    }
    protected_names = (
        "WANDB_API_KEY",
        "WANDB_ACCESS_TOKEN",
        "WANDB_IDENTITY_TOKEN",
        "WANDB_OIDC_TOKEN",
    )
    saved = {name: os.environ.get(name) for name in (*protected_names, "NETRC")}
    try:
        for name in protected_names:
            os.environ.pop(name, None)
        os.environ["NETRC"] = str(netrc_path)
        import wandb

        api = wandb.Api(timeout=30)
        viewer = api.viewer
        viewer_email = _viewer_value(viewer, "email")
        report["viewer_email"] = viewer_email
        report["authorized"] = viewer_email == expected_email
        remote_filters: dict[str, Any] = {
            "config.round_dagger/campaign_id": campaign_id
        }
        immutable_filters = []
        if expected_campaign_sha is not None:
            immutable_filters.append(
                {
                    "config.round_dagger/campaign_sha256": expected_campaign_sha
                }
            )
        if expected_contract_sha is not None:
            immutable_filters.append(
                {
                    "config.round_dagger/campaign_contract_sha256": (
                        expected_contract_sha
                    )
                }
            )
        if immutable_filters:
            remote_filters = {
                "$and": [
                    remote_filters,
                    {"$or": immutable_filters},
                ]
            }
        remote_runs = api.runs(
            f"{entity}/{project}",
            order="-updated_at",
            per_page=100,
            filters=remote_filters,
        )
        runs: list[dict[str, Any]] = []
        binding_counts: dict[str, int] = {}
        for remote in remote_runs:
            config = getattr(remote, "config", {})
            if (
                not isinstance(config, Mapping)
                or config.get("round_dagger/campaign_id") != campaign_id
            ):
                report["ignored_run_counts"]["other_campaign_id"] += 1
                continue
            observed_bindings: list[str] = []
            immutable_mismatch = False
            for key, expected, label in (
                (
                    "round_dagger/campaign_sha256",
                    expected_campaign_sha,
                    "campaign_sha256",
                ),
                (
                    "round_dagger/campaign_contract_sha256",
                    expected_contract_sha,
                    "campaign_contract_sha256",
                ),
            ):
                observed = config.get(key)
                if expected is None or observed is None:
                    continue
                if observed != expected:
                    immutable_mismatch = True
                else:
                    observed_bindings.append(label)
            if immutable_mismatch:
                report["ignored_run_counts"]["immutable_mismatch"] += 1
                continue
            if immutable_filters and not observed_bindings:
                report["ignored_run_counts"]["missing_immutable_binding"] += 1
                continue
            immutable_binding = (
                "+".join(observed_bindings)
                if observed_bindings
                else "campaign_id_only"
            )
            binding_counts[immutable_binding] = (
                binding_counts.get(immutable_binding, 0) + 1
            )
            runs.append(
                {
                    "id": str(getattr(remote, "id", "")),
                    "name": str(getattr(remote, "name", "")),
                    "state": str(getattr(remote, "state", "")),
                    "updated_at": str(getattr(remote, "updated_at", "")),
                    "last_history_step": int(
                        getattr(remote, "lastHistoryStep", 0) or 0
                    ),
                    "immutable_binding": immutable_binding,
                }
            )
            if len(runs) >= 50:
                break
        report["reachable"] = True
        report["campaign_run_count"] = len(runs)
        report["immutable_binding_counts"] = binding_counts
        report["runs"] = runs
    except Exception as exc:
        # Never serialize exception text: third-party clients can include
        # request headers or credentials in it.
        report["error"] = type(exc).__name__
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return report


def _wandb_remote_progress(
    current: Mapping[str, Any], previous: Mapping[str, Any] | None
) -> bool | None:
    if previous is None or not current.get("reachable"):
        return None
    prior = previous.get("wandb_remote", {})
    if not isinstance(prior, Mapping) or not prior.get("reachable"):
        return None
    def identities(report: Mapping[str, Any]) -> dict[str, tuple[str, int]]:
        result: dict[str, tuple[str, int]] = {}
        for item in report.get("runs", []):
            if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                result[item["id"]] = (
                    str(item.get("updated_at", "")),
                    int(item.get("last_history_step", 0) or 0),
                )
        return result
    return identities(current) != identities(prior)


def _load_previous(path: Path) -> Mapping[str, Any] | None:
    try:
        document = artifacts.load_strict_json(path)
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return document if isinstance(document, Mapping) else None


def collect_report(
    *,
    campaign_root: Path,
    campaign_contract: Path,
    campaign_contract_sha256: str,
    actor_patterns: Sequence[str],
    fleet_patterns: Sequence[str],
    telemetry_roots: Sequence[Path],
    wandb_roots: Sequence[Path],
    disk_paths: Sequence[Path],
    oci_nrt_host: str,
    slurm_user: str,
    slurm_job_prefix: str,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_expected_email: str | None = None,
    wandb_netrc: Path | None = None,
    required_process_patterns: Sequence[str] = (),
    required_process_pidfiles: Sequence[Path] = (),
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _regular_immutable(campaign_contract, campaign_contract_sha256, "runtime campaign contract")
    coordinator = _coordinator(campaign_root)
    prior_trees: dict[str, Mapping[str, Any]] = {}
    if previous is not None:
        for kind in ("telemetry", "wandb"):
            for entry in previous.get("file_movement", {}).get(kind, []):
                if isinstance(entry, Mapping) and isinstance(entry.get("root"), str):
                    prior_trees[entry["root"]] = entry

    file_movement: dict[str, list[dict[str, Any]]] = {"telemetry": [], "wandb": []}
    for kind, roots in (("telemetry", telemetry_roots), ("wandb", wandb_roots)):
        for root in roots:
            snapshot = _tree_snapshot(root)
            snapshot["movement"] = _movement(snapshot, prior_trees.get(str(root)))
            file_movement[kind].append(snapshot)

    warnings: list[str] = []
    fleet = _process_matches(fleet_patterns)
    required_processes = _process_matches(required_process_patterns)
    required_bound_processes = _bound_process_matches(
        required_process_pidfiles, campaign_root=campaign_root
    )
    actors = _process_matches(actor_patterns)
    actor_visible = any(item["count"] for item in actors.values())
    fleet_visible = any(item["count"] for item in fleet.values())
    if (
        coordinator["phase"] == "COLLECTING"
        and actor_patterns
        and not actor_visible
    ):
        warnings.append("COLLECTING phase has no configured local actor process visible")
    if fleet_patterns and not fleet_visible and not coordinator["completed"]:
        warnings.append("no configured local renderer-fleet process is visible")
    missing_required_processes = [
        pattern
        for pattern, report in required_processes.items()
        if not report["count"]
    ]
    missing_required_bound_processes = [
        path
        for path, report in required_bound_processes.items()
        if not report["healthy"]
    ]
    if missing_required_processes and not coordinator["completed"]:
        warnings.extend(
            f"required production process is not visible: {pattern}"
            for pattern in missing_required_processes
        )
    if missing_required_bound_processes and not coordinator["completed"]:
        warnings.extend(
            f"required campaign-bound process is not healthy: {path}"
            for path in missing_required_bound_processes
        )
    slurm = _slurm_health(oci_nrt_host, slurm_user, slurm_job_prefix)
    if not slurm["reachable"]:
        warnings.append("OCI-NRT Slurm status is unreachable")
    disks = _disk_health(disk_paths)
    for disk in disks:
        if disk.get("free_percent", 100.0) < 5.0:
            warnings.append(f"disk free space below 5%: {disk['path']}")
    gpu = _gpu_health_burst()
    prior_gpu_processes = (
        previous.get("gpu", {}).get("compute_processes", [])
        if isinstance(previous, Mapping)
        else []
    )
    prior_gpu_keys = {
        (item.get("pid"), item.get("gpu_uuid"), item.get("process_name"))
        for item in prior_gpu_processes
        if isinstance(item, Mapping)
    }
    for process in gpu.get("compute_processes", []):
        identity = (
            process.get("pid"),
            process.get("gpu_uuid"),
            process.get("process_name"),
        )
        process["temporal_classification"] = (
            "baseline_at_monitor_start"
            if previous is None
            else (
                "persisted_from_previous_poll"
                if identity in prior_gpu_keys
                else "new_since_previous_poll"
            )
        )
    warnings.extend(gpu.get("alerts", []))

    telemetry_progressed = _movement_progress(file_movement["telemetry"])
    local_wandb_progressed = _movement_progress(file_movement["wandb"])
    collection_active = coordinator["phase"] == "COLLECTING"
    telemetry_stale_polls = _stale_poll_count(
        previous=previous,
        key="telemetry_stale_polls",
        expected_now=collection_active,
        progressed=telemetry_progressed,
        coordinator=coordinator,
    )
    local_wandb_stale_polls = _stale_poll_count(
        previous=previous,
        key="local_wandb_stale_polls",
        expected_now=collection_active,
        progressed=local_wandb_progressed,
        coordinator=coordinator,
    )
    wandb_remote: dict[str, Any] | None = None
    remote_wandb_progressed: bool | None = None
    remote_wandb_stale_polls = 0
    remote_values = (
        wandb_entity,
        wandb_project,
        wandb_expected_email,
        wandb_netrc,
    )
    if any(value is not None for value in remote_values):
        if not all(value is not None for value in remote_values):
            raise ProductionMonitorError(
                "remote W&B health requires entity, project, expected email, and netrc"
            )
        wandb_remote = _wandb_remote_health(
            entity=wandb_entity,
            project=wandb_project,
            expected_email=wandb_expected_email,
            netrc_path=wandb_netrc,
            campaign_id=coordinator["campaign_id"],
            campaign_sha256=coordinator.get("campaign_sha256"),
            campaign_contract_sha256=campaign_contract_sha256,
        )
        remote_wandb_progressed = _wandb_remote_progress(wandb_remote, previous)
        remote_wandb_stale_polls = _stale_poll_count(
            previous=previous,
            key="remote_wandb_stale_polls",
            expected_now=coordinator["phase"] in {"COLLECTING", "INGESTED", "TRAINED"},
            progressed=remote_wandb_progressed,
            coordinator=coordinator,
        )
        if not wandb_remote["reachable"]:
            warnings.append("remote W&B API readback is unreachable")
        elif not wandb_remote["authorized"]:
            warnings.append("remote W&B API viewer does not match the expected account")
        elif wandb_remote["campaign_run_count"] == 0 and coordinator["next_macro_round"] > 0:
            warnings.append("remote W&B API found no run bound to this campaign")
        if remote_wandb_stale_polls >= 2:
            warnings.append(
                "remote W&B campaign runs have not moved for two consecutive 30-minute polls"
            )
    if telemetry_stale_polls >= 2:
        warnings.append(
            "COLLECTING telemetry has not moved for two consecutive 30-minute polls"
        )
    if wandb_roots and local_wandb_stale_polls >= 2:
        warnings.append(
            "COLLECTING local W&B artifacts have not moved for two consecutive 30-minute polls"
        )
    progress_health = {
        "telemetry_progressed": telemetry_progressed,
        "telemetry_stale_polls": telemetry_stale_polls,
        "local_wandb_progressed": local_wandb_progressed,
        "local_wandb_stale_polls": local_wandb_stale_polls,
        "remote_wandb_progressed": remote_wandb_progressed,
        "remote_wandb_stale_polls": remote_wandb_stale_polls,
        "stale_warning_after_polls": 2,
        "required_processes_healthy": not (
            missing_required_processes or missing_required_bound_processes
        ),
    }

    return {
        "schema": MONITOR_REPORT_SCHEMA,
        "observed_unix_ns": time.time_ns(),
        "interval_seconds": MONITOR_INTERVAL_SECONDS,
        "read_only": True,
        "campaign_contract_path": str(campaign_contract),
        "campaign_contract_sha256": campaign_contract_sha256,
        "campaign": coordinator,
        "local_process_health": {
            "actors": actors,
            "renderer_fleet": fleet,
            "required_processes": required_processes,
            "missing_required_processes": missing_required_processes,
            "required_bound_processes": required_bound_processes,
            "missing_required_bound_processes": missing_required_bound_processes,
        },
        "oci_nrt_slurm": slurm,
        "file_movement": file_movement,
        "wandb_remote": wandb_remote,
        "progress_health": progress_health,
        "disk": disks,
        "gpu": gpu,
        "warnings": warnings,
    }


def run_monitor(
    *,
    campaign_root: Path,
    campaign_contract: Path,
    campaign_contract_sha256: str,
    output_root: Path,
    actor_patterns: Sequence[str],
    fleet_patterns: Sequence[str],
    telemetry_roots: Sequence[Path],
    wandb_roots: Sequence[Path],
    disk_paths: Sequence[Path],
    oci_nrt_host: str,
    slurm_user: str,
    slurm_job_prefix: str,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_expected_email: str | None = None,
    wandb_netrc: Path | None = None,
    required_process_patterns: Sequence[str] = (),
    required_process_pidfiles: Sequence[Path] = (),
    max_polls: int | None = None,
    test_poll_interval_seconds: float | None = None,
) -> dict[str, Any]:
    _outside_campaign(output_root, campaign_root)
    if max_polls is not None and max_polls < 1:
        raise ProductionMonitorError("max_polls must be positive")
    if test_poll_interval_seconds is not None:
        if max_polls is None or test_poll_interval_seconds < 0:
            raise ProductionMonitorError("bounded test polling requires max_polls and a nonnegative interval")
        interval = test_poll_interval_seconds
    else:
        interval = MONITOR_INTERVAL_SECONDS
    latest = output_root / "latest.json"
    with MonitorLock(output_root):
        poll = 0
        previous = _load_previous(latest)
        while True:
            report = collect_report(
                campaign_root=campaign_root,
                campaign_contract=campaign_contract,
                campaign_contract_sha256=campaign_contract_sha256,
                actor_patterns=actor_patterns,
                fleet_patterns=fleet_patterns,
                telemetry_roots=telemetry_roots,
                wandb_roots=wandb_roots,
                disk_paths=disk_paths,
                oci_nrt_host=oci_nrt_host,
                slurm_user=slurm_user,
                slurm_job_prefix=slurm_job_prefix,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
                wandb_expected_email=wandb_expected_email,
                wandb_netrc=wandb_netrc,
                required_process_patterns=required_process_patterns,
                required_process_pidfiles=required_process_pidfiles,
                previous=previous,
            )
            poll += 1
            report["poll_index"] = poll
            snapshot_path = output_root / "snapshots" / f"{report['observed_unix_ns']}-{poll:06d}.json"
            _atomic_json(snapshot_path, report)
            _atomic_json(latest, report)
            print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
            previous = report
            if max_polls is not None and poll >= max_polls:
                return report
            time.sleep(interval)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--campaign-contract", type=Path, required=True)
    parser.add_argument("--campaign-contract-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--actor-pattern", action="append", default=[])
    parser.add_argument("--fleet-pattern", action="append", default=[])
    parser.add_argument("--required-process-pattern", action="append", default=[])
    parser.add_argument(
        "--required-process-pidfile", type=Path, action="append", default=[]
    )
    parser.add_argument("--telemetry-root", type=Path, action="append", default=[])
    parser.add_argument("--wandb-root", type=Path, action="append", default=[])
    parser.add_argument("--disk-path", type=Path, action="append", default=[])
    parser.add_argument("--oci-nrt-host", required=True)
    parser.add_argument("--slurm-user", required=True)
    parser.add_argument("--slurm-job-prefix", required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-expected-email", required=True)
    parser.add_argument("--wandb-netrc", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    # Explicitly test-only: production polling remains fixed at 1800 seconds.
    parser.add_argument("--test-max-polls", type=int)
    parser.add_argument("--test-poll-interval-seconds", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.once and arguments.test_max_polls is not None:
        print("production monitor failed: --once and --test-max-polls are mutually exclusive", file=os.sys.stderr)
        return 2
    try:
        campaign_root = _absolute(arguments.campaign_root, "campaign root")
        telemetry_roots = [
            _absolute(item, "telemetry root") for item in arguments.telemetry_root
        ] or [campaign_root / "rounds", campaign_root / "actors"]
        wandb_roots = [
            _absolute(item, "W&B root") for item in arguments.wandb_root
        ] or [campaign_root / "rounds", campaign_root / "actors"]
        max_polls = 1 if arguments.once else arguments.test_max_polls
        test_interval = (
            0.0 if arguments.once else arguments.test_poll_interval_seconds
        )
        result = run_monitor(
            campaign_root=campaign_root,
            campaign_contract=_absolute(arguments.campaign_contract, "campaign contract"),
            campaign_contract_sha256=arguments.campaign_contract_sha256,
            output_root=_absolute(arguments.output_root, "monitor output root"),
            actor_patterns=tuple(arguments.actor_pattern),
            fleet_patterns=tuple(arguments.fleet_pattern),
            required_process_patterns=tuple(arguments.required_process_pattern),
            required_process_pidfiles=tuple(
                _absolute(item, "required process pidfile")
                for item in arguments.required_process_pidfile
            ),
            telemetry_roots=tuple(telemetry_roots),
            wandb_roots=tuple(wandb_roots),
            disk_paths=tuple(_absolute(item, "disk path") for item in arguments.disk_path)
            or (campaign_root.parent,),
            oci_nrt_host=arguments.oci_nrt_host,
            slurm_user=arguments.slurm_user,
            slurm_job_prefix=arguments.slurm_job_prefix,
            wandb_entity=arguments.wandb_entity,
            wandb_project=arguments.wandb_project,
            wandb_expected_email=arguments.wandb_expected_email,
            wandb_netrc=_absolute(arguments.wandb_netrc, "W&B netrc"),
            max_polls=max_polls,
            test_poll_interval_seconds=test_interval,
        )
    except (ProductionMonitorError, OSError, ValueError) as exc:
        print(f"production monitor failed: {exc}", file=os.sys.stderr)
        return 2
    if max_polls is not None:
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


__all__ = [
    "MONITOR_INTERVAL_SECONDS",
    "ProductionMonitorError",
    "collect_report",
    "run_monitor",
]
