#!/usr/bin/env python3
"""Launch and supervise one cooperative Image3F service per local GPU.

This entry point is intentionally local-process based.  It is used by the RTX
actor host for staged validation and round-based DAgger collection; cluster
fleets continue to use ``image3f_server_array.sbatch``.  Children are never
sent SIGTERM/SIGKILL.  Shutdown is requested through the server's STOP file
and is accepted only after every child publishes a valid STOP acknowledgement
and exits successfully.
"""

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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scaling.image3f_probe import capability_error, probe  # noqa: E402
from scripts.server.renderer_identity import (  # noqa: E402
    canonical_json_sha256 as _renderer_json_sha256,
    regular_file_sha256,
    validate_renderer_identity,
)


SCHEMA = "fast_roborender.local_image3f_fleet.v1"
STOP_SCHEMA = "fast_roborender.renderer_stop_request.v1"
ACK_SCHEMA = "fast_roborender.image3f_stop_ack.v1"
DEFAULT_SERVER = ROOT / "scripts/server/image3f_server.py"
GPU_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_VIEWS = ("head_color", "hand_left", "hand_right")
EXPECTED_TILE_SHAPE = (240, 416, 3)
EXPECTED_STACKED_SHAPE = (720, 416, 3)


class FleetError(RuntimeError):
    """A fail-closed fleet lifecycle or readiness error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FleetError(f"refusing to replace symlink: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FleetError(f"temporary output already exists: {temporary}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FleetError(f"refusing to replace symlink: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except FileNotFoundError as exc:
        raise FleetError(f"missing {label}: {path}") from exc
    if not stat.S_ISREG(mode):
        raise FleetError(f"{label} must be a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FleetError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FleetError(f"{label} must contain a JSON object: {path}")
    return payload


def _parse_gpus(raw: str) -> tuple[str, ...]:
    gpus = tuple(token.strip() for token in raw.split(","))
    if not gpus or any(not token for token in gpus):
        raise FleetError("--gpus must be a non-empty comma-separated list")
    if any(not GPU_RE.fullmatch(token) for token in gpus):
        raise FleetError("--gpus contains an unsafe GPU identifier")
    if len(set(gpus)) != len(gpus):
        raise FleetError("--gpus must not contain duplicates")
    return gpus


def _require_path(path: Path, *, directory: bool, label: str) -> Path:
    path = path.expanduser().resolve(strict=True)
    mode = path.stat(follow_symlinks=False).st_mode
    expected = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
    if not expected:
        kind = "directory" if directory else "regular file"
        raise FleetError(f"{label} must be a {kind}: {path}")
    return path


def _require_executable(path: Path, *, label: str) -> Path:
    """Validate an executable without resolving away a virtualenv entry path.

    A venv's ``bin/python`` is normally a symlink.  Invoking its resolved base
    interpreter bypasses ``pyvenv.cfg`` and therefore the venv site-packages.
    Validate the final target, but retain the absolute lexical path for exec.
    """

    lexical = path.expanduser().absolute()
    try:
        target = lexical.resolve(strict=True)
    except OSError as exc:
        raise FleetError(f"missing {label}: {lexical}") from exc
    if not stat.S_ISREG(target.stat(follow_symlinks=False).st_mode):
        raise FleetError(f"{label} target must be a regular file: {target}")
    if not os.access(lexical, os.X_OK):
        raise FleetError(f"{label} is not executable: {lexical}")
    return lexical


def _prepare_new_fleet_dir(path: Path) -> Path:
    path = path.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FleetError(
            f"fleet directory already exists; use a fresh path to avoid stale state: {path}"
        )
    path.mkdir(parents=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise FleetError(f"fleet directory is unsafe: {path}")
    for child in ("endpoints", "queues", "stop_acks", "logs"):
        (path / child).mkdir(mode=0o700)
    return path


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_stop(fleet_dir: Path, *, run_id: str, reason: str) -> dict[str, Any]:
    stop = fleet_dir / "STOP"
    payload = {
        "schema": STOP_SCHEMA,
        "run_id": run_id,
        "reason": reason,
        "requested_at": _utc_now(),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(
            stop,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        existing = _load_json(stop, "STOP request")
        if existing.get("schema") != STOP_SCHEMA or existing.get("run_id") != run_id:
            raise FleetError("existing STOP request belongs to another fleet run")
        return existing
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def _safe_unlink_regular(path: Path) -> None:
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISREG(mode):
        raise FleetError(f"refusing to clean non-regular runtime file: {path}")
    path.unlink()


def _validate_acknowledgements(
    fleet_dir: Path,
    *,
    run_id: str,
    count: int,
) -> list[dict[str, Any]]:
    stop = fleet_dir / "STOP"
    try:
        stop_bytes = stop.read_bytes()
    except OSError as exc:
        raise FleetError(f"cannot read STOP request: {exc}") from exc
    stop_sha256 = hashlib.sha256(stop_bytes).hexdigest()
    rows: list[dict[str, Any]] = []
    expected_names = {f"renderer_{index:03d}.json" for index in range(count)}
    actual_paths = list((fleet_dir / "stop_acks").iterdir())
    if {path.name for path in actual_paths} != expected_names:
        raise FleetError("STOP acknowledgement set is incomplete or contains residue")
    for index in range(count):
        path = fleet_dir / "stop_acks" / f"renderer_{index:03d}.json"
        row = _load_json(path, f"renderer {index} STOP acknowledgement")
        expected_id = f"{run_id}:{index:03d}"
        if (
            row.get("schema") != ACK_SCHEMA
            or row.get("status") != "STOP_observed"
            or row.get("server_id") != expected_id
            or row.get("stop_sentinel_sha256") != stop_sha256
        ):
            raise FleetError(f"renderer {index} STOP acknowledgement is incompatible")
        rows.append(row)
    return rows


def _build_probe_report(
    endpoint_dir: Path,
    *,
    expected: int,
    timeout_s: float,
    required_reference_lag: int,
    expected_release_id: str | None,
    expected_adapter_sha256: str | None,
    expected_engine_sha256: str | None,
    expected_inference_steps: int,
    expected_teacache_threshold: float,
    expected_cfg_scale: float,
    expected_cuda_graph: bool,
    expected_server_script_sha256: str | None = None,
    expected_server_ids: set[str] | None = None,
) -> tuple[dict[str, Any], bool]:
    paths = sorted(endpoint_dir.glob("renderer_*.txt"))
    unsafe = []
    for path in paths:
        try:
            if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
                unsafe.append(path)
        except FileNotFoundError:
            unsafe.append(path)
    safe_paths = [path for path in paths if path not in unsafe]
    healthy, errors = probe(safe_paths, timeout_s) if safe_paths else ([], [])
    errors.extend(f"{path}: endpoint is not a regular file" for path in unsafe)
    capable: list[dict[str, Any]] = []
    for item in healthy:
        response = item.get("response")
        if not isinstance(response, dict):
            errors.append(f"{item.get('file')}: ping response is not a mapping")
            continue
        incompatible = capability_error(
            response,
            minimum_protocol_version=2,
            required_reference_lag=required_reference_lag,
            require_reference_history=True,
            expected_release_id=expected_release_id,
            expected_adapter_sha256=expected_adapter_sha256,
            expected_inference_steps=expected_inference_steps,
            expected_teacache_threshold=expected_teacache_threshold,
            expected_cfg_scale=expected_cfg_scale,
            expected_cuda_graph=expected_cuda_graph,
            expected_engine_sha256=expected_engine_sha256,
            require_tensorrt_engine=True,
            expected_output_views=EXPECTED_VIEWS,
            expected_output_layout="vertical",
            expected_output_tile_shape=EXPECTED_TILE_SHAPE,
            expected_output_stacked_shape=EXPECTED_STACKED_SHAPE,
        )
        if incompatible is not None:
            errors.append(f"{item.get('file')}: incompatible endpoint: {incompatible}")
            continue
        if expected_server_script_sha256 is not None:
            try:
                renderer_identity = validate_renderer_identity(
                    response.get("renderer_identity"),
                    runtime_identity=response.get("runtime_identity"),
                    expected_server_script_sha256=expected_server_script_sha256,
                )
                renderer_digest = _renderer_json_sha256(renderer_identity)
                if response.get("renderer_identity_sha256") != renderer_digest:
                    raise RuntimeError(
                        "declared renderer identity SHA-256 does not match its document"
                    )
            except RuntimeError as exc:
                errors.append(
                    f"{item.get('file')}: incompatible endpoint renderer identity: {exc}"
                )
                continue
        capable.append(item)
    endpoint_pairs = {(row["host"], row["port"]) for row in capable}
    identities = {
        _canonical_json_sha256(row["response"]["runtime_identity"])
        for row in capable
    }
    renderer_identities = {
        str(row["response"].get("renderer_identity_sha256"))
        for row in capable
        if isinstance(row["response"].get("renderer_identity_sha256"), str)
    }
    expected_names = {f"renderer_{index:03d}.txt" for index in range(expected)}
    actual_names = {path.name for path in paths}
    server_ids = {
        str(row["response"].get("server_id", "")) for row in capable
    }
    ready = (
        len(capable) == expected
        and len(endpoint_pairs) == expected
        and len(identities) == 1
        and (
            expected_server_script_sha256 is None
            or len(renderer_identities) == 1
        )
        and actual_names == expected_names
        and len(server_ids) == expected
        and (expected_server_ids is None or server_ids == expected_server_ids)
        and not errors
    )
    return {
        "schema": f"{SCHEMA}.readiness",
        "expected": expected,
        "discovered": len(paths),
        "healthy": len(healthy),
        "capable": len(capable),
        "unique_endpoints": len(endpoint_pairs),
        "runtime_identity_sha256": next(iter(identities)) if len(identities) == 1 else None,
        "renderer_identity_mode": (
            "content_bound_v1"
            if expected_server_script_sha256 is not None
            else "legacy_runtime_only"
        ),
        "renderer_identity_sha256": (
            next(iter(renderer_identities))
            if expected_server_script_sha256 is not None
            and len(renderer_identities) == 1
            else None
        ),
        "server_script_sha256": expected_server_script_sha256,
        "server_ids": sorted(server_ids),
        "endpoints": capable,
        "errors": sorted(errors),
        "ready": ready,
        "checked_at": _utc_now(),
    }, ready


def _state_path(fleet_dir: Path) -> Path:
    return fleet_dir / "supervisor.json"


def _write_state(
    fleet_dir: Path,
    *,
    run_id: str,
    status: str,
    services: Sequence[Mapping[str, Any]],
    detail: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "run_id": run_id,
        "status": status,
        "supervisor_pid": os.getpid(),
        "services": list(services),
        "updated_at": _utc_now(),
    }
    if detail:
        payload["detail"] = detail
    _atomic_json(_state_path(fleet_dir), payload)


def _wait_for_children(
    children: Sequence[subprocess.Popen[bytes]], timeout_s: float
) -> list[int] | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        codes = [child.poll() for child in children]
        if all(code is not None for code in codes):
            return [int(code) for code in codes if code is not None]
        time.sleep(min(0.1, max(0.01, deadline - time.monotonic())))
    return None


def run_fleet(args: argparse.Namespace) -> int:
    gpus = _parse_gpus(args.gpus)
    image3f_root = _require_path(args.image3f_root, directory=True, label="Image3F root")
    model_root = _require_path(args.model_root, directory=True, label="model root")
    python = _require_executable(args.python, label="Image3F Python")
    server_script = _require_path(args.server_script, directory=False, label="server script")
    server_script_sha256 = regular_file_sha256(server_script)
    fleet_dir = _prepare_new_fleet_dir(args.fleet_dir)
    run_id = args.run_id or f"local-{int(time.time())}-{os.getpid()}"
    services: list[dict[str, Any]] = []
    plan = {
        "schema": f"{SCHEMA}.plan",
        "run_id": run_id,
        "fleet_dir": str(fleet_dir),
        "gpus": list(gpus),
        "image3f_root": str(image3f_root),
        "model_root": str(model_root),
        "python": str(python),
        "server_script": str(server_script),
        "server_script_sha256": server_script_sha256,
        "dynamic_ports": True,
        "required_reference_lag": args.required_reference_lag,
        "expected_release_id": args.expected_release_id,
        "expected_adapter_sha256": args.expected_adapter_sha256,
        "expected_engine_sha256": args.expected_engine_sha256,
        "expected_inference_steps": args.expected_inference_steps,
        "expected_teacache_threshold": args.expected_teacache_threshold,
        "expected_cfg_scale": args.expected_cfg_scale,
        "expected_cuda_graph": args.expected_cuda_graph == "enabled",
        "created_at": _utc_now(),
    }
    _atomic_json(fleet_dir / "plan.json", plan)
    children: list[subprocess.Popen[bytes]] = []
    log_handles = []
    stop_requested = False
    became_ready = False

    def request_stop_from_signal(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        signal_number: signal.signal(signal_number, request_stop_from_signal)
        for signal_number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        for index, gpu in enumerate(gpus):
            endpoint = fleet_dir / "endpoints" / f"renderer_{index:03d}.txt"
            queue = fleet_dir / "queues" / f"renderer_{index:03d}"
            ack = fleet_dir / "stop_acks" / f"renderer_{index:03d}.json"
            queue.joinpath("inbox").mkdir(parents=True)
            queue.joinpath("done").mkdir()
            log = fleet_dir / "logs" / f"renderer_{index:03d}.log"
            handle = log.open("xb")
            log_handles.append(handle)
            server_id = f"{run_id}:{index:03d}"
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "IMAGE3F_ROOT": str(image3f_root),
                    "ROBORENDER_MODEL_BASE": str(model_root),
                    "ENDPOINT_FILE": str(endpoint),
                    "IMAGE3F_QUEUE": str(queue),
                    "IMAGE3F_SERVER_ID": server_id,
                    "IMAGE3F_STOP_FILE": str(fleet_dir / "STOP"),
                    "IMAGE3F_STOP_ACK_FILE": str(ack),
                    "IMAGE3F_STOP_POLL_S": str(args.stop_poll_s),
                    "IMAGE3F_PORT": "0",
                    "IMAGE3F_ADVERTISE_HOST": args.advertise_host,
                    "IMAGE3F_MAX_STREAMS": str(args.max_streams),
                    "IMAGE3F_MAX_REFERENCE_LAG": str(args.max_reference_lag),
                    "IMAGE3F_ALLOW_PY_QUEUE": "0",
                    "IMAGE3F_EXPECT_SERVER_SCRIPT_SHA256": server_script_sha256,
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            child = subprocess.Popen(
                [str(python), str(server_script)],
                cwd=str(ROOT),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                # Keep terminal SIGINT/SIGTERM on the supervisor. It translates
                # them to STOP rather than letting the signal bypass ACK/drain.
                start_new_session=True,
            )
            children.append(child)
            services.append(
                {
                    "index": index,
                    "gpu": gpu,
                    "server_id": server_id,
                    "pid": child.pid,
                    "endpoint_file": str(endpoint),
                    "queue_dir": str(queue),
                    "ack_file": str(ack),
                    "log_file": str(log),
                    "port_selection": "kernel_dynamic",
                }
            )
        _write_state(
            fleet_dir,
            run_id=run_id,
            status="starting",
            services=services,
        )
        deadline = time.monotonic() + args.startup_timeout_s
        last_report: dict[str, Any] = {}
        while time.monotonic() < deadline:
            if stop_requested:
                _write_stop(fleet_dir, run_id=run_id, reason="supervisor_signal")
                break
            exited = [child.poll() for child in children]
            if any(code is not None for code in exited):
                _write_stop(fleet_dir, run_id=run_id, reason="child_exited_during_startup")
                break
            last_report, ready = _build_probe_report(
                fleet_dir / "endpoints",
                expected=len(gpus),
                timeout_s=args.connect_timeout_s,
                required_reference_lag=args.required_reference_lag,
                expected_release_id=args.expected_release_id,
                expected_adapter_sha256=args.expected_adapter_sha256,
                expected_engine_sha256=args.expected_engine_sha256,
                expected_inference_steps=args.expected_inference_steps,
                expected_teacache_threshold=args.expected_teacache_threshold,
                expected_cfg_scale=args.expected_cfg_scale,
                expected_cuda_graph=args.expected_cuda_graph == "enabled",
                expected_server_script_sha256=server_script_sha256,
                expected_server_ids={
                    f"{run_id}:{index:03d}" for index in range(len(gpus))
                },
            )
            if ready:
                became_ready = True
                _atomic_json(fleet_dir / "readiness.json", last_report)
                _atomic_text(
                    fleet_dir / "endpoint_manifest.txt",
                    "".join(
                        f"{fleet_dir / 'endpoints' / f'renderer_{index:03d}.txt'}\n"
                        for index in range(len(gpus))
                    ),
                )
                _write_state(
                    fleet_dir,
                    run_id=run_id,
                    status="ready",
                    services=services,
                )
                break
            time.sleep(args.poll_s)
        else:
            _atomic_json(fleet_dir / "readiness_failed.json", last_report)
            _write_stop(fleet_dir, run_id=run_id, reason="startup_timeout")

        if not (fleet_dir / "readiness.json").is_file():
            stop_requested = True

        while not stop_requested and not (fleet_dir / "STOP").exists():
            codes = [child.poll() for child in children]
            if any(code is not None for code in codes):
                _write_stop(fleet_dir, run_id=run_id, reason="unexpected_child_exit")
                stop_requested = True
                break
            time.sleep(args.poll_s)
        if stop_requested and not (fleet_dir / "STOP").exists():
            _write_stop(fleet_dir, run_id=run_id, reason="supervisor_stop")
        stop_requested = True
        _write_state(
            fleet_dir,
            run_id=run_id,
            status="stopping",
            services=services,
        )
        codes = _wait_for_children(children, args.shutdown_timeout_s)
        if codes is None:
            _write_state(
                fleet_dir,
                run_id=run_id,
                status="stop_timeout",
                services=services,
                detail="children left running; no signals were sent",
            )
            raise FleetError("cooperative renderer shutdown timed out; children were not killed")
        if any(code != 0 for code in codes):
            _write_state(
                fleet_dir,
                run_id=run_id,
                status="failed",
                services=services,
                detail=f"renderer exit codes: {codes}",
            )
            raise FleetError(f"renderer fleet exited unsuccessfully: {codes}")
        try:
            acknowledgements = _validate_acknowledgements(
                fleet_dir, run_id=run_id, count=len(gpus)
            )
            for index in range(len(gpus)):
                _safe_unlink_regular(
                    fleet_dir / "endpoints" / f"renderer_{index:03d}.txt"
                )
                _safe_unlink_regular(
                    fleet_dir / "queues" / f"renderer_{index:03d}" / "READY"
                )
        except FleetError as exc:
            _write_state(
                fleet_dir,
                run_id=run_id,
                status="failed",
                services=services,
                detail=str(exc),
            )
            raise
        _atomic_json(
            fleet_dir / "clean_stop.json",
            {
                "schema": f"{SCHEMA}.clean_stop",
                "run_id": run_id,
                "renderer_count": len(gpus),
                "exit_codes": codes,
                "acknowledgements": acknowledgements,
                "readiness_succeeded": became_ready,
                "stopped_at": _utc_now(),
            },
        )
        if not became_ready:
            raise FleetError(
                "renderer fleet stopped cleanly but never passed the readiness gate"
            )
        _write_state(
            fleet_dir,
            run_id=run_id,
            status="stopped",
            services=services,
        )
        return 0
    except Exception as exc:
        # A partially started fleet must not be orphaned.  Request the same
        # cooperative shutdown used by operators, but never signal a child.
        if children:
            cleanup_detail = ""
            try:
                _write_stop(fleet_dir, run_id=run_id, reason="supervisor_error")
                codes = _wait_for_children(children, args.shutdown_timeout_s)
                cleanup_detail = (
                    "cooperative cleanup timed out; children were left running"
                    if codes is None
                    else f"cooperative cleanup exit codes: {codes}"
                )
            except Exception as cleanup_exc:  # preserve the original failure too
                cleanup_detail = f"cooperative cleanup also failed: {cleanup_exc}"
            try:
                _write_state(
                    fleet_dir,
                    run_id=run_id,
                    status="failed",
                    services=services,
                    detail=f"{type(exc).__name__}: {exc}; {cleanup_detail}",
                )
            except Exception:
                pass
        if isinstance(exc, FleetError):
            raise
        raise FleetError(f"fleet supervisor failed: {type(exc).__name__}: {exc}") from exc
    finally:
        for signal_number, handler in previous_handlers.items():
            signal.signal(signal_number, handler)
        for handle in log_handles:
            handle.close()


def status_fleet(args: argparse.Namespace) -> int:
    fleet_dir = args.fleet_dir.expanduser().absolute()
    state = _load_json(_state_path(fleet_dir), "fleet supervisor state")
    if state.get("schema") != SCHEMA:
        raise FleetError("fleet supervisor state has an incompatible schema")
    status = state.get("status")
    report: dict[str, Any] = {"state": state}
    if status == "ready":
        plan = _load_json(fleet_dir / "plan.json", "fleet plan")
        expected = len(plan.get("gpus", []))
        readiness, healthy = _build_probe_report(
            fleet_dir / "endpoints",
            expected=expected,
            timeout_s=args.connect_timeout_s,
            required_reference_lag=int(plan.get("required_reference_lag", 0)),
            expected_release_id=plan.get("expected_release_id"),
            expected_adapter_sha256=plan.get("expected_adapter_sha256"),
            expected_engine_sha256=plan.get("expected_engine_sha256"),
            expected_inference_steps=int(plan["expected_inference_steps"]),
            expected_teacache_threshold=float(plan["expected_teacache_threshold"]),
            expected_cfg_scale=float(plan["expected_cfg_scale"]),
            expected_cuda_graph=bool(plan["expected_cuda_graph"]),
            expected_server_script_sha256=plan.get("server_script_sha256"),
            expected_server_ids={
                f"{state['run_id']}:{index:03d}" for index in range(expected)
            },
        )
        report["live_readiness"] = readiness
        return_code = 0 if healthy else 2
    elif status == "stopped":
        report["clean_stop"] = _load_json(
            fleet_dir / "clean_stop.json", "clean-stop evidence"
        )
        return_code = 0
    else:
        return_code = 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return return_code


def stop_fleet(args: argparse.Namespace) -> int:
    fleet_dir = args.fleet_dir.expanduser().absolute()
    state = _load_json(_state_path(fleet_dir), "fleet supervisor state")
    if state.get("schema") != SCHEMA:
        raise FleetError("fleet supervisor state has an incompatible schema")
    run_id = state.get("run_id")
    services = state.get("services")
    if not isinstance(run_id, str) or not isinstance(services, list) or not services:
        raise FleetError("fleet supervisor state is incomplete")
    if state.get("status") == "stopped":
        _load_json(fleet_dir / "clean_stop.json", "clean-stop evidence")
        return 0
    _write_stop(fleet_dir, run_id=run_id, reason=args.reason)
    deadline = time.monotonic() + args.wait_timeout_s
    while time.monotonic() < deadline:
        latest = _load_json(_state_path(fleet_dir), "fleet supervisor state")
        if latest.get("run_id") != run_id:
            raise FleetError("fleet supervisor run identity changed during shutdown")
        if latest.get("status") == "stopped":
            _load_json(fleet_dir / "clean_stop.json", "clean-stop evidence")
            print(f"STOPPED: {len(services)} Image3F renderers")
            return 0
        if latest.get("status") in {"failed", "stop_timeout"}:
            raise FleetError(f"fleet shutdown failed: {latest.get('detail', latest['status'])}")
        time.sleep(args.poll_s)
    raise FleetError("timed out waiting for cooperative fleet shutdown; no signals were sent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="launch and supervise the fleet in foreground")
    run.add_argument("--fleet-dir", type=Path, required=True)
    run.add_argument("--gpus", required=True, help="comma-separated GPU indices/UUIDs")
    run.add_argument("--image3f-root", type=Path, required=True)
    run.add_argument("--model-root", type=Path, required=True)
    run.add_argument("--python", type=Path, required=True)
    run.add_argument("--server-script", type=Path, default=DEFAULT_SERVER)
    run.add_argument("--run-id")
    run.add_argument("--advertise-host", default="127.0.0.1")
    run.add_argument("--max-streams", type=int, default=64)
    run.add_argument("--max-reference-lag", type=int, default=512)
    run.add_argument("--required-reference-lag", type=int, default=256)
    run.add_argument("--startup-timeout-s", type=float, default=1800.0)
    run.add_argument("--shutdown-timeout-s", type=float, default=120.0)
    run.add_argument("--connect-timeout-s", type=float, default=2.0)
    run.add_argument("--poll-s", type=float, default=0.5)
    run.add_argument("--stop-poll-s", type=float, default=0.5)
    run.add_argument("--expected-release-id")
    run.add_argument("--expected-adapter-sha256")
    run.add_argument("--expected-engine-sha256")
    run.add_argument("--expected-inference-steps", type=int, default=5)
    run.add_argument("--expected-teacache-threshold", type=float, default=0.3)
    run.add_argument("--expected-cfg-scale", type=float, default=1.0)
    run.add_argument(
        "--expected-cuda-graph", choices=("enabled", "disabled"), default="disabled"
    )
    run.set_defaults(func=run_fleet)

    status = subparsers.add_parser("status", help="probe a running or stopped fleet")
    status.add_argument("--fleet-dir", type=Path, required=True)
    status.add_argument("--connect-timeout-s", type=float, default=2.0)
    status.set_defaults(func=status_fleet)

    stop = subparsers.add_parser("stop", help="request cooperative STOP and wait for ACKs")
    stop.add_argument("--fleet-dir", type=Path, required=True)
    stop.add_argument("--reason", default="operator_request")
    stop.add_argument("--wait-timeout-s", type=float, default=180.0)
    stop.add_argument("--poll-s", type=float, default=0.2)
    stop.set_defaults(func=stop_fleet)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "startup_timeout_s",
        "shutdown_timeout_s",
        "connect_timeout_s",
        "poll_s",
        "stop_poll_s",
        "wait_timeout_s",
    ):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            raise FleetError(f"--{name.replace('_', '-')} must be positive")
    if getattr(args, "max_streams", 1) < 1:
        raise FleetError("--max-streams must be positive")
    max_lag = getattr(args, "max_reference_lag", 0)
    required_lag = getattr(args, "required_reference_lag", 0)
    if max_lag < required_lag or required_lag < 0:
        raise FleetError("reference lag bounds are inconsistent")
    stop_poll = getattr(args, "stop_poll_s", 1.0)
    if stop_poll > 2.0:
        raise FleetError("--stop-poll-s must be at most 2 seconds")
    for name in ("expected_adapter_sha256", "expected_engine_sha256"):
        value = getattr(args, name, None)
        if value is not None and not SHA256_RE.fullmatch(value):
            raise FleetError(f"--{name.replace('_', '-')} must be lowercase SHA-256")
    expected_steps = getattr(args, "expected_inference_steps", 1)
    if expected_steps < 1:
        raise FleetError("--expected-inference-steps must be positive")
    for name in ("expected_teacache_threshold", "expected_cfg_scale"):
        value = getattr(args, name, 0.0)
        if not math.isfinite(value) or value < 0:
            raise FleetError(f"--{name.replace('_', '-')} must be finite and non-negative")
    run_id = getattr(args, "run_id", None)
    if run_id is not None and (not run_id or not GPU_RE.fullmatch(run_id)):
        raise FleetError("--run-id contains unsafe characters")
    advertise_host = getattr(args, "advertise_host", "127.0.0.1")
    if not advertise_host or any(character.isspace() for character in advertise_host):
        raise FleetError("--advertise-host must be one non-empty host token")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        return int(args.func(args))
    except FleetError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
