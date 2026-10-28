"""Bounded, source-pinned multi-cohort BC collection with one native Kit boot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from roborender.infrastructure.source_locations import resolve_source_release

try:
    from .isaac6_bc_session import validate_session
    from .isaac6_bc_production import publish_json
    from .run_bc_cohort import build_command as cohort_command
    from .bc_execution_evidence import canonical, collector_identity, publish_completed_session
except ImportError:
    from isaac6_bc_session import validate_session
    from isaac6_bc_production import publish_json
    from run_bc_cohort import build_command as cohort_command
    from bc_execution_evidence import canonical, collector_identity, publish_completed_session


def read_requests(session, directory):
    directory = Path(directory).resolve(strict=True)

    def reader(path):
        path = Path(path)
        if path.parent != Path("/bc-session-requests"):
            raise ValueError("session requests must be direct children of their private mount")
        host = directory / path.name
        if host.is_symlink() or not host.is_file():
            raise ValueError("request must be a regular immutable file")
        return host.read_bytes()

    return validate_session(session, read_bytes=reader)


def build_command(*, session, requests, session_path, session_sha256, requests_dir,
                  runtime, overlay, source, physical_gpu, memory_gib, base_builder=None, image_id=None):
    first = session["requests"][0]
    request_path = Path(requests_dir) / Path(first["path"]).name
    # The existing cohort builder scopes its Docker name with this prefix.
    name = "bc-cohort-leo-" + session["session_id"].removeprefix("bc-session-leo-")
    cmd = cohort_command(request=requests[0], request_path=request_path,
        request_sha256=first["sha256"], name=name, runtime=runtime, overlay=overlay,
        source=source, physical_gpu=physical_gpu, memory_gib=memory_gib,
        base_builder=base_builder, image_id=image_id)
    for i, value in enumerate(cmd):
        if value.startswith("trainer._target_="):
            cmd[i] = "trainer._target_=isaac6_bc_session.Isaac6BCSessionTrainer"
        elif value.startswith("ROBORENDER_BC_COLLECTION_REQUEST="):
            cmd[i] = "ROBORENDER_BC_COLLECTION_REQUEST=" + first["path"]
    if image_id is None:
        from qualify_native_isaac6 import IMAGE
        image_id = IMAGE
    extra = ["--mount", f"type=bind,src={session_path},dst=/run/bc-session.json,readonly",
        "--mount", f"type=bind,src={requests_dir},dst=/bc-session-requests,readonly",
        "--env", "ROBORENDER_BC_SESSION=/run/bc-session.json",
        "--env", "ROBORENDER_BC_SESSION_SHA256=" + session_sha256]
    at = cmd.index(image_id)
    cmd[at:at] = extra
    return name, cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--session-sha256", required=True)
    parser.add_argument("--requests-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, choices=[0, 1], required=True)
    parser.add_argument("--memory-gib", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    raw = args.session.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.session_sha256:
        raise ValueError("session hash changed")
    session, requests = read_requests(json.loads(raw), args.requests_dir)
    source = resolve_source_release(args.source_root, error="native session requires a frozen source release")
    manifest = json.loads(args.source_manifest.read_bytes())
    required = {"pipeline/deployment/rtx_actor/" + name for name in (
        "isaac6_bc_session.py", "run_bc_session.py", "isaac6_bc_production.py", "run_bc_cohort.py",
        "native_motion_assignment.py", "bc_execution_evidence.py")}
    observed = set()
    for row in manifest["files"]:
        path = (source / row["path"]).resolve(strict=True)
        if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("frozen session source differs")
        observed.add(row["path"])
    if not required <= observed:
        raise ValueError("session source files missing from manifest")
    runtime = args.runtime.resolve()
    if (runtime.exists() or not runtime.is_relative_to("/home/ldu/roborender/runtime/bc-sessions")
            or not 60 <= args.timeout <= 12 * 3600):
        raise ValueError("fresh scoped runtime and bounded timeout required")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from materialize_isaac6 import materialize
    from qualify_native_isaac6 import RAVEN, IMAGE
    if not args.execute:
        with tempfile.TemporaryDirectory(prefix="bc-session-plan-") as temporary:
            overlay = Path(temporary) / "overlay"
            materialize(RAVEN, overlay, source / "pipeline/dagger/rtx/config/assets.manifest.json")
            name, cmd = build_command(session=session, requests=requests, session_path=args.session.resolve(),
                session_sha256=args.session_sha256, requests_dir=args.requests_dir.resolve(), runtime=runtime,
                overlay=overlay, source=source, physical_gpu=args.physical_gpu, memory_gib=args.memory_gib)
            print(json.dumps({"execute": False, "cohorts": len(requests), "name": name,
                "maximum_vector_steps": sum(r["max_windows"] * 81 for r in requests), "command": cmd}))
        return
    if socket.gethostname() != "leo-rtx-6k":
        raise RuntimeError("native session is explicitly bound to Leo")
    from qualify_graphics import cleanup_owned
    from run_success_diagnostic import gpu_gate
    from run_throughput_pair import memory_kib
    gpu_gate(args.physical_gpu, deployment_host="leo-rtx-6k")
    initial = memory_kib()
    if initial["MemAvailable"] < (args.memory_gib + 6) * 1024**2:
        raise RuntimeError("host RAM admission floor not satisfied")
    runtime.mkdir(parents=True)
    (runtime / "home").mkdir()
    materialize(RAVEN, runtime / "overlay", source / "pipeline/dagger/rtx/config/assets.manifest.json")
    name, cmd = build_command(session=session, requests=requests, session_path=args.session.resolve(),
        session_sha256=args.session_sha256, requests_dir=args.requests_dir.resolve(), runtime=runtime,
        overlay=runtime / "overlay", source=source, physical_gpu=args.physical_gpu, memory_gib=args.memory_gib)
    port_raw = (runtime / "overlay/port-manifest.json").read_bytes()
    execution = collector_identity(manifest, json.loads(port_raw), IMAGE)
    publish_json(runtime / "launch.json", {"command": cmd, "source_sha256": manifest["source_sha256"],
        "source_manifest_raw_sha256": hashlib.sha256(args.source_manifest.read_bytes()).hexdigest(),
        "port_manifest_raw_sha256": hashlib.sha256(port_raw).hexdigest(), "collector_execution": execution,
        "session_canonical_sha256": hashlib.sha256(canonical(session)).hexdigest(),
        "request_identities": {row["sha256"]: hashlib.sha256(canonical(request)).hexdigest()
                               for row, request in zip(session["requests"], requests)},
        "session_sha256": args.session_sha256, "production_admitted": False})
    process, failure = None, None
    started = time.monotonic()
    min_available = initial["MemAvailable"]

    def interrupted(signum, frame):
        raise KeyboardInterrupt("scoped BC session stop requested")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (runtime / "console.log").open("xb") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if time.monotonic() - started > args.timeout:
                    raise TimeoutError("bounded native BC session timeout")
                current = memory_kib()
                min_available = min(min_available, current["MemAvailable"])
                new_swap = (current["SwapTotal"] - current["SwapFree"]) - (initial["SwapTotal"] - initial["SwapFree"])
                if current["MemAvailable"] < 6 * 1024**2 or new_swap > 128 * 1024:
                    raise RuntimeError("host RAM/swap safety floor violated")
                time.sleep(3)
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        cleanup_owned(name)
        if process is not None and process.poll() is None:
            process.wait(timeout=30)
    path = runtime / "bc-session-result.json"
    result = json.loads(path.read_bytes()) if path.is_file() else {}
    passed = (failure is None and process is not None and process.returncode == 0
        and result.get("status") == "complete" and result.get("native_initializations") == 1
        and len(result.get("cohorts", [])) == len(requests))
    report = {"passed": passed, "session_id": session["session_id"], "failure": failure,
        "exit_code": process.returncode if process else None, "elapsed_seconds": time.monotonic() - started,
        "minimum_host_available_kib": min_available, "production_admitted": False}
    publish_json(runtime / "host-result.json", report)
    if passed:
        report["execution_evidence"] = publish_completed_session(runtime, session, report, publish_json)
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
