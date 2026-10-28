"""Run one frozen explicit-motion BC cohort on one owned Leo RTX GPU.

No H200, renderer, learner, upload, policy update, or implicit restart is used.
Plan-only is the default; --execute authorizes the bounded native workload.
The caller schedules independent cohort scenes and ingests only complete
manifests. Reusing an interrupted simulator/output identity is unsupported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from roborender.infrastructure.source_locations import resolve_source_release

try:
    from .isaac6_bc_production import load_request, validate_collection_request, native_overrides, publish_json
except ImportError:
    from isaac6_bc_production import load_request, validate_collection_request, native_overrides, publish_json


# Compatibility import: historical cohort scheduling remains here; reusable
# native command assembly is owned by the shared runtime boundary.
from pipeline.deployment.rtx_actor.bc_runtime import build_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--physical-gpu", type=int, choices=[0, 1], required=True)
    parser.add_argument("--memory-gib", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    request_path = args.request.resolve(strict=True)
    raw = request_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.request_sha256:
        raise ValueError("cohort request hash differs")
    request = validate_collection_request(json.loads(raw))
    if request['schema'] == 'roborender.bc-native-continuous-request.v1':
        raise ValueError('continuous collection requires pipeline.bc_collection.run_continuous and its capacity guards')
    source = resolve_source_release(args.source_root, error="native execution requires a frozen source release")
    manifest = json.loads(args.source_manifest.read_bytes())
    required_source = {"pipeline/deployment/rtx_actor/isaac6_bc_production.py",
                       "pipeline/deployment/rtx_actor/run_bc_cohort.py",
                       "pipeline/deployment/rtx_actor/bc_runtime.py",
                       "roborender/infrastructure/paths.py",
                       "pipeline/deployment/rtx_actor/native_motion_assignment.py"}
    observed = set()
    for row in manifest["files"]:
        path = (source / row["path"]).resolve(strict=True)
        if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("immutable native source release differs")
        observed.add(row["path"])
    if not required_source <= observed:
        raise ValueError("new BC adapter/launcher are absent from source attestation")
    runtime = args.runtime.resolve()
    if (runtime.exists() or not runtime.is_relative_to(Path("/home/ldu/roborender/runtime/bc-cohorts"))
            or not 60 <= args.timeout <= 12 * 3600):
        raise ValueError("fresh scoped runtime and bounded timeout required")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from materialize_isaac6 import materialize
    from qualify_native_isaac6 import RAVEN

    def command_for(overlay):
        # The qualified builder reads overlay/port-manifest.json. Materialize
        # first; do not call it against a not-yet-created fresh runtime.
        return build_command(request=request, request_path=request_path,
            request_sha256=args.request_sha256, name=args.name, runtime=runtime,
            overlay=overlay, source=source, physical_gpu=args.physical_gpu,
            memory_gib=args.memory_gib)

    if not args.execute:
        with tempfile.TemporaryDirectory(prefix="bc-cohort-preview-") as temporary:
            preview_overlay = Path(temporary) / "overlay"
            materialize(RAVEN, preview_overlay, source / "pipeline/dagger/rtx/config/assets.manifest.json")
            preview = command_for(preview_overlay)
            command = [arg.replace(str(preview_overlay), str(runtime / "overlay")) for arg in preview]
        print(json.dumps({"execute": False, "request_sha256": args.request_sha256,
            "cohort_id": request["cohort_id"], "num_envs": len(request["motions"]),
            "chunks_per_env": request["chunks_per_env"], "max_vector_steps": request["max_windows"] * 81,
            "physical_gpu": args.physical_gpu, "command": command}))
        return
    if socket.gethostname() != "leo-rtx-6k":
        raise RuntimeError("native BC cohort is explicitly bound to Leo")
    from qualify_graphics import cleanup_owned
    from run_success_diagnostic import gpu_gate
    from run_throughput_pair import memory_kib
    gpu_gate(args.physical_gpu, deployment_host="leo-rtx-6k")
    initial_memory = memory_kib()
    if initial_memory["MemAvailable"] < (args.memory_gib + 6) * 1024**2:
        raise RuntimeError("host RAM admission floor not satisfied")
    runtime.mkdir(parents=True)
    (runtime / "home").mkdir()
    materialize(RAVEN, runtime / "overlay", source / "pipeline/dagger/rtx/config/assets.manifest.json")
    command = command_for(runtime / "overlay")
    publish_json(runtime / "launch.json", {"command": command, "request_sha256": args.request_sha256,
        "source_sha256": manifest["source_sha256"], "physical_gpu": args.physical_gpu,
        "production_admitted": False})
    process, failure = None, None
    start = time.monotonic()

    def interrupted(signum, frame):
        raise KeyboardInterrupt("scoped BC cohort stop requested")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (runtime / "console.log").open("xb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if time.monotonic() - start > args.timeout:
                    raise TimeoutError("bounded native BC cohort timeout")
                memory = memory_kib()
                new_swap = (memory["SwapTotal"] - memory["SwapFree"]) - (initial_memory["SwapTotal"] - initial_memory["SwapFree"])
                if memory["MemAvailable"] < 6 * 1024**2 or new_swap > 128 * 1024:
                    raise RuntimeError("host RAM/swap safety floor violated")
                time.sleep(3)
    except BaseException as exc:
        failure = type(exc).__name__ + ": " + str(exc)
    finally:
        cleanup_owned(args.name)
        if process is not None and process.poll() is None:
            process.wait(timeout=30)
    result_path = runtime / "bc_isaac/cohort-manifest.json"
    result = json.loads(result_path.read_bytes()) if result_path.is_file() else {}
    passed = (failure is None and process is not None and process.returncode == 0
        and result.get("status") == "complete" and result.get("request_sha256") == args.request_sha256
        and result.get("selected_chunks") == sum(request["chunks_per_env"])
        and result.get("optimizer_steps") == 0)
    report = {"passed": bool(passed), "cohort_id": request["cohort_id"], "request_sha256": args.request_sha256,
        "elapsed_seconds": time.monotonic() - start, "failure": failure,
        "exit_code": process.returncode if process else None, "physical_gpu": args.physical_gpu,
        "cohort_manifest": str(result_path), "production_admitted": False}
    publish_json(runtime / "host-result.json", report)
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
