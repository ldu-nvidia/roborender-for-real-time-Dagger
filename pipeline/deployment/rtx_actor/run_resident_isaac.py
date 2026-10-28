"""Launch one immutable standing Isaac worker on an explicitly bound Leo GPU.

No automatic restart policy, GPU reservations, driver reset or unknown cleanup.
The native worker is started only with --execute and an explicit workload ID.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from roborender.infrastructure.source_locations import resolve_source_release

from materialize_isaac6 import materialize
from qualify_native_isaac6 import IMAGE, REPO, RAVEN, command
from leo_binding import GPU_UUIDS
from run_success_diagnostic import gpu_gate
from run_throughput_pair import memory_kib, owned_memory_sample, native_completion, memory_safety


def build_resident_command(*, name, runtime, overlay, source, jobs, assets, token, source_sha, gpu, memory_gib=28):
    cmd = command(name, runtime, overlay, source / "pipeline/dagger/rtx/code", 6,
                  mode="capture-only", physical_gpu=gpu, seed=42+gpu, memory_gib=memory_gib,
                  compact_host_heap=True, deployment_host="leo-rtx-6k")
    cmd = ["trainer._target_=resident_isaac_trainer.ResidentIsaacTrainer"
           if arg.startswith("trainer._target_=") else arg for arg in cmd]
    cmd += ["env._target_=resident_isaac_trainer.ResidentTrainingEnvironment",
            "simulator._target_=resident_isaac_trainer.IsaacSim"]
    cmd = [arg.replace("PYTHONPATH=", "PYTHONPATH=/roborender-source:/resident-source:/evaluation-source:")
           if arg.startswith("PYTHONPATH=") else arg for arg in cmd]
    cmd[cmd.index("--network")+1] = "host"
    extra = ["--no-healthcheck", "--mount", f"type=bind,src={jobs},dst=/resident/jobs,readonly",
             "--mount", f"type=bind,src={assets},dst=/diagnostic-assets,readonly",
             "--mount", f"type=bind,src={token},dst=/resident/token,readonly",
             "--mount", f"type=bind,src={source},dst=/roborender-source,readonly",
             "--mount", f"type=bind,src={source / 'pipeline/deployment/rtx_actor'},dst=/resident-source,readonly",
             "--mount", f"type=bind,src={source / 'pipeline/evaluation'},dst=/evaluation-source,readonly",
             "--env", "RESIDENT_RUNTIME_ROOT=/dagger-runtime", "--env", "RESIDENT_JOBS_ROOT=/resident/jobs",
             "--env", "RESIDENT_TOKEN_FILE=/resident/token", "--env", f"RESIDENT_PORT={18640+gpu}",
             "--env", f"RESIDENT_PHYSICAL_GPU={gpu}", "--env", f"RESIDENT_SOURCE_SHA256={source_sha}",
             "--env", f"RESIDENT_INITIAL_SCENE_SEED={42+gpu}",
             "--env", "RESIDENT_MOTION_INVENTORY=/workspace/dagger_rtx_server/code/../config/fixed_panel_motion_inventory.v1.json",
             "--mount", f"type=bind,src={source / 'pipeline/dagger/rtx/config'},dst=/workspace/dagger_rtx_server/config,readonly"]
    cmd[cmd.index(IMAGE):cmd.index(IMAGE)] = extra
    return cmd


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt("scoped resident stop requested")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--physical-gpu", type=int, choices=[0, 1], required=True)
    parser.add_argument("--jobs-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--memory-gib", type=int, default=28)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"standing-isaac-leo-[a-z0-9-]{1,48}", args.run_id):
        raise ValueError("fresh dedicated standing workload identity required")
    source = resolve_source_release(args.source_root, error="standing workers require immutable source")
    manifest = json.loads(args.source_manifest.read_bytes())
    for row in manifest["files"]:
        path = source / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("standing worker source drift")
    runtime = REPO / "runtime/standing-isaac" / args.run_id / f"gpu{args.physical_gpu}"
    if runtime.exists() or not 21 <= args.memory_gib <= 32:
        raise ValueError("used runtime or unsafe worker memory cap")
    token = args.token_file.resolve(strict=True)
    if token.stat().st_mode & 0o077:
        raise ValueError("private resident token required")
    if not args.execute:
        print(json.dumps({"execute": False, "run_id": args.run_id, "physical_gpu": args.physical_gpu,
                          "port": 18640+args.physical_gpu, "memory_gib": args.memory_gib,
                          "source_sha256": manifest["source_sha256"]}))
        return
    if socket.gethostname() != "leo-rtx-6k":
        raise RuntimeError("wrong host for bound standing worker")
    gpu_gate(args.physical_gpu, deployment_host="leo-rtx-6k")
    import socket as sockets
    with sockets.socket() as listener:
        listener.bind(("127.0.0.1", 18640+args.physical_gpu))
    if memory_kib()["MemAvailable"] < (args.memory_gib+6)*1024**2:
        raise RuntimeError("standing worker host RAM admission failed")
    runtime.mkdir(parents=True)
    (runtime / "home").mkdir()
    overlay = runtime / "overlay"
    materialize(RAVEN, overlay, source / "pipeline/dagger/rtx/config/assets.manifest.json")
    name = args.run_id+f"-g{args.physical_gpu}"
    cmd = build_resident_command(name=name, runtime=runtime, overlay=overlay, source=source,
        jobs=args.jobs_root.resolve(strict=True), assets=args.assets_root.resolve(strict=True), token=token,
        source_sha=manifest["source_sha256"], gpu=args.physical_gpu, memory_gib=args.memory_gib)
    (runtime / "launch.json").write_text(json.dumps({"command": cmd, "source_sha256": manifest["source_sha256"]}, indent=2))
    with (runtime / "console.log").open("wb") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        (runtime / "launcher.json").write_text(json.dumps({"container": name, "docker_pid": process.pid,
                                                          "port": 18640+args.physical_gpu}))
        initial = memory_kib()
        started = time.monotonic()
        try:
            with (runtime / "resources.jsonl").open("w") as resources:
                while process.poll() is None:
                    if native_completion(process, runtime / "resident-state.json",
                                         completion_field="phase", completion_value="STOPPED") is not None:
                        break
                    # No transient empty cgroup attribution at startup; the
                    # live counter gate starts only after Docker creates it.
                    sample = owned_memory_sample(name, process, runtime / "resident-state.json",
                        completion_field="phase", completion_value="STOPPED")
                    current = memory_kib()
                    if time.monotonic()-started > 30 or "memory_swap.current" in sample:
                        safety = memory_safety(initial, current, [sample])
                    else:
                        safety = {"container_starting": True}
                    resources.write(json.dumps({"elapsed": time.monotonic()-started,
                        "mem_available_kib": current["MemAvailable"], "container": sample,
                        "memory_safety": safety})+"\n")
                    resources.flush()
                    if not (runtime / "resident-state.json").exists() and time.monotonic()-started > 1200:
                        raise TimeoutError("resident Isaac startup deadline; no automatic restart")
                    time.sleep(5)
                status = process.wait(timeout=30)
        except BaseException:
            # Only this exact labelled workload is ours. Preserve receipts;
            # killing native state is abandonment, never a resumable restart.
            from qualify_graphics import cleanup_owned
            cleanup_owned(name)
            process.wait(timeout=30)
            raise
    if status:
        raise RuntimeError("standing Isaac process failed; no automatic replay/restart")


if __name__ == "__main__":
    main()
