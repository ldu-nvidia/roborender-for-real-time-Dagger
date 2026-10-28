"""Bounded real Isaac/teacher qualification on the dedicated Leo workstation.

Uses the same image, materializer and native callbacks as Beta. This owns only
its named diagnostic container. It never starts a learner or a campaign.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from roborender.infrastructure.source_locations import resolve_source_release

from leo_binding import HOSTNAME, checked_gpu
from materialize_isaac6 import materialize
from qualify_native_isaac6 import IMAGE, RAVEN, REPO, command, cleanup_owned


def source_gate(root, manifest_path):
    root = resolve_source_release(root, error="native tests require an immutable source release")
    sys.path.insert(0, str(root))
    from pipeline.deployment.source_snapshot import canonical, file_record
    manifest = json.loads(manifest_path.read_bytes())
    rows = [file_record(root, row["path"]) for row in manifest["files"]]
    if (rows != manifest["files"] or len(rows) != manifest["file_count"]
            or hashlib.sha256(canonical(rows)).hexdigest() != manifest["source_sha256"]):
        raise ValueError("immutable source release mismatch")
    return manifest["source_sha256"]


def memory_kib():
    return {k: int(v.split()[0]) for k, v in
            (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--physical-gpu", type=int, choices=[0, 1], required=True)
    p.add_argument("--num-envs", type=int, choices=[1, 2, 6, 8, 12, 16], default=2)
    p.add_argument("--mode", choices=["capture-only", "teacher-capacity", "bc-collect", "isolated-one-step", "single-window"], default="capture-only")
    p.add_argument("--capacity-steps", type=int, choices=[3, 10], default=10)
    p.add_argument("--run-id")
    p.add_argument("--token-file", type=Path)
    p.add_argument("--h200-pid", type=int)
    p.add_argument("--seed", type=int, default=610130)
    p.add_argument("--memory-gib", type=int, choices=[21, 32, 40], default=32)
    p.add_argument("--timeout", type=int, default=1200)
    a = p.parse_args()
    if socket.gethostname() != HOSTNAME or not 60 <= a.timeout <= 1800:
        raise RuntimeError("wrong host or unbounded native timeout")
    gpu_uuid = checked_gpu(a.physical_gpu, "capture-only" if a.mode == "teacher-capacity" else a.mode)
    source_sha = source_gate(a.source_root, a.source_manifest)
    if not 0 <= a.seed < 2**31 - 1:
        raise ValueError("invalid simulator seed")
    if a.mode in {"isolated-one-step", "single-window"}:
        if not a.token_file or not a.token_file.is_file() or a.token_file.stat().st_mode & 0o077:
            raise RuntimeError("owner-only scoped H200 token is required")
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", IMAGE], timeout=30))[0]
    if image["Id"] != IMAGE or image["Config"].get("Labels", {}).get("org.roborender.runtime") != "isaac6-lab3-migration-candidate":
        raise RuntimeError("pinned native image identity mismatch")
    samples = []
    for _ in range(3):
        raw = subprocess.check_output(["nvidia-smi", "-i", gpu_uuid,
            "--query-gpu=index,uuid,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits"], text=True, timeout=15).strip().split(",")
        index, device, used, util, driver = [value.strip() for value in raw]
        if int(index) != a.physical_gpu or device != gpu_uuid or int(used) > 2048 or int(util) > 10:
            raise RuntimeError("requested Leo RTX device is not identity-verified and idle")
        samples.append({"physical_gpu": int(index), "uuid": device, "used_mib": int(used), "util": int(util), "driver": driver})
        time.sleep(1)
    mem = memory_kib()
    if mem["MemAvailable"] < (a.memory_gib + 6) * 1024 * 1024:
        raise RuntimeError("insufficient host RAM; keep 6 GiB outside actor cap")
    identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = REPO / "runtime/leo-native-qualification" / identity
    output.mkdir(parents=True, exist_ok=False)
    (output / "home").mkdir()
    (REPO / f"runtime/isaac6-cache-leo-gpu{a.physical_gpu}").mkdir(parents=True, exist_ok=True)
    overlay = output / "overlay"
    materialize(RAVEN, overlay, a.source_root / "pipeline/dagger/rtx/config/assets.manifest.json")
    name = "roborender-leo-qualification-" + identity.lower()
    cmd = command(name, output, overlay, a.source_root / "pipeline/dagger/rtx/code", a.num_envs,
                  mode=a.mode, run_id=a.run_id, token_file=a.token_file, h200_pid=a.h200_pid,
                  physical_gpu=a.physical_gpu, seed=a.seed, memory_gib=a.memory_gib,
                  compact_host_heap=True, deployment_host=HOSTNAME, capacity_steps=a.capacity_steps)
    plan = {"source_sha256": source_sha, "image_id": IMAGE, "mode": a.mode,
            "command": cmd, "gpu_samples": samples, "host_memory_available_kib": mem["MemAvailable"],
            "production_admitted": False, "automatic_learner": False}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"phase": "native_started", "mode": a.mode, "output": str(output), "container": name,
                      "physical_gpu": a.physical_gpu, "num_envs": a.num_envs}), flush=True)
    def interrupted(signum, frame):
        raise InterruptedError("scoped Leo qualification stop requested")
    signal.signal(signal.SIGTERM, interrupted)
    start = time.monotonic()
    process = None
    failure = None
    try:
        with (output / "console.log").open("wb") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if time.monotonic() - start > a.timeout:
                    raise TimeoutError("native qualification timeout")
                now = memory_kib()
                if now["MemAvailable"] < 6 * 1024 * 1024 or (now["SwapTotal"] - now["SwapFree"]) - (mem["SwapTotal"] - mem["SwapFree"]) > 128 * 1024:
                    raise RuntimeError("host RAM/swap safety floor violated")
                time.sleep(3)
    except BaseException as exc:
        failure = type(exc).__name__ + ": " + str(exc)
    finally:
        cleanup_owned(name)
        if process and process.poll() is None:
            process.wait(timeout=30)
    result_name = {"capture-only": "capture-smoke.json", "teacher-capacity": "capture-smoke.json",
                   "bc-collect": "bc-collection-smoke.json"}.get(a.mode, "native-service-result.json")
    result_path = output / result_name
    if result_path.is_file():
        subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/chown",
            "--mount", f"type=bind,src={output},dst=/evidence", IMAGE, f"{os.getuid()}:{os.getgid()}",
            "/evidence/" + result_name], check=True, timeout=30)
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    passed = failure is None and process is not None and process.returncode == 0 and result.get("result") == "PASS"
    if a.mode == "capture-only":
        passed = passed and result.get("num_envs") == a.num_envs and result.get("environment_step_calls") == 0
    elif a.mode == "teacher-capacity":
        passed = passed and (result.get("mode") == a.mode and result.get("num_envs") == a.num_envs
            and result.get("environment_step_calls") == a.capacity_steps
            and result.get("physical_transition_rows") == a.capacity_steps * a.num_envs
            and result.get("optimizer_steps") == 0 and result.get("chunks_published") == 0
            and result.get("H200_renderer_tested") is False and result.get("production_qualified") is False)
    elif a.mode == "bc-collect":
        passed = passed and result.get("vector_steps") == 243 and result.get("optimizer_steps") == 0
    else:
        transaction = result.get("transaction", {})
        passed = passed and result.get("mode") == a.mode and result.get("num_envs") == a.num_envs
        if a.mode == "single-window":
            passed = passed and transaction.get("transition_batches") == 81 and transaction.get("physical_transition_rows") == 81 * a.num_envs
        else:
            passed = passed and transaction.get("phase") == "COMPLETE" and len(transaction.get("remote_acknowledgements", [])) == a.num_envs
    report = {"passed": bool(passed), "mode": a.mode, "physical_gpu": a.physical_gpu, "gpu_uuid": gpu_uuid,
              "num_envs": a.num_envs, "elapsed_seconds": time.monotonic() - start,
              "source_sha256": source_sha, "image_id": IMAGE, "output": str(output),
              "failure": failure, "exit_code": process.returncode if process else None,
              "production_admitted": False}
    (output / "host-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
