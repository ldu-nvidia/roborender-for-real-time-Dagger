"""Bounded GPU1-only diagnostic runner. No driver changes, training, or H200 use."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import socket
import subprocess
import time
import uuid

GPU_UUID = "GPU-0c1949ae-379a-2f2e-0f23-621de2eee409"
HOSTNAME = "465d6b0-lcedt"
OUTPUT_ROOT = Path("/home/ldu/roborender/runtime/rtx_container_qualification")
LABEL = "org.roborender.qualification"


def command(image_id, name, output):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("an immutable Docker image ID is required")
    return [
        "docker", "run", "--rm", "--name", name, "--label", f"{LABEL}={name}",
        "--gpus", f"device={GPU_UUID}", "--network", "none", "--shm-size", "4g",
        "--memory", "32g", "--cpus", "8", "--ulimit", "core=0",
        "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility",
        "--env", f"CUDA_VISIBLE_DEVICES={GPU_UUID}",
        "--env", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "--mount", f"type=bind,src={output},dst=/qualification", image_id,
    ]


def run_json(args):
    return json.loads(subprocess.check_output(args, text=True, timeout=30))


def cleanup_owned(name):
    result = subprocess.run(["docker", "inspect", "--type", "container", name],
                            capture_output=True, text=True, timeout=20)
    if result.returncode:
        return
    info = json.loads(result.stdout)[0]
    if (info.get("Config", {}).get("Labels") or {}).get(LABEL) != name:
        raise RuntimeError("cleanup refused: container ownership mismatch")
    if info["State"]["Running"]:
        subprocess.run(["docker", "stop", "--time", "10", name], check=True, timeout=25)
    # --rm owns container removal. Host evidence is retained, never deleted.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not 30 <= args.timeout <= 600:
        parser.error("timeout must be 30..600 seconds")
    identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    name = "roborender-graphics-probe-" + identity.lower()
    output = OUTPUT_ROOT / identity
    cmd = command(args.image_id, name, output)
    if args.plan_only:
        print(json.dumps({"command": cmd, "executes": False}, indent=2))
        return
    if socket.gethostname() != HOSTNAME:
        raise RuntimeError("this runner is restricted to RTX beta")
    # Isaac 6 specifies 32 GB minimum RAM. Leave at least 8 GiB available for
    # the host rather than testing below that floor or risking host OOM.
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    memory_available_kib = int(meminfo["MemAvailable"].split()[0])
    if memory_available_kib < 40 * 1024 * 1024:
        raise RuntimeError("less than 40 GiB host RAM available; no diagnostic started")
    image = run_json(["docker", "image", "inspect", args.image_id])[0]
    if image["Id"] != args.image_id or (image["Config"].get("Labels") or {}).get(
            "org.roborender.role") != "graphics-qualification-only":
        raise RuntimeError("image ID/diagnostic role mismatch")
    inventory = []
    for _ in range(3):
        row = subprocess.check_output([
            "nvidia-smi", "-i", GPU_UUID,
            "--query-gpu=index,uuid,driver_version,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=15).strip().split(",")
        if len(row) != 5:
            raise RuntimeError("GPU1 inventory did not return exactly one row")
        index, gpu, driver, memory, util = [x.strip() for x in row]
        if index != "1" or gpu != GPU_UUID or int(memory) > 2048 or int(util) > 10:
            raise RuntimeError("GPU1 is not identity-verified and idle; no job started")
        inventory.append({"uuid": gpu, "driver": driver, "memory_mib": int(memory), "util": int(util)})
        time.sleep(1)
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps({
        "image_id": args.image_id, "command": cmd, "gpu_samples": inventory,
        "host_memory_available_kib": memory_available_kib,
        "scope": "graphics_only_no_Raven_no_training", "timeout": args.timeout,
    }, indent=2) + "\n")
    started = time.monotonic()
    exit_code, timed_out = None, False
    try:
        with (output / "console.log").open("wb") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            try:
                exit_code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                cleanup_owned(name)
                exit_code = process.wait(timeout=30)
    finally:
        cleanup_owned(name)
        result_path = output / "probe-result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        passed = exit_code == 0 and result is not None and result.get("result") == "PASS"
        report = {"passed": passed, "timed_out": timed_out, "exit_code": exit_code,
                  "elapsed_seconds": time.monotonic() - started, "output": str(output),
                  "production_qualified": False, "image_id": args.image_id}
        (output / "host-result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
