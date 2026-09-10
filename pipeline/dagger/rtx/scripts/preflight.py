#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))

from dagger_rtx.contracts import MODEL_VIEW_MAP, RAVEN_VIEW_ORDER
from dagger_rtx.capacity_safety import (
    attest_group_container_pids,
    parse_collector_processes,
    validate_capacity_group_id,
    validate_collector_ownership,
)
from dagger_rtx.provenance import verify_assets

IMAGE = "roborender-round-actor:isaaclab2.3.0-checkpoint-compat-20260901"
EXPECTED_IMAGE_ID = "sha256:fcaa595df8f175763ac326187a97dc8bbddca55d6bcc3531cdfecd6be1705a43"
EXPECTED_PHYSICAL_GPU_INDEX = 1
EXPECTED_PHYSICAL_GPU_UUID = "GPU-a28701dd-27fa-67d2-8b76-23be7a57fcf9"


def run_json(command: list[str]):
    result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only RTX Raven runtime and asset preflight")
    parser.add_argument("--probe-container", action="store_true", help="start a temporary no-network CUDA import probe; never initializes an environment")
    parser.add_argument(
        "--gpu",
        type=int,
        default=EXPECTED_PHYSICAL_GPU_INDEX,
        help="physical RTX GPU; this deployment is permanently restricted to GPU1",
    )
    parser.add_argument(
        "--capacity-group-id",
        help=(
            "allow only collector PIDs attested inside another certified container "
            "with this exact capacity group and a different physical GPU"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.gpu != EXPECTED_PHYSICAL_GPU_INDEX:
        raise RuntimeError(
            "this deployment is restricted to physical GPU1; GPU0 must never be probed or used"
        )
    capacity_group_id = validate_capacity_group_id(args.capacity_group_id)

    static, assets = verify_assets(ROOT / "config/assets.manifest.json")
    image_text = run_json(["docker", "image", "inspect", IMAGE, "--format", "{{json .}}"])
    image = json.loads(image_text)
    if image["Id"] != EXPECTED_IMAGE_ID:
        raise RuntimeError(f"container image ID mismatch: {image['Id']}")
    gpu_csv = run_json(
        [
            "nvidia-smi",
            "-i",
            EXPECTED_PHYSICAL_GPU_UUID,
            "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = [line for line in gpu_csv.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            "UUID-scoped nvidia-smi query did not return exactly one GPU1 row"
        )
    index, name, uuid, total, used, util = [part.strip() for part in rows[0].split(",")]
    selected = {
        "index": int(index),
        "name": name,
        "uuid": uuid,
        "memory_total_mib": int(total),
        "memory_used_mib": int(used),
        "utilization_percent": int(util),
    }
    if selected["index"] != EXPECTED_PHYSICAL_GPU_INDEX:
        raise RuntimeError(f"GPU UUID resolved to the wrong physical index: {selected}")
    if selected["uuid"] != EXPECTED_PHYSICAL_GPU_UUID:
        raise RuntimeError(f"physical GPU1 UUID mismatch: {selected}")
    if "RTX PRO 6000" not in selected["name"]:
        raise RuntimeError("physical GPU1 is not an RTX PRO 6000")
    if selected["memory_used_mib"] > 2048 or selected["utilization_percent"] > 10:
        raise RuntimeError(f"selected GPU {args.gpu} is not preflight-idle: {selected}")
    active_collectors_text = run_json(
        [
            "bash", "-lc",
            "pgrep -af 'run_raven_bc|distill_trainer_videogen|roborender-bc-' || true",
        ]
    )
    active_collectors = parse_collector_processes(active_collectors_text)
    allowed_collector_pids: set[int] = set()
    group_containers = []
    if capacity_group_id is not None:
        allowed_collector_pids, group_containers = attest_group_container_pids(
            capacity_group_id,
            selected_gpu=args.gpu,
            expected_image_id=EXPECTED_IMAGE_ID,
            run=run_json,
        )
    validate_collector_ownership(
        active_collectors,
        capacity_group_id=capacity_group_id,
        allowed_pids=allowed_collector_pids,
    )
    container_probe = None
    if args.probe_container:
        code = (
            "import json,torch; "
            "print(json.dumps({'python_cuda':torch.version.cuda,'torch':torch.__version__,"
            "'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),"
            "'gpu':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),"
            "'cuda_visible_devices':__import__('os').environ.get('CUDA_VISIBLE_DEVICES')},sort_keys=True))"
        )
        raw = run_json(
            [
                "docker", "run", "--rm", "--network", "none",
                "--gpus", f"device={EXPECTED_PHYSICAL_GPU_UUID}",
                "--env", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
                "--env", f"CUDA_VISIBLE_DEVICES={EXPECTED_PHYSICAL_GPU_UUID}",
                IMAGE, "/opt/round-dagger-rtx/bin/python", "-c", code,
            ]
        )
        container_probe = json.loads(raw.splitlines()[-1])
        if container_probe["cuda_available"] is not True:
            raise RuntimeError(f"GPU1 CUDA unavailable in isolated probe: {container_probe}")
        if container_probe["device_count"] != 1:
            raise RuntimeError(f"isolated GPU1 probe exposed more than one device: {container_probe}")
        if container_probe["cuda_visible_devices"] != EXPECTED_PHYSICAL_GPU_UUID:
            raise RuntimeError(f"isolated GPU1 UUID binding mismatch: {container_probe}")
        if container_probe["capability"] != [12, 0]:
            raise RuntimeError(f"unexpected GPU compute capability: {container_probe}")
    report = {
        "schema": "roborender.dagger.rtx.preflight.v1",
        "time_ns": time.time_ns(),
        "result": "PASS",
        "no_environment_initialized": True,
        "no_environment_step_executed": True,
        "online_view_order": list(RAVEN_VIEW_ORDER),
        "model_view_map": MODEL_VIEW_MAP,
        "static_provenance": static,
        "assets": assets,
        "container": {"tag": IMAGE, "id": image["Id"], "architecture": image["Architecture"]},
        "gpu_inventory_scope": "expected_gpu1_uuid_only",
        "gpus": [selected],
        "selected_gpu": args.gpu,
        "selected_gpu_uuid": EXPECTED_PHYSICAL_GPU_UUID,
        "collector_gate": {
            "capacity_group_id": capacity_group_id,
            "active_collectors": active_collectors,
            "attested_same_group_containers": group_containers,
            "allowed_collector_pids": sorted(allowed_collector_pids),
        },
        "container_probe": container_probe,
    }
    output = args.output or ROOT / "logs" / f"preflight-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.chmod(0o600)
    print(json.dumps({"result": "PASS", "report": str(output), "selected_gpu": args.gpu, "container_probe": container_probe}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
