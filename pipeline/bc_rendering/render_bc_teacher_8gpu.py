#!/usr/bin/env python3
"""Plan and launch persistent 8-GPU × 4-lane offline BC teacher rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Raven used one flat videogen_dataset for rendering and offline BC. Keep the
# active parity path flat; train/validation/test partitioning is not part of
# Raven's published training flow.
SOURCE_ROOT = PROJECT_ROOT / "data/BC/isaac"
OUTPUT_ROOT = PROJECT_ROOT / "data/BC/rendered"
ADAPTER_SHA256 = "3f09aa23a26ddb2da32d027dc5e883c992c820854bc07020d6d254c806afb01b"
RAVEN_SEED = 42
RAVEN_RENDER_PROMPT = (
    PROJECT_ROOT
    / "pipeline/vendor/raven-fast-roborender-reference/render_prompt.txt"
)
RAVEN_NUM_INFERENCE_STEPS = 50
RAVEN_CFG_SCALE = 5.0
RAVEN_CONTROL_RESOLUTION = (240, 416)
RAVEN_STUDENT_RESOLUTION = (108, 192)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def discover_jobs() -> list[dict]:
    jobs: list[dict] = []
    raven_prompt = RAVEN_RENDER_PROMPT.read_text(encoding="utf-8").strip()
    if not raven_prompt:
        raise ValueError(f"empty canonical Raven prompt: {RAVEN_RENDER_PROMPT}")
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"missing Raven BC source directory: {SOURCE_ROOT}")
    manifest_path = SOURCE_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "sealed" or manifest.get("chunk_length") != 81:
        raise ValueError(f"source dataset contract differs: {manifest_path}")
    for item in manifest.get("items", []):
        source_npz = SOURCE_ROOT / item["path"]
        if source_npz.stat().st_size != item["size_bytes"]:
            raise ValueError(f"source payload size mismatch: {source_npz}")
        if sha256_file(source_npz) != item["sha256"]:
            raise ValueError(f"source payload hash mismatch: {source_npz}")
        with np.load(source_npz, allow_pickle=False) as payload:
            frame_count = int(payload["gt_actions"].shape[0])
            local_env_id = int(payload["env_id"].item())
        if frame_count != 81 or local_env_id != int(item.get("env_id", local_env_id)):
            raise ValueError(f"Raven chunk identity differs: {source_npz}")
        global_env_id = int(item.get("global_env_id", local_env_id))
        # Raven's NPZ schema has env_id and seq but no episode_id. Preserve an
        # optional staged lineage value when present; otherwise seq is the
        # non-scientific diagnostic identity.
        episode_id = int(item.get("episode_id", item["seq"]))
        jobs.append(
            {
                "source_npz": str(source_npz.resolve()),
                "source_metadata": str(manifest_path.resolve()),
                "source_payload_sha256": item["sha256"],
                "global_env_id": global_env_id,
                "env_id": local_env_id,
                "episode_id": episode_id,
                "sequence_id": int(item["seq"]),
                "frame_count": frame_count,
                # Raven's production three-view re-render overrides every
                # chunk prompt with render_prompt.txt and reuses seed 42.
                "prompt": raven_prompt,
                "seed": RAVEN_SEED,
                "output_npz": str((OUTPUT_ROOT / source_npz.name).resolve()),
                "output_dir": str((OUTPUT_ROOT / source_npz.stem).resolve()),
            }
        )
    return jobs


def assign_lpt(jobs: list[dict], count: int) -> tuple[list[list[dict]], list[int]]:
    assignments: list[list[dict]] = [[] for _ in range(count)]
    totals = [0] * count
    base, remainder = divmod(len(jobs), count)
    capacities = [base + (index < remainder) for index in range(count)]
    for job in sorted(
        jobs,
        key=lambda item: (
            -item["frame_count"],
            item["global_env_id"],
            item["episode_id"],
        ),
    ):
        eligible = [
            index for index in range(count) if len(assignments[index]) < capacities[index]
        ]
        target = min(
            eligible,
            key=lambda index: (totals[index], len(assignments[index]), index),
        )
        assignments[target].append(job)
        totals[target] += job["frame_count"]
    return assignments, totals


def verify_h200s(gpus: list[int]) -> list[dict]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,compute_cap,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory = {}
    for line in result.stdout.splitlines():
        index, name, uuid, compute_capability, memory_mib = [part.strip() for part in line.split(",")]
        inventory[int(index)] = {
            "index": int(index),
            "name": name,
            "uuid": uuid,
            "compute_capability": compute_capability,
            "memory_total_mib": int(memory_mib),
        }
    selected = []
    for gpu in gpus:
        item = inventory.get(gpu)
        if item is None:
            raise RuntimeError(f"GPU {gpu} is not visible")
        if "H200" not in item["name"] or item["compute_capability"] != "9.0":
            raise RuntimeError(f"GPU {gpu} is not an H200 SM90 device: {item}")
        selected.append(item)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--launch-stagger-seconds", type=float, default=0.25)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    gpus = [int(value) for value in args.gpus.split(",")]
    if len(gpus) != len(set(gpus)) or not gpus:
        raise ValueError("--gpus must contain unique GPU indices")
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be positive")

    adapter = PROJECT_ROOT / (
        "renderer/offline_teacher/checkpoints/"
        "roborender_offline_bc_video_teacher_50step_lora_step05294.safetensors"
    )
    foundation = PROJECT_ROOT / "renderer/models/wan21_fun_control_1p3b_foundation"
    source = PROJECT_ROOT / "renderer/offline_teacher/source"
    worker = Path(__file__).with_name("render_bc_teacher_worker.py")
    python = PROJECT_ROOT / "renderer/envs/image3f-py313-cu130-trt10133/venv/bin/python"
    for required in (adapter, foundation, source, worker, python, RAVEN_RENDER_PROMPT):
        if not required.exists():
            raise FileNotFoundError(required)
    if sha256_file(adapter) != ADAPTER_SHA256:
        raise RuntimeError("50-step teacher adapter hash mismatch")

    jobs = discover_jobs()
    if args.limit is not None:
        jobs = jobs[: args.limit]
    gpu_assignments, gpu_frame_totals = assign_lpt(jobs, len(gpus))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir
        else PROJECT_ROOT / f"runtime/outputs/bc_rendering/{timestamp}"
    )
    config = {
        "adapter_path": str(adapter.resolve()),
        "adapter_sha256": ADAPTER_SHA256,
        "foundation_path": str(foundation.resolve()),
        "source_path": str(source.resolve()),
        "num_inference_steps": RAVEN_NUM_INFERENCE_STEPS,
        "cfg_scale": RAVEN_CFG_SCALE,
        "height": RAVEN_CONTROL_RESOLUTION[0],
        "width": RAVEN_CONTROL_RESOLUTION[1],
        "student_height": RAVEN_STUDENT_RESOLUTION[0],
        "student_width": RAVEN_STUDENT_RESOLUTION[1],
        "seed": RAVEN_SEED,
        "prompt_file": str(RAVEN_RENDER_PROMPT.resolve()),
        "prompt_sha256": sha256_file(RAVEN_RENDER_PROMPT),
        # Diagnostic media only. Raven's learner consumes the NPZ arrays.
        "preview_fps": 15,
    }
    lane_plans: list[dict] = []
    for gpu_slot, physical_gpu in enumerate(gpus):
        lane_assignments, lane_totals = assign_lpt(
            gpu_assignments[gpu_slot], args.workers_per_gpu
        )
        for lane, lane_jobs in enumerate(lane_assignments):
            lane_plan = {
                "schema": "roborender.bc_teacher_lane_plan.v1",
                "physical_gpu": physical_gpu,
                "lane": lane,
                "frame_count": lane_totals[lane],
                "config": config,
                "jobs": lane_jobs,
            }
            lane_path = run_dir / "lanes" / f"gpu-{physical_gpu:02d}-lane-{lane:02d}.json"
            write_json(lane_path, lane_plan)
            lane_plans.append({"path": str(lane_path), **lane_plan})

    plan = {
        "schema": "roborender.bc_teacher_8gpu_plan.v1",
        "created_utc": timestamp,
        "status": "planned" if args.plan_only else "launching",
        "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "total_workers": len(lane_plans),
        "episode_count": len(jobs),
        "frame_count": sum(job["frame_count"] for job in jobs),
        "source_root": str(SOURCE_ROOT.resolve()),
        "output_root": str(OUTPUT_ROOT.resolve()),
        "gpu_frame_totals": {
            str(gpus[index]): gpu_frame_totals[index] for index in range(len(gpus))
        },
        "lane_plan_paths": [item["path"] for item in lane_plans],
        "config": config,
    }
    write_json(run_dir / "plan.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.plan_only:
        return

    inventory = verify_h200s(gpus)
    processes: list[tuple[subprocess.Popen, object, dict]] = []
    for lane_plan in lane_plans:
        if not lane_plan["jobs"]:
            continue
        lane_path = Path(lane_plan["path"])
        log_path = run_dir / "logs" / (lane_path.stem + ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(lane_plan["physical_gpu"]),
                "PYTHONPATH": str(source.resolve()),
                "DIFFSYNTH_MODEL_BASE_PATH": str((PROJECT_ROOT / "renderer/models").resolve()),
                "DIFFSYNTH_SKIP_DOWNLOAD": "true",
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        process = subprocess.Popen(
            [str(python), str(worker), "--job-file", str(lane_path)],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        processes.append((process, log_handle, lane_plan))
        time.sleep(args.launch_stagger_seconds)

    failure: tuple[int, dict] | None = None
    while processes:
        live = 0
        for process, _, lane_plan in processes:
            status = process.poll()
            if status is None:
                live += 1
            elif status != 0 and failure is None:
                failure = (status, lane_plan)
        print(f"workers_live={live}/{len(processes)}", flush=True)
        if failure or live == 0:
            break
        time.sleep(10)

    if failure:
        for process, _, _ in processes:
            if process.poll() is None:
                process.terminate()
    for process, log_handle, _ in processes:
        process.wait()
        log_handle.close()
    if failure:
        status, lane_plan = failure
        raise RuntimeError(
            f"worker failed with exit {status}: GPU {lane_plan['physical_gpu']} "
            f"lane {lane_plan['lane']}"
        )

    missing = [
        job["output_npz"]
        for job in jobs
        if not Path(job["output_npz"]).is_file()
    ]
    if missing:
        raise RuntimeError(f"{len(missing)} planned episode outputs are missing")
    plan["status"] = "complete"
    plan["hardware"] = inventory
    plan["completed_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_json(run_dir / "plan.json", plan)
    print(f"rendering complete: {len(jobs)} episodes, {plan['frame_count']} frames", flush=True)


if __name__ == "__main__":
    main()
