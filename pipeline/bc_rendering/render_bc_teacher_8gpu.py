#!/usr/bin/env python3
"""Plan, launch, and monitor persistent multi-GPU offline BC teacher rendering."""

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
# Transferred Raven-contract data is sealed per collection batch. Rendering
# keeps those boundaries for provenance while the learner may enumerate all
# rendered batches as one logical dataset.
SOURCE_ROOT = PROJECT_ROOT / "data/BC/isaac"
OUTPUT_ROOT = PROJECT_ROOT / "data/BC/rendered"
POOL_MANIFEST = PROJECT_ROOT / "data/BC/manifest.json"
POOL_SEAL = PROJECT_ROOT / "data/BC/SEALED.json"
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
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def publish_progress(
    run_dir: Path,
    lane_plans: list[dict],
    *,
    status: str,
) -> dict:
    """Atomically publish aggregate and per-GPU rendering progress."""
    per_gpu: dict[int, dict] = {}
    for lane in lane_plans:
        physical_gpu = int(lane["physical_gpu"])
        gpu = per_gpu.setdefault(
            physical_gpu,
            {
                "physical_gpu": physical_gpu,
                "assigned": 0,
                "done": 0,
                "rendered_this_run": 0,
                "skipped_preexisting": 0,
                "remaining": 0,
                "lane_statuses": {},
            },
        )
        assigned = len(lane["jobs"])
        gpu["assigned"] += assigned
        progress_path = Path(lane["progress_path"])
        observed = {}
        if progress_path.is_file():
            try:
                observed = json.loads(progress_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                observed = {}
        done = min(int(observed.get("done", 0)), assigned)
        rendered = min(int(observed.get("rendered_this_run", 0)), done)
        skipped = min(int(observed.get("skipped_preexisting", 0)), done - rendered)
        gpu["done"] += done
        gpu["rendered_this_run"] += rendered
        gpu["skipped_preexisting"] += skipped
        gpu["lane_statuses"][str(lane["lane"])] = observed.get("status", status)

    for gpu in per_gpu.values():
        gpu["remaining"] = gpu["assigned"] - gpu["done"]
        lane_states = set(gpu["lane_statuses"].values())
        if "failed" in lane_states:
            gpu["status"] = "failed"
        elif gpu["remaining"] == 0 and gpu["assigned"]:
            gpu["status"] = "complete"
        elif lane_states & {"loading", "rendering"}:
            gpu["status"] = "running"
        else:
            gpu["status"] = status

    assigned_total = sum(item["assigned"] for item in per_gpu.values())
    done_total = sum(item["done"] for item in per_gpu.values())
    snapshot = {
        "schema": "roborender.bc_teacher_progress.v1",
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": status,
        "assigned": assigned_total,
        "done": done_total,
        "rendered_this_run": sum(
            item["rendered_this_run"] for item in per_gpu.values()
        ),
        "skipped_preexisting": sum(
            item["skipped_preexisting"] for item in per_gpu.values()
        ),
        "remaining": assigned_total - done_total,
        "percent": 100.0 * done_total / assigned_total if assigned_total else 100.0,
        "gpus": [per_gpu[index] for index in sorted(per_gpu)],
    }
    write_json(run_dir / "progress.json", snapshot)
    text_lines = [
        (
            f"overall {done_total}/{assigned_total} "
            f"({snapshot['percent']:.1f}%) remaining={snapshot['remaining']} "
            f"rendered_this_run={snapshot['rendered_this_run']} "
            f"skipped_preexisting={snapshot['skipped_preexisting']} status={status}"
        )
    ]
    for gpu in snapshot["gpus"]:
        text_lines.append(
            f"GPU{gpu['physical_gpu']} {gpu['done']}/{gpu['assigned']} "
            f"remaining={gpu['remaining']} rendered_this_run={gpu['rendered_this_run']} "
            f"skipped_preexisting={gpu['skipped_preexisting']} status={gpu['status']}"
        )
    progress_text = run_dir / "progress.txt"
    temporary = progress_text.with_suffix(".txt.partial")
    temporary.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    os.replace(temporary, progress_text)
    return snapshot


def discover_jobs(
    *, limit: int | None = None, selection_seed: int | None = None
) -> list[dict]:
    candidates: list[dict] = []
    raven_prompt = RAVEN_RENDER_PROMPT.read_text(encoding="utf-8").strip()
    if not raven_prompt:
        raise ValueError(f"empty canonical Raven prompt: {RAVEN_RENDER_PROMPT}")
    if not SOURCE_ROOT.is_dir() or not POOL_MANIFEST.is_file() or not POOL_SEAL.is_file():
        raise FileNotFoundError("missing unified sealed BC pool")
    manifest = json.loads(POOL_MANIFEST.read_text(encoding="utf-8"))
    seal = json.loads(POOL_SEAL.read_text(encoding="utf-8"))
    items = manifest.get("chunks", [])
    if (
        manifest.get("status") != "sealed"
        or int(manifest.get("chunk_count", -1)) != len(items)
        or seal.get("verdict") != "PASS"
        or int(seal.get("isaac_chunks", -1)) != len(items)
    ):
        raise ValueError("unified BC pool manifest/seal contract failed")
    for item in items:
        source_npz = (SOURCE_ROOT / Path(item["isaac_npz"]).name).resolve()
        try:
            source_npz.relative_to(SOURCE_ROOT.resolve())
        except ValueError as error:
            raise ValueError(f"chunk escapes unified Isaac pool: {source_npz}") from error
        candidates.append(
            {
                "source_npz": source_npz,
                "manifest_path": POOL_MANIFEST,
                "seal_path": POOL_SEAL,
                "item": item,
                "prompt": raven_prompt,
            }
        )

    if selection_seed is not None:
        np.random.default_rng(selection_seed).shuffle(candidates)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        candidates = candidates[:limit]

    jobs: list[dict] = []
    outputs: set[str] = set()
    for candidate in candidates:
        item = candidate["item"]
        source_npz = candidate["source_npz"]
        if source_npz.stat().st_size != item["source_size_bytes"]:
            raise ValueError(f"source payload size mismatch: {source_npz}")
        payload_sha256 = sha256_file(source_npz)
        if payload_sha256 != item["source_sha256"]:
            raise ValueError(f"source payload hash mismatch: {source_npz}")
        with np.load(source_npz, allow_pickle=False) as payload:
            frame_count = int(payload["gt_actions"].shape[0])
            local_env_id = int(payload["env_id"].item())
        manifest_local_env_id = int(
            item.get("local_env_id", local_env_id)
        )
        if frame_count != 81 or local_env_id != manifest_local_env_id:
            raise ValueError(f"Raven chunk identity differs: {source_npz}")
        global_env_id = int(item.get("global_env_id", local_env_id))
        # Raven's NPZ schema has env_id and seq but no episode_id. Preserve an
        # optional staged lineage value when present; otherwise seq is the
        # non-scientific diagnostic identity.
        episode_id = int(item.get("episode_id", item["sequence_id"]))
        output_npz = (OUTPUT_ROOT / source_npz.name).resolve()
        if str(output_npz) in outputs:
            raise ValueError(f"duplicate rendered output: {output_npz}")
        outputs.add(str(output_npz))
        jobs.append(
            {
                "source_npz": str(source_npz.resolve()),
                "source_metadata": str(candidate["manifest_path"].resolve()),
                "source_seal": str(candidate["seal_path"].resolve()),
                "source_payload_sha256": payload_sha256,
                "global_env_id": global_env_id,
                "env_id": local_env_id,
                "episode_id": episode_id,
                "sequence_id": int(item["sequence_id"]),
                "frame_count": frame_count,
                # Raven's production three-view re-render overrides every
                # chunk prompt with render_prompt.txt and reuses seed 42.
                "prompt": candidate["prompt"],
                "seed": RAVEN_SEED,
                "output_npz": str(output_npz),
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
    # One persistent worker saturates an H200 for this workload. More workers
    # only duplicate the model in memory and contend for the same GPU compute.
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--selection-seed",
        type=int,
        help="reproducibly shuffle chunks before applying --limit",
    )
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

    jobs = discover_jobs(
        limit=args.limit,
        selection_seed=args.selection_seed,
    )
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
            lane_plan["progress_path"] = str(
                (
                    run_dir
                    / "progress"
                    / f"gpu-{physical_gpu:02d}-lane-{lane:02d}.json"
                ).resolve()
            )
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
        "source_pool": "unified",
        "output_root": str(OUTPUT_ROOT.resolve()),
        "gpu_frame_totals": {
            str(gpus[index]): gpu_frame_totals[index] for index in range(len(gpus))
        },
        "lane_plan_paths": [item["path"] for item in lane_plans],
        "config": config,
    }
    write_json(run_dir / "plan.json", plan)
    publish_progress(run_dir, lane_plans, status="planned")
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
    publish_progress(run_dir, lane_plans, status="running")
    while processes:
        live = 0
        for process, _, lane_plan in processes:
            status = process.poll()
            if status is None:
                live += 1
            elif status != 0 and failure is None:
                failure = (status, lane_plan)
        snapshot = publish_progress(run_dir, lane_plans, status="running")
        print(
            f"workers_live={live}/{len(processes)} "
            f"chunks_done={snapshot['done']}/{snapshot['assigned']}",
            flush=True,
        )
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
        publish_progress(run_dir, lane_plans, status="failed")
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
    publish_progress(run_dir, lane_plans, status="complete")
    print(f"rendering complete: {len(jobs)} episodes, {plan['frame_count']} frames", flush=True)


if __name__ == "__main__":
    main()
