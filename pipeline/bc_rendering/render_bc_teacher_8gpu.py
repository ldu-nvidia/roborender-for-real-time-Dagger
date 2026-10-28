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

try:
    from .render_recipes import CURRENT_RECIPE, RECIPE_NAMES, context_metadata, recipe_config, collection_provenance
except ImportError:
    from render_recipes import CURRENT_RECIPE, RECIPE_NAMES, context_metadata, recipe_config, collection_provenance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Transferred Raven-contract data is sealed per collection batch. Rendering
# keeps those boundaries for provenance while the learner may enumerate all
# rendered batches as one logical dataset.
SOURCE_ROOT = PROJECT_ROOT / "data/BC/isaac"
OUTPUT_ROOT = PROJECT_ROOT / "data/BC/rendered_independent_v2"
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
RAVEN_MAX_CHAIN = 1
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


def raven_stream_tag(item: dict) -> str:
    """Recover Raven's exact `basename.split('_')[1]` grouping tag."""
    collection_filename = Path(str(item.get("collection_filename", ""))).name
    parts = collection_filename.split("_")
    if len(parts) > 1 and parts[1]:
        return parts[1]
    return str(item.get("source_run_tag", f"batch-{item.get('source_batch', -1)}"))


def source_stream_identity(job: dict) -> dict:
    """Keep independent collectors distinct even when local env IDs repeat."""
    return {
        "stream_tag": str(job["stream_tag"]),
        "source_run_tag": str(job.get("source_run_tag", job["stream_tag"])),
        "source_batch": int(job.get("source_batch", -1)),
        "rank": int(job.get("rank", 0)),
        "global_env_id": int(job.get("global_env_id", job["env_id"])),
        "local_env_id": int(job["env_id"]),
    }


def discover_jobs(
    *,
    limit: int | None = None,
    selection_seed: int | None = None,
    output_root: Path | None = None,
    source_root: Path | None = None,
    pool_manifest: Path | None = None,
    pool_seal: Path | None = None,
    verify_source_hashes: bool = False,
) -> list[dict]:
    if output_root is None:
        output_root = OUTPUT_ROOT
    # Resolve deployment defaults at call time. Import-time default arguments
    # otherwise retain a stale pool after activation or test/deployment rebinding.
    source_root = SOURCE_ROOT if source_root is None else source_root
    pool_manifest = POOL_MANIFEST if pool_manifest is None else pool_manifest
    pool_seal = POOL_SEAL if pool_seal is None else pool_seal
    candidates: list[dict] = []
    raven_prompt = RAVEN_RENDER_PROMPT.read_text(encoding="utf-8").strip()
    if not raven_prompt:
        raise ValueError(f"empty canonical Raven prompt: {RAVEN_RENDER_PROMPT}")
    source_root = source_root.resolve()
    if output_root.resolve() == source_root:
        raise ValueError("rendered output must not replace immutable Isaac sources")
    pool_manifest = pool_manifest.resolve()
    pool_seal = pool_seal.resolve()
    if not source_root.is_dir() or not pool_manifest.is_file() or not pool_seal.is_file():
        raise FileNotFoundError("missing unified sealed BC pool")
    manifest = json.loads(pool_manifest.read_text(encoding="utf-8"))
    seal = json.loads(pool_seal.read_text(encoding="utf-8"))
    continuous=manifest.get('schema')=='roborender.bc-pool.continuous.v1'
    if continuous:
        try:
            from .continuous_source import load_expected
        except ImportError:
            from continuous_source import load_expected
        items=load_expected(pool_manifest,pool_seal)
        if source_root!=pool_manifest.parent/'isaac':
            raise ValueError('continuous source directory differs from its bound pool')
    else:
        items = manifest.get("chunks", [])
    if not continuous and (
        manifest.get("status") != "sealed"
        or int(manifest.get("chunk_count", -1)) != len(items)
        or seal.get("verdict") != "PASS"
        or int(seal.get("isaac_chunks", -1)) != len(items)
    ):
        raise ValueError("unified BC pool manifest/seal contract failed")
    for item in items:
        source_npz = (source_root / Path(item["isaac_npz"]).name).resolve()
        try:
            source_npz.relative_to(source_root)
        except ValueError as error:
            raise ValueError(f"chunk escapes unified Isaac pool: {source_npz}") from error
        candidates.append(
            {
                "source_npz": source_npz,
                "manifest_path": pool_manifest,
                "seal_path": pool_seal,
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
        payload_sha256 = item["source_sha256"]
        if (not isinstance(payload_sha256, str) or len(payload_sha256) != 64
                or any(c not in "0123456789abcdef" for c in payload_sha256)):
            raise ValueError(f"invalid source hash in manifest: {source_npz}")
        if verify_source_hashes and sha256_file(source_npz) != payload_sha256:
            raise ValueError(f"source payload hash mismatch: {source_npz}")
        with np.load(source_npz, allow_pickle=False) as payload:
            frame_count = int(payload["gt_actions"].shape[0])
            local_env_id = int(payload["env_id"].item())
            has_done = (
                bool(payload["dones"].any()) if "dones" in payload.files else False
            )
        manifest_local_env_id = int(
            item.get("local_env_id", local_env_id)
        )
        if frame_count != 81 or local_env_id != manifest_local_env_id:
            raise ValueError(f"Raven chunk identity differs: {source_npz}")
        global_env_id = int(item.get("global_env_id", local_env_id))
        # Raven's NPZ schema has env_id and seq but no episode_id. Preserve an
        # optional staged lineage value when present; otherwise seq is the
        # non-scientific diagnostic identity.
        episode_id = item.get("episode_id", item["sequence_id"])
        provenance = collection_provenance(item)
        output_npz = (output_root / source_npz.name).resolve()
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
                "collection_provenance": provenance,
                "sequence_id": int(item["sequence_id"]),
                # Raven's exact tag is basename.split("_")[1].  The canonical
                # pool renamed files, so derive it from collection_filename;
                # source_run_tag is not equivalent for recovery runs.
                "stream_tag": raven_stream_tag(item),
                "source_run_tag": item.get("source_run_tag", raven_stream_tag(item)),
                "source_batch": int(item.get("source_batch", -1)),
                "rank": int(item.get("rank", 0)),
                "has_done": has_done,
                "frame_count": frame_count,
                # Raven's production three-view re-render overrides every
                # chunk prompt with render_prompt.txt and reuses seed 42.
                "prompt": candidate["prompt"],
                "seed": RAVEN_SEED,
                "output_npz": str(output_npz),
            }
        )
    return jobs


def build_render_chains(
    jobs: list[dict], max_chain: int | None = None, *, recipe: str = CURRENT_RECIPE
) -> list[dict]:
    """Default to Raven's three-view launcher: one independent 81-frame clip.

    The generic six-clip renderer is available only through an explicitly
    historical recipe. Both recipes preserve globally distinct source streams.
    """
    settings = recipe_config(recipe)
    max_chain = settings["max_chain_chunks"] if max_chain is None else max_chain
    if max_chain > settings["max_chain_chunks"]:
        raise ValueError("multi-chunk rendering requires the explicit legacy recipe")
    if max_chain < 1:
        raise ValueError("max_chain must be positive")
    streams: dict[str, list[dict]] = {}
    for job in jobs:
        key = json.dumps(source_stream_identity(job), sort_keys=True, separators=(",", ":"))
        streams.setdefault(key, []).append(job)

    chains: list[dict] = []
    for stream_key in sorted(streams):
        ordered = sorted(
            streams[stream_key],
            key=lambda job: (int(job["sequence_id"]), str(job["source_npz"])),
        )
        current: list[dict] = []
        previous_seq: int | None = None

        def emit() -> None:
            nonlocal current
            if not current:
                return
            identity = {
                "render_recipe": recipe,
                "source_stream": json.loads(stream_key),
                "chunks": [
                    {
                        "sequence_id": int(job["sequence_id"]),
                        "source": job.get("source_payload_sha256", job["source_npz"]),
                    }
                    for job in current
                ],
            }
            chain_id = "chain-" + hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:24]
            source_stream_id = "stream-" + hashlib.sha256(stream_key.encode()).hexdigest()
            jobs_with_context = []
            for index, job in enumerate(current):
                jobs_with_context.append(
                    {
                        **job,
                        **settings,
                        "source_stream_id": source_stream_id,
                        "chain_id": chain_id,
                        "chain_index": index,
                        "chain_length": len(current),
                        "inherits_previous_reference": index > 0,
                    }
                )
            chains.append(
                {
                    "chain_id": chain_id,
                    **settings,
                    "source_stream_id": source_stream_id,
                    "stream_tag": str(current[0]["stream_tag"]),
                    "env_id": int(current[0]["env_id"]),
                    "global_env_id": int(current[0]["global_env_id"]),
                    "episode_id": current[0]["episode_id"],
                    "frame_count": sum(job["frame_count"] for job in current),
                    "jobs": jobs_with_context,
                }
            )
            current = []

        for job in ordered:
            sequence_id = int(job["sequence_id"])
            discontinuous = previous_seq is not None and sequence_id != previous_seq + 1
            if current and (discontinuous or len(current) >= max_chain):
                emit()
            current.append(job)
            previous_seq = sequence_id
            if bool(job["has_done"]):
                emit()
                previous_seq = None
        emit()
    return chains


def incompatible_existing_outputs(chains: list[dict]) -> list[str]:
    """Return outputs that cannot be resumed under the current chain identity."""
    incompatible: list[str] = []
    for chain in chains:
        for job in chain["jobs"]:
            path = Path(job["output_npz"])
            if not path.exists():
                continue
            try:
                with np.load(path, allow_pickle=False) as payload:
                    matches = all(
                        payload[key].item() == value
                        for key, value in context_metadata(job).items()
                    )
            except (KeyError, OSError, ValueError):
                matches = False
            if not matches:
                incompatible.append(str(path))
    return incompatible


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
            str(item["episode_id"]),
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
    parser.add_argument("--recipe", choices=RECIPE_NAMES, default=CURRENT_RECIPE)
    parser.add_argument("--verify-source-hashes", action="store_true",
                        help="also hash every source during planning; workers verify before rendering")
    parser.add_argument("--launch-stagger-seconds", type=float, default=0.25)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--execution-identity", type=Path)
    parser.add_argument("--execution-identity-sha256")
    parser.add_argument("--asset-root", type=Path, default=PROJECT_ROOT,
                        help="shared heavy-asset root when executing an immutable source release")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=SOURCE_ROOT,
        help="immutable Isaac source directory (may be an archived generation)",
    )
    parser.add_argument("--pool-manifest", type=Path, default=POOL_MANIFEST)
    parser.add_argument("--pool-seal", type=Path, default=POOL_SEAL)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="destination for corrected archives; use a fresh root for smoke/re-render",
    )
    args = parser.parse_args()
    if (args.execution_identity is None) != (args.execution_identity_sha256 is None):
        raise ValueError("renderer execution identity requires file and raw SHA256 together")

    gpus = [int(value) for value in args.gpus.split(",")]
    if len(gpus) != len(set(gpus)) or not gpus:
        raise ValueError("--gpus must contain unique GPU indices")
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be positive")

    asset_root = args.asset_root.resolve(strict=True)
    adapter = asset_root / (
        "renderer/offline_teacher/checkpoints/"
        "roborender_offline_bc_video_teacher_50step_lora_step05294.safetensors"
    )
    foundation = asset_root / "renderer/models/wan21_fun_control_1p3b_foundation"
    source = asset_root / "renderer/offline_teacher/source"
    worker = Path(__file__).with_name("render_bc_teacher_worker.py")
    python = asset_root / "renderer/envs/image3f-py313-cu130-trt10133/venv/bin/python"
    for required in (adapter, foundation, source, worker, python, RAVEN_RENDER_PROMPT):
        if not required.exists():
            raise FileNotFoundError(required)
    if sha256_file(adapter) != ADAPTER_SHA256:
        raise RuntimeError("50-step teacher adapter hash mismatch")

    jobs = discover_jobs(
        limit=args.limit,
        selection_seed=args.selection_seed,
        output_root=args.output_root.resolve(),
        source_root=args.source_root,
        pool_manifest=args.pool_manifest,
        pool_seal=args.pool_seal,
        verify_source_hashes=args.verify_source_hashes,
    )
    settings = recipe_config(args.recipe)
    chains = build_render_chains(jobs, recipe=args.recipe)
    incompatible = incompatible_existing_outputs(chains)
    if incompatible:
        raise FileExistsError(
            f"{len(incompatible)} outputs conflict with {args.recipe}; use a fresh --output-root: "
            + ", ".join(incompatible[:3])
        )
    gpu_assignments, gpu_frame_totals = assign_lpt(chains, len(gpus))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir
        else PROJECT_ROOT / f"runtime/outputs/bc_rendering/{timestamp}"
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"preserve existing plan artifacts; select a new --run-dir: {run_dir}")
    if (args.limit is not None or args.selection_seed is not None) and (
        args.output_root.resolve() == OUTPUT_ROOT.resolve()
    ):
        raise ValueError(
            "smoke selection must use --output-root outside the production rendered pool"
        )
    config = {
        **settings,
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
    if args.execution_identity is not None:
        sys.path.insert(0, str(PROJECT_ROOT))
        from pipeline.bc_training.render_execution import verify_worker
        config["execution_identity"] = {"path": str(args.execution_identity.resolve()),
                                         "raw_sha256": args.execution_identity_sha256}
        verify_worker(config, worker)
    lane_plans: list[dict] = []
    for gpu_slot, physical_gpu in enumerate(gpus):
        lane_assignments, lane_totals = assign_lpt(
            gpu_assignments[gpu_slot], args.workers_per_gpu
        )
        for lane, lane_chains in enumerate(lane_assignments):
            lane_jobs = [job for chain in lane_chains for job in chain["jobs"]]
            lane_plan = {
                "schema": "roborender.bc_teacher_lane_plan.v3",
                "physical_gpu": physical_gpu,
                "lane": lane,
                "frame_count": lane_totals[lane],
                "config": config,
                "chains": lane_chains,
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
        "schema": "roborender.bc_teacher_8gpu_plan.v3",
        **settings,
        "created_utc": timestamp,
        "status": "planned" if args.plan_only else "launching",
        "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "total_workers": len(lane_plans),
        "episode_count": len(jobs),
        "chain_count": len(chains),
        "max_chain_chunks": settings["max_chain_chunks"],
        "frame_count": sum(job["frame_count"] for job in jobs),
        "source_root": str(args.source_root.resolve()),
        "source_pool": "unified",
        "output_root": str(args.output_root.resolve()),
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

    incompatible = incompatible_existing_outputs(chains)
    if incompatible:
        preview = ", ".join(incompatible[:3])
        raise FileExistsError(
            f"{len(incompatible)} existing outputs lack the active render "
            f"recipe/chain identity ({preview}); use a fresh --output-root"
        )

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
                "DIFFSYNTH_MODEL_BASE_PATH": str((asset_root / "renderer/models").resolve()),
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
