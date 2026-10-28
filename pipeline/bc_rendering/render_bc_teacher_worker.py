#!/usr/bin/env python3
"""Persistent 50-step teacher worker for one GPU lane."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from .render_recipes import (
        CURRENT_RECIPE, INDEPENDENT_CONTRACT, context_metadata, recipe_config, validate_context,
    )
except ImportError:
    from render_recipes import (
        CURRENT_RECIPE, INDEPENDENT_CONTRACT, context_metadata, recipe_config, validate_context,
    )

VIEW_SLOTS = (
    ("ext1", "hand_left"),
    ("ext2", "hand_right"),
    ("wrist", "head_color"),
)
RAVEN_STUDENT_RESOLUTION = (108, 192)
RAVEN_PRIMARY_VIEW = "head_color"
RAVEN_CHAIN_RENDER_CONTRACT = INDEPENDENT_CONTRACT


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


def write_lane_progress(
    plan: dict,
    *,
    status: str,
    done: int,
    rendered_this_run: int,
    skipped_preexisting: int,
    current_job: dict | None = None,
    error: str | None = None,
) -> None:
    value = {
        "schema": "roborender.bc_teacher_lane_progress.v1",
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "physical_gpu": int(plan["physical_gpu"]),
        "lane": int(plan["lane"]),
        "status": status,
        "assigned": len(plan["jobs"]),
        "done": done,
        "rendered_this_run": rendered_this_run,
        "skipped_preexisting": skipped_preexisting,
        "remaining": len(plan["jobs"]) - done,
        "current_job": current_job,
        "error": error,
    }
    write_json(Path(plan["progress_path"]), value)


def next_wan_frame_count(frame_count: int) -> int:
    """Wan video lengths are 4k+1; pad only the model input, then trim output."""
    if frame_count < 1:
        raise ValueError("an episode must contain at least one rendered frame")
    return 1 + 4 * ((frame_count - 1 + 3) // 4)


def pil_depth(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array, mode="L").convert("RGB")


def pil_rgb(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array, mode="RGB")


def pad_frames(frames: list[Image.Image], length: int) -> list[Image.Image]:
    if len(frames) > length:
        raise ValueError("cannot pad to a length shorter than the source")
    return frames + [frames[-1].copy() for _ in range(length - len(frames))]


def downsample_raven_rgb(
    frames: list[Image.Image], *, height: int = 108, width: int = 192
) -> np.ndarray:
    """Match Raven's PIL bilinear conversion to [T,108,192,3] uint8."""
    return np.stack(
        [
            np.asarray(frame.convert("RGB").resize((width, height), Image.Resampling.BILINEAR))
            for frame in frames
        ]
    ).astype(np.uint8, copy=False)


def write_raven_rendered_chunk(
    source_npz: Path,
    output_npz: Path,
    by_camera: dict[str, list[Image.Image]],
    *,
    student_height: int = 108,
    student_width: int = 192,
    chain_context: dict | None = None,
) -> None:
    """Publish the exact archive contract consumed by Raven's offline loader."""
    with np.load(source_npz, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    rendered = {
        camera: downsample_raven_rgb(
            by_camera[camera], height=student_height, width=student_width
        )
        for camera in ("hand_left", "hand_right", "head_color")
    }
    frame_count = int(arrays["gt_actions"].shape[0])
    expected = (frame_count, student_height, student_width, 3)
    for camera, value in rendered.items():
        if value.shape != expected or value.dtype != np.uint8:
            raise ValueError(
                f"rgb_gen__{camera} must be {expected} uint8, got "
                f"{value.shape} {value.dtype}"
            )
        arrays[f"rgb_gen__{camera}"] = value
    arrays["rgb_gen"] = rendered[RAVEN_PRIMARY_VIEW].copy()
    if chain_context is not None:
        arrays.update({key: np.asarray(value) for key, value in context_metadata(chain_context).items()})

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_npz.with_suffix(output_npz.suffix + ".partial")
    if output_npz.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {output_npz}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_npz)


def validate_completed_chunk(
    path: Path, frame_count: int, *, chain_context: dict | None = None
) -> bool:
    if not path.is_file():
        return False
    expected = (frame_count, *RAVEN_STUDENT_RESOLUTION, 3)
    try:
        with np.load(path, allow_pickle=False) as payload:
            valid = (
                payload["rgb_gen"].shape == expected
                and payload["rgb_gen"].dtype == np.uint8
                and all(
                    payload[f"rgb_gen__{view}"].shape == expected
                    and payload[f"rgb_gen__{view}"].dtype == np.uint8
                    for _, view in VIEW_SLOTS
                )
            )
            if not valid or chain_context is None:
                return valid
            return all(
                payload[key].item() == value
                for key, value in context_metadata(chain_context).items()
            )
    except (KeyError, OSError, ValueError):
        return False


def load_pipeline(config: dict):
    from diffsynth.core.loader import load_state_dict
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    from sim2realgen.inference_long_video import setup_multiview_pipeline

    model_configs = [
            ModelConfig(
                model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control",
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
            ),
            ModelConfig(
                model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            ),
            ModelConfig(
                model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control",
                origin_file_pattern="Wan2.1_VAE.pth",
            ),
            ModelConfig(
                model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control",
                origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ),
        ]
    tokenizer_config = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        )
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        redirect_common_files=False,
    )
    state_dict = load_state_dict(
        config["adapter_path"], torch_dtype=pipe.torch_dtype, device=pipe.device
    )
    pipe.load_lora(pipe.dit, state_dict=state_dict, alpha=1.0)
    setup_multiview_pipeline(pipe)
    if config.get("execution_identity"):
        from pipeline.bc_training.render_execution import verify_worker
        loaded = [path for model in model_configs for path in ([model.path] if isinstance(model.path, str) else model.path)]
        pipe._roborender_execution_identity = verify_worker(config, __file__, loaded_paths=loaded,
            loaded_modules={name: sys.modules[name].__file__ for name in ("diffsynth", "sim2realgen.inference_long_video")},
            tokenizer_path=tokenizer_config.path)
    return pipe


def split_video_vertical(
    frames: list[Image.Image], num_views: int
) -> list[list[Image.Image]]:
    if not frames:
        raise ValueError("renderer returned no frames")
    if any(frame.height % num_views for frame in frames):
        raise ValueError("stacked renderer frame height is not divisible by num_views")
    view_height = frames[0].height // num_views
    return [
        [
            frame.crop((0, view * view_height, frame.width, (view + 1) * view_height))
            for frame in frames
        ]
        for view in range(num_views)
    ]


def render_chain(pipe, chain: dict, config: dict) -> dict:
    """Render the selected recipe; current three-view clips have no prior reference.

    If a run resumes after a completed prefix, the prefix is rendered again in
    memory to reconstruct its full-resolution final frames.  Existing outputs
    are never overwritten.  A fully completed chain makes no model calls.
    """
    jobs = chain["jobs"]
    settings = recipe_config(config.get("render_recipe", CURRENT_RECIPE))
    if not jobs or len(jobs) > settings["max_chain_chunks"]:
        raise ValueError("chain length violates the explicit render recipe")
    for index, job in enumerate(jobs):
        validate_context(job, recipe=settings["render_recipe"])
        context_metadata(job)
        if (int(job["chain_index"]) != index or int(job["chain_length"]) != len(jobs)
                or job["chain_id"] != chain["chain_id"]):
            raise ValueError("job ordering/chain identity differs from the selected chain")
    completion = [
        validate_completed_chunk(
            Path(job["output_npz"]), job["frame_count"], chain_context=job
        )
        for job in jobs
    ]
    invalid_existing = [
        job["output_npz"]
        for job, complete in zip(jobs, completion)
        if Path(job["output_npz"]).exists() and not complete
    ]
    if invalid_existing:
        raise FileExistsError(
            "refusing to overwrite output lacking the active chain contract: "
            + ", ".join(invalid_existing)
        )
    if all(completion):
        return {
            "status": "skipped_complete",
            "chain_id": chain["chain_id"],
            "completed": len(jobs),
            "rendered": 0,
            "replayed_prefix": 0,
        }

    first_missing = completion.index(False)
    if any(completion[first_missing + 1 :]):
        raise RuntimeError("completed outputs must form a contiguous chain prefix")

    started = time.monotonic()
    previous_frame_views: list[Image.Image] | None = None
    rendered = 0
    for job_index, job in enumerate(jobs):
        source_npz = Path(job["source_npz"]).resolve()
        output_npz = Path(job["output_npz"]).resolve()
        if source_npz == output_npz:
            raise ValueError("renderer cannot overwrite its Isaac source")
        if job.get("source_payload_sha256") and sha256_file(source_npz) != job["source_payload_sha256"]:
            raise ValueError(f"source payload hash mismatch before rendering: {source_npz}")
        with np.load(source_npz, allow_pickle=False) as payload:
            required = {
                "gt_actions",
                "obs_actor_obs",
                *(f"depth_ctrl__{camera}" for _, camera in VIEW_SLOTS),
                *(f"rgb_robot__{camera}" for _, camera in VIEW_SLOTS),
            }
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"Raven BC chunk lacks required fields: {missing}")
            frame_count = int(payload["gt_actions"].shape[0])
            if frame_count != job["frame_count"]:
                raise ValueError(
                    f"planned {job['frame_count']} frames but source contains {frame_count}"
                )
            depth_views = [
                [pil_depth(frame) for frame in payload[f"depth_ctrl__{camera}"]]
                for _, camera in VIEW_SLOTS
            ]
            mask_views = [
                [pil_rgb(frame) for frame in payload[f"rgb_robot__{camera}"]]
                for _, camera in VIEW_SLOTS
            ]

        model_frame_count = next_wan_frame_count(frame_count)
        depth_views = [pad_frames(frames, model_frame_count) for frames in depth_views]
        mask_views = [pad_frames(frames, model_frame_count) for frames in mask_views]
        generated = pipe(
            prompt=job["prompt"],
            negative_prompt="",
            control_video_views=depth_views,
            control_video_2_views=mask_views,
            reference_image_views=previous_frame_views,
            height=config["height"] * 3,
            width=config["width"],
            num_frames=model_frame_count,
            num_views=3,
            seed=job["seed"],
            cfg_scale=config["cfg_scale"],
            num_inference_steps=config["num_inference_steps"],
            tiled=True,
        )
        split = split_video_vertical(generated, 3)
        split = [frames[:frame_count] for frames in split]
        previous_frame_views = [frames[-1].copy() for frames in split]
        by_camera = {
            "hand_left": split[0],
            "hand_right": split[1],
            "head_color": split[2],
        }
        if not completion[job_index]:
            write_raven_rendered_chunk(
                source_npz,
                output_npz,
                by_camera,
                student_height=config["student_height"],
                student_width=config["student_width"],
                chain_context=job,
            )
            rendered += 1
        del generated, split, by_camera, depth_views, mask_views

    gc.collect()
    torch.cuda.empty_cache()
    return {
        "status": "complete",
        "chain_id": chain["chain_id"],
        "completed": len(jobs),
        "rendered": rendered,
        "replayed_prefix": first_missing,
        "wall_seconds": time.monotonic() - started,
    }


def render_episode(pipe, job: dict, config: dict) -> dict:
    """Backward-compatible singleton wrapper; production uses render_chain."""
    job = {
        **job,
        **recipe_config(config.get("render_recipe", CURRENT_RECIPE)),
        "source_stream_id": job.get("source_stream_id", "singleton-" + hashlib.sha256(
            str(Path(job["source_npz"]).resolve()).encode()
        ).hexdigest()),
        "chain_id": job.get("chain_id", "singleton"),
        "chain_index": 0,
        "chain_length": 1,
        "inherits_previous_reference": False,
    }
    return render_chain(
        pipe,
        {"chain_id": job["chain_id"], "jobs": [job]},
        config,
    )


def validate_lane_plan(plan: dict) -> dict:
    """Reject ambiguous/historical/tampered plans before initializing CUDA."""
    if plan.get("schema") != "roborender.bc_teacher_lane_plan.v3":
        raise ValueError("worker requires a versioned v3 lane plan; re-plan historical inputs")
    config = plan["config"]
    if config.get("execution_identity"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from pipeline.bc_training.render_execution import verify_worker
        verify_worker(config, __file__)
    settings = recipe_config(config["render_recipe"])
    if any(config.get(key) != value for key, value in settings.items()):
        raise ValueError("worker plan does not declare the complete selected render recipe")
    scientific = {"num_inference_steps": 50, "cfg_scale": 5.0, "height": 240,
                  "width": 416, "student_height": 108, "student_width": 192, "seed": 42}
    if any(config.get(key) != value for key, value in scientific.items()):
        raise ValueError("worker plan differs from Raven's pinned three-view rendering parameters")
    chains = plan.get("chains")
    if not isinstance(chains, list) or not chains:
        raise ValueError("worker requires explicit nonempty render chains")
    flattened = [job for chain in chains for job in chain["jobs"]]
    if flattened != plan.get("jobs"):
        raise ValueError("lane job list differs from its ordered render chains")
    for chain in chains:
        if not 1 <= len(chain["jobs"]) <= settings["max_chain_chunks"]:
            raise ValueError("worker chain length differs from the selected recipe")
        for index, job in enumerate(chain["jobs"]):
            validate_context(job, recipe=settings["render_recipe"])
            context_metadata(job)
            if (job["chain_id"] != chain["chain_id"] or job["chain_index"] != index
                    or job["chain_length"] != len(chain["jobs"])):
                raise ValueError("worker chain ordering/identity is inconsistent")
            if job.get("seed") != 42 or job.get("frame_count") != 81:
                raise ValueError("worker job is not a seed-42 81-frame clip")
            if config.get("execution_identity") and job.get("prompt") != Path(config["prompt_file"]).read_text().strip():
                raise ValueError("worker prompt differs from the attested prompt asset")
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-file", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.job_file.read_text(encoding="utf-8"))
    config = plan["config"]
    # Historical plans without an explicit recipe must not silently acquire a
    # new contract or relabel old autoregressive outputs as independent clips.
    validate_lane_plan(plan)

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker must see exactly one CUDA device")
    if "H200" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"teacher production requires H200, got {torch.cuda.get_device_name(0)}")
    if sha256_file(Path(config["adapter_path"])) != config["adapter_sha256"]:
        raise RuntimeError("teacher adapter hash mismatch")

    done = 0
    rendered_this_run = 0
    skipped_preexisting = 0
    current_job = None
    write_lane_progress(
        plan,
        status="loading",
        done=done,
        rendered_this_run=rendered_this_run,
        skipped_preexisting=skipped_preexisting,
    )
    print(
        f"loading teacher for physical_gpu={plan['physical_gpu']} lane={plan['lane']} "
        f"jobs={len(plan['jobs'])}",
        flush=True,
    )
    try:
        pipe = load_pipeline(config)
        chains = plan.get("chains")
        if chains is None:
            chains = [
                {
                    "chain_id": job.get("chain_id", f"singleton-{index:06d}"),
                    "jobs": [job],
                }
                for index, job in enumerate(plan["jobs"])
            ]
        for index, chain in enumerate(chains, start=1):
            first_job = chain["jobs"][0]
            current_job = {
                "index": index,
                "chain_id": chain["chain_id"],
                "chain_chunks": len(chain["jobs"]),
                "global_env_id": first_job["global_env_id"],
                "first_sequence_id": first_job["sequence_id"],
                "source_npz": first_job["source_npz"],
            }
            write_lane_progress(
                plan,
                status="rendering",
                done=done,
                rendered_this_run=rendered_this_run,
                skipped_preexisting=skipped_preexisting,
                current_job=current_job,
            )
            print(
                f"chain {index}/{len(chains)}: id={chain['chain_id']} "
                f"env={first_job['global_env_id']} chunks={len(chain['jobs'])} "
                f"frames={sum(job['frame_count'] for job in chain['jobs'])}",
                flush=True,
            )
            result = render_chain(pipe, chain, config)
            chain_size = len(chain["jobs"])
            rendered_this_run += int(result["rendered"])
            skipped_preexisting += chain_size - int(result["rendered"])
            done += chain_size
            if result["status"] not in {"complete", "skipped_complete"}:
                raise RuntimeError(f"unexpected render result: {result}")
            write_lane_progress(
                plan,
                status="rendering",
                done=done,
                rendered_this_run=rendered_this_run,
                skipped_preexisting=skipped_preexisting,
            )
            print(json.dumps(result, sort_keys=True), flush=True)
        if config.get("execution_identity"):
            from pipeline.bc_training.render_execution import complete_lane
            evidence = complete_lane(plan, pipe._roborender_execution_identity,
                rendered_count=rendered_this_run, skipped_count=skipped_preexisting)
            write_json(Path(plan["progress_path"]).with_suffix(".execution.json"), evidence)
        write_lane_progress(
            plan,
            status="complete",
            done=done,
            rendered_this_run=rendered_this_run,
            skipped_preexisting=skipped_preexisting,
        )
    except Exception as error:
        write_lane_progress(
            plan,
            status="failed",
            done=done,
            rendered_this_run=rendered_this_run,
            skipped_preexisting=skipped_preexisting,
            current_job=current_job,
            error=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()
