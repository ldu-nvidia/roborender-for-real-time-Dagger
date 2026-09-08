#!/usr/bin/env python3
"""Persistent 50-step teacher worker for one GPU lane."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


VIEW_SLOTS = (
    ("ext1", "hand_left"),
    ("ext2", "hand_right"),
    ("wrist", "head_color"),
)
RAVEN_STUDENT_RESOLUTION = (108, 192)
RAVEN_PRIMARY_VIEW = "head_color"


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

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_npz.with_suffix(output_npz.suffix + ".partial")
    if output_npz.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {output_npz}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_npz)


def validate_completed_chunk(path: Path, frame_count: int) -> bool:
    if not path.is_file():
        return False
    expected = (frame_count, *RAVEN_STUDENT_RESOLUTION, 3)
    try:
        with np.load(path, allow_pickle=False) as payload:
            return (
                payload["rgb_gen"].shape == expected
                and payload["rgb_gen"].dtype == np.uint8
                and all(
                    payload[f"rgb_gen__{view}"].shape == expected
                    and payload[f"rgb_gen__{view}"].dtype == np.uint8
                    for _, view in VIEW_SLOTS
                )
            )
    except (KeyError, OSError, ValueError):
        return False


def load_pipeline(config: dict):
    from diffsynth.core.loader import load_state_dict
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    from sim2realgen.inference_long_video import setup_multiview_pipeline

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
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
        ],
        tokenizer_config=ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        ),
        redirect_common_files=False,
    )
    state_dict = load_state_dict(
        config["adapter_path"], torch_dtype=pipe.torch_dtype, device=pipe.device
    )
    pipe.load_lora(pipe.dit, state_dict=state_dict, alpha=1.0)
    setup_multiview_pipeline(pipe)
    return pipe


def render_episode(pipe, job: dict, config: dict) -> dict:
    from sim2realgen.inference_long_video import split_video_vertical

    source_npz = Path(job["source_npz"]).resolve()
    output_npz = Path(job["output_npz"]).resolve()
    if validate_completed_chunk(output_npz, job["frame_count"]):
        return {"status": "skipped_complete", "output_npz": str(output_npz)}
    if output_npz.exists():
        raise FileExistsError(f"refusing to overwrite invalid output: {output_npz}")

    started = time.monotonic()
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
        rows = np.arange(payload["gt_actions"].shape[0], dtype=np.int64)
        if int(rows.size) != job["frame_count"]:
            raise ValueError(
                f"planned {job['frame_count']} frames but source contains {rows.size}"
            )

        depth_views: list[list[Image.Image]] = []
        mask_views: list[list[Image.Image]] = []
        for _, camera in VIEW_SLOTS:
            depth_views.append(
                [pil_depth(frame) for frame in payload[f"depth_ctrl__{camera}"][rows]]
            )
            mask_views.append(
                [pil_rgb(frame) for frame in payload[f"rgb_robot__{camera}"][rows]]
            )

    model_frame_count = next_wan_frame_count(job["frame_count"])
    depth_views = [pad_frames(frames, model_frame_count) for frames in depth_views]
    mask_views = [pad_frames(frames, model_frame_count) for frames in mask_views]

    generated = pipe(
        prompt=job["prompt"],
        negative_prompt="",
        control_video_views=depth_views,
        control_video_2_views=mask_views,
        reference_image_views=None,
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
    split = [frames[: job["frame_count"]] for frames in split]
    by_camera = {
        "hand_left": split[0],
        "hand_right": split[1],
        "head_color": split[2],
    }

    write_raven_rendered_chunk(
        source_npz,
        output_npz,
        by_camera,
        student_height=config["student_height"],
        student_width=config["student_width"],
    )

    wall_seconds = time.monotonic() - started
    del generated, split, by_camera, depth_views, mask_views
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "status": "complete",
        "output_npz": str(output_npz),
        "wall_seconds": wall_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-file", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.job_file.read_text(encoding="utf-8"))
    config = plan["config"]

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
        for index, job in enumerate(plan["jobs"], start=1):
            current_job = {
                "index": index,
                "global_env_id": job["global_env_id"],
                "sequence_id": job["sequence_id"],
                "source_npz": job["source_npz"],
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
                f"job {index}/{len(plan['jobs'])}: "
                f"env={job['global_env_id']} episode={job['episode_id']} "
                f"frames={job['frame_count']}",
                flush=True,
            )
            result = render_episode(pipe, job, config)
            if result["status"] == "complete":
                rendered_this_run += 1
            elif result["status"] == "skipped_complete":
                skipped_preexisting += 1
            else:
                raise RuntimeError(f"unexpected render result: {result}")
            done += 1
            write_lane_progress(
                plan,
                status="rendering",
                done=done,
                rendered_this_run=rendered_this_run,
                skipped_preexisting=skipped_preexisting,
            )
            print(json.dumps(result, sort_keys=True), flush=True)
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
