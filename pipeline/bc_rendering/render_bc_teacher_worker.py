#!/usr/bin/env python3
"""Persistent 50-step teacher worker for one GPU lane."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def join_three_views(frames: tuple[Image.Image, Image.Image, Image.Image]) -> Image.Image:
    width = sum(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    output = Image.new("RGB", (width, height))
    x = 0
    for frame in frames:
        output.paste(frame, (x, 0))
        x += frame.width
    return output


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
    from diffsynth.utils.data import save_video
    from sim2realgen.inference_long_video import split_video_vertical

    source_npz = Path(job["source_npz"]).resolve()
    output_npz = Path(job["output_npz"]).resolve()
    output_dir = Path(job["output_dir"]).resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")

    if validate_completed_chunk(output_npz, job["frame_count"]):
        return {"status": "skipped_complete", "output_npz": str(output_npz)}
    if output_npz.exists():
        raise FileExistsError(f"refusing to overwrite invalid output: {output_npz}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite incomplete output: {output_dir}")
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)

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

    video_paths = {
        camera: partial_dir / f"rgb_gen__{camera}.mp4" for camera in by_camera
    }
    for camera, frames in by_camera.items():
        save_video(frames, str(video_paths[camera]), fps=config["preview_fps"], quality=5)

    primary_path = partial_dir / "rgb_gen.mp4"
    os.link(video_paths["head_color"], primary_path)
    combined_path = partial_dir / "rendered_3view.mp4"
    combined_frames = [
        join_three_views((left, right, head))
        for left, right, head in zip(
            by_camera["hand_left"],
            by_camera["hand_right"],
            by_camera["head_color"],
            strict=True,
        )
    ]
    save_video(combined_frames, str(combined_path), fps=config["preview_fps"], quality=5)

    output_files = [
        output_npz,
        primary_path,
        video_paths["head_color"],
        video_paths["hand_left"],
        video_paths["hand_right"],
        combined_path,
    ]
    output_hashes = {path.name: sha256_file(path) for path in output_files}
    manifest = {
        "schema": "roborender.bc_teacher_render.v1",
        "status": "complete",
        "global_env_id": job["global_env_id"],
        "env_id": job["env_id"],
        "episode_id": job["episode_id"],
        "source_npz": str(source_npz),
        "source_payload_sha256": job["source_payload_sha256"],
        "frame_count": job["frame_count"],
        "model_frame_count": model_frame_count,
        "prompt": job["prompt"],
        "seed": job["seed"],
        "renderer": {
            "role": "offline_bc_50step_teacher",
            "foundation": "PAI/Wan2.1-Fun-V1.1-1.3B-Control",
            "adapter_path": config["adapter_path"],
            "adapter_sha256": config["adapter_sha256"],
            "num_inference_steps": config["num_inference_steps"],
            "cfg_scale": config["cfg_scale"],
            "view_mapping": {slot: camera for slot, camera in VIEW_SLOTS},
        },
        "learner_archive": str(output_npz),
        "learner_archive_sha256": sha256_file(output_npz),
        "rgb_gen_shape": [job["frame_count"], config["student_height"], config["student_width"], 3],
        "rgb_gen_dtype": "uint8",
        "alignment": "rgb_gen[t] -> obs_*[t] -> gt_actions[t]",
        "state_dim": 113,
        "action_dim": 26,
        "wall_seconds": time.monotonic() - started,
        "outputs": output_hashes,
    }
    manifest_path = partial_dir / "render_manifest.json"
    write_json(manifest_path, manifest)
    checksum_paths = output_files + [manifest_path]
    checksum_lines = [f"{sha256_file(path)}  {path.name}" for path in checksum_paths]
    (partial_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial_dir.replace(output_dir)
    del generated, split, by_camera, combined_frames, depth_views, mask_views
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "status": "complete",
        "output_dir": str(output_dir),
        "output_npz": str(output_npz),
        "wall_seconds": manifest["wall_seconds"],
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

    print(
        f"loading teacher for physical_gpu={plan['physical_gpu']} lane={plan['lane']} "
        f"jobs={len(plan['jobs'])}",
        flush=True,
    )
    pipe = load_pipeline(config)
    for index, job in enumerate(plan["jobs"], start=1):
        print(
            f"job {index}/{len(plan['jobs'])}: "
            f"env={job['global_env_id']} episode={job['episode_id']} "
            f"frames={job['frame_count']}",
            flush=True,
        )
        result = render_episode(pipe, job, config)
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
