#!/usr/bin/env python3
"""Materialize Raven-compatible 81-frame offline-BC chunks.

This compatibility projection converts a previously sealed native H200 actor
stream into Raven's flat chunk schema. The production Raven collector writes
that schema directly; no train/validation/test split is part of this path. The
projection replays Raven's collector buffer semantics exactly:

* append one dense policy-step row at a time;
* emit after 81 consecutive rows from the same episode;
* emit before handling done, as Raven's collector does;
* discard an incomplete tail at done or at the end of the source stream;
* store native quantized depth, metric depth, depth control, textured robot
  RGB, non-vision observations, teacher actions, an all-false dones vector,
  prompt, env_id, and seq.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


CHUNK_T = 81
# This is Raven's collection insertion order at GR00T commit 94d7985. Named
# keys make the archive unambiguous, but preserving order catches wiring drift.
VIEWS = ("hand_left", "hand_right", "head_color")


class RavenChunkError(RuntimeError):
    """A source stream cannot be projected into Raven's chunk contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RavenChunkError(f"{path} must contain a JSON object")
    return value


def image_rows_for_transition(payload: dict[str, np.ndarray]) -> np.ndarray:
    mapping = payload["transition_image_indices"]
    if mapping.ndim != 2 or mapping.shape[1] != 1:
        raise RavenChunkError("each materialization source must contain one environment")
    rows = mapping[:, 0].astype(np.int64, copy=False)
    if not np.array_equal(rows, np.arange(rows.size, dtype=np.int64)):
        raise RavenChunkError(
            "Raven BC requires one unique conditioning image per policy transition"
        )
    return rows


def source_observations(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    observations = {}
    for key, value in payload.items():
        if not key.startswith("obs__"):
            continue
        raven_key = "obs_" + key[5:]
        if value.ndim < 2 or value.shape[1] != 1:
            raise RavenChunkError(f"{key} must have one environment axis")
        observations[raven_key] = np.ascontiguousarray(value[:, 0], dtype=np.float32)
    if "obs_actor_obs" not in observations:
        raise RavenChunkError("source stream lacks obs__actor_obs")
    return observations


def validate_source(payload: dict[str, np.ndarray]) -> tuple[int, int]:
    required = {
        "expert_actions",
        "behavior_actions",
        "expert_mask",
        "dones",
        "global_env_ids",
        "episode_ids",
        "episode_steps",
        "transition_image_indices",
        *(f"depth_rawq__{view}" for view in VIEWS),
        *(f"depth_raw__{view}" for view in VIEWS),
        *(f"depth_ctrl__{view}" for view in VIEWS),
        *(f"rgb_robot__{view}" for view in VIEWS),
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RavenChunkError(f"source stream lacks fields: {missing}")
    expert = payload["expert_actions"]
    if expert.ndim != 3 or expert.shape[1] != 1:
        raise RavenChunkError("expert_actions must have shape [T,1,A]")
    steps = int(expert.shape[0])
    if steps <= 0:
        raise RavenChunkError("source stream is empty")
    for key in ("behavior_actions",):
        if payload[key].shape != expert.shape:
            raise RavenChunkError(f"{key} shape differs from expert_actions")
    if not np.array_equal(payload["behavior_actions"], expert):
        raise RavenChunkError("BC behavior_actions must exactly equal expert_actions")
    mask = payload["expert_mask"]
    if mask.shape != (steps, 1) or not bool(mask.all()):
        raise RavenChunkError("BC expert_mask must be true for every transition")
    for key in ("dones", "global_env_ids", "episode_ids", "episode_steps"):
        if payload[key].shape != (steps, 1):
            raise RavenChunkError(f"{key} must have shape [T,1]")
    global_ids = payload["global_env_ids"][:, 0]
    if not np.all(global_ids == global_ids[0]):
        raise RavenChunkError("global environment ID changed within a source stream")
    if not np.isfinite(expert).all():
        raise RavenChunkError("expert actions contain non-finite values")
    rows = image_rows_for_transition(payload)
    for view in VIEWS:
        raw_quantized = payload[f"depth_rawq__{view}"]
        raw_metric = payload[f"depth_raw__{view}"]
        depth = payload[f"depth_ctrl__{view}"]
        robot = payload[f"rgb_robot__{view}"]
        if raw_quantized.shape != (steps, 480, 832) or raw_quantized.dtype != np.uint16:
            raise RavenChunkError(
                f"depth_rawq__{view} differs from [T,480,832] uint16"
            )
        if raw_metric.shape != (steps, 240, 416) or raw_metric.dtype != np.float16:
            raise RavenChunkError(
                f"depth_raw__{view} differs from [T,240,416] float16"
            )
        if depth.shape != (steps, 240, 416) or depth.dtype != np.uint8:
            raise RavenChunkError(f"depth_ctrl__{view} differs from [T,240,416] uint8")
        if robot.shape != (steps, 240, 416, 3) or robot.dtype != np.uint8:
            raise RavenChunkError(f"rgb_robot__{view} differs from [T,240,416,3] uint8")
        if rows.max(initial=-1) >= depth.shape[0]:
            raise RavenChunkError("transition image mapping exceeds raster rows")
    return steps, int(global_ids[0])


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise RavenChunkError(f"refusing to overwrite {path}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise RavenChunkError(f"refusing to overwrite {path}")
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def materialize_environment(
    *,
    source_path: Path,
    prompt: str,
    output_dir: Path,
    chunk_index: int,
) -> tuple[list[dict], int, int]:
    with np.load(source_path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    steps, global_env_id = validate_source(payload)
    observations = source_observations(payload)
    episode_ids = payload["episode_ids"][:, 0].astype(np.int64, copy=False)
    episode_steps = payload["episode_steps"][:, 0].astype(np.int64, copy=False)
    dones = payload["dones"][:, 0].astype(np.bool_, copy=False)
    rows = image_rows_for_transition(payload)

    sequence = 0
    buffer: list[int] = []
    emitted: list[dict] = []
    discarded = 0
    previous_episode = None

    def discard_tail() -> None:
        nonlocal discarded
        discarded += len(buffer)
        buffer.clear()

    for transition in range(steps):
        episode = int(episode_ids[transition])
        if previous_episode is not None and episode != previous_episode:
            discard_tail()
            sequence += 2
        previous_episode = episode
        buffer.append(transition)
        if len(buffer) == CHUNK_T:
            selected = np.asarray(buffer, dtype=np.int64)
            selected_rows = rows[selected]
            name = f"chunk_nativeh200_{chunk_index:07d}.npz"
            output_path = output_dir / name
            arrays: dict[str, np.ndarray] = {
                "gt_actions": np.ascontiguousarray(
                    payload["expert_actions"][selected, 0], dtype=np.float32
                ),
                "dones": np.zeros(CHUNK_T, dtype=np.bool_),
                "prompt": np.asarray(prompt),
                "env_id": np.asarray(global_env_id),
                "seq": np.asarray(sequence),
                **{
                    key: np.ascontiguousarray(value[selected])
                    for key, value in observations.items()
                },
            }
            for view in VIEWS:
                arrays[f"depth_rawq__{view}"] = np.ascontiguousarray(
                    payload[f"depth_rawq__{view}"][selected_rows], dtype=np.uint16
                )
                arrays[f"depth_raw__{view}"] = np.ascontiguousarray(
                    payload[f"depth_raw__{view}"][selected_rows], dtype=np.float16
                )
                arrays[f"depth_ctrl__{view}"] = np.ascontiguousarray(
                    payload[f"depth_ctrl__{view}"][selected_rows], dtype=np.uint8
                )
                arrays[f"rgb_robot__{view}"] = np.ascontiguousarray(
                    payload[f"rgb_robot__{view}"][selected_rows], dtype=np.uint8
                )
            atomic_save_npz(output_path, arrays)
            emitted.append(
                {
                    "path": name,
                    "sha256": sha256_file(output_path),
                    "size_bytes": output_path.stat().st_size,
                    "global_env_id": global_env_id,
                    "episode_id": episode,
                    "episode_start_step": int(episode_steps[selected[0]]),
                    "episode_stop_step": int(episode_steps[selected[-1]]),
                    "seq": sequence,
                }
            )
            chunk_index += 1
            sequence += 1
            buffer.clear()
        if bool(dones[transition]):
            discard_tail()
            sequence += 2
            previous_episode = None

    discard_tail()
    return emitted, chunk_index, discarded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    split_manifest = args.split_manifest.resolve()
    output_dir = args.output_dir.resolve()
    document = load_json(split_manifest)
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RavenChunkError("split manifest has no environment artifacts")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RavenChunkError("output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = None
    emitted: list[dict] = []
    discarded = 0
    chunk_index = 0
    for artifact in artifacts:
        manifest_path = split_manifest.parent / artifact["manifest_path"]
        source_manifest = load_json(manifest_path)
        source_prompt = source_manifest["render"]["prompt"]
        if prompt is None:
            prompt = source_prompt
        elif prompt != source_prompt:
            raise RavenChunkError("source environment prompts differ")
        source_path = split_manifest.parent / artifact["payload_path"]
        environment_chunks, chunk_index, environment_discarded = materialize_environment(
            source_path=source_path,
            prompt=source_prompt,
            output_dir=output_dir,
            chunk_index=chunk_index,
        )
        emitted.extend(environment_chunks)
        discarded += environment_discarded

    manifest = {
        "schema": "roborender.raven_bc_chunks.v1",
        "status": "sealed",
        "source_split_manifest": str(split_manifest),
        "source_split_manifest_sha256": sha256_file(split_manifest),
        "split": document.get("split"),
        "views": list(VIEWS),
        "view_order": list(VIEWS),
        "chunk_length": CHUNK_T,
        "chunk_count": len(emitted),
        "kept_transition_count": len(emitted) * CHUNK_T,
        "discarded_partial_tail_count": discarded,
        "prompt": prompt,
        "seed": 42,
        "depth_encoding": "h5_v4_uint8",
        "robot_control": "textured_robot_on_black",
        "items": emitted,
    }
    manifest_path = output_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({key: manifest[key] for key in (
        "split", "chunk_count", "kept_transition_count",
        "discarded_partial_tail_count"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
