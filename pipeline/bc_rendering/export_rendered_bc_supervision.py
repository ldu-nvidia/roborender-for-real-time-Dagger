#!/usr/bin/env python3
"""Export frame-aligned supervision beside one rendered Raven BC chunk."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


REQUIRED_KEYS = ("env_id", "seq", "obs_actor_obs", "gt_actions", "dones")


def video_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--global-env-id", type=int, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rendered-video", type=Path, action="append", default=[])
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--render-hz", type=float, default=5.0)
    args = parser.parse_args()

    source = args.source_npz.resolve()
    with np.load(source, allow_pickle=False) as payload:
        missing = sorted(set(REQUIRED_KEYS) - set(payload.files))
        if missing:
            raise ValueError(f"Raven BC chunk is missing arrays: {missing}")
        if int(payload["env_id"].item()) != args.global_env_id:
            raise ValueError("Raven chunk environment identity differs")
        states = np.ascontiguousarray(payload["obs_actor_obs"], dtype=np.float32)
        expert_actions = np.ascontiguousarray(payload["gt_actions"], dtype=np.float32)
        dones = np.ascontiguousarray(payload["dones"], dtype=np.bool_)
        sequence_id = int(payload["seq"].item())

    frame_count = int(expert_actions.shape[0])
    if frame_count != 81:
        raise ValueError("Raven BC chunks must contain exactly 81 frames")
    if states.shape[0] != frame_count or dones.shape != (frame_count,):
        raise ValueError("state/action/done lengths are not frame-aligned")
    if not np.isfinite(states).all() or not np.isfinite(expert_actions).all():
        raise ValueError("non-finite state or expert-action values detected")

    checked_videos: list[str] = []
    for video in args.rendered_video:
        resolved = video.resolve()
        observed = video_frame_count(resolved)
        if observed != frame_count:
            raise ValueError(
                f"{resolved} has {observed} frames; supervision has {frame_count}"
            )
        checked_videos.append(video.name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    supervision_path = args.output_dir / "supervision.npz"
    frame_index = np.arange(frame_count, dtype=np.int64)
    np.savez_compressed(
        supervision_path,
        frame_index=frame_index,
        video_time_seconds=frame_index.astype(np.float64) / args.render_hz,
        simulation_time_seconds=frame_index.astype(np.float64) / args.control_hz,
        global_env_id=np.full(frame_count, args.global_env_id, dtype=np.int64),
        episode_id=np.full(frame_count, args.episode_id, dtype=np.int64),
        sequence_id=np.full(frame_count, sequence_id, dtype=np.int64),
        state=states,
        expert_action=expert_actions,
        behavior_action=expert_actions,
        expert_mask=np.ones(frame_count, dtype=np.bool_),
        done=dones,
    )

    manifest = {
        "schema": "roborender.rendered_bc_supervision.v2",
        "alignment": "video_frame_t -> state_t -> expert_action_t",
        "source_npz": str(source),
        "global_env_id": args.global_env_id,
        "episode_id": args.episode_id,
        "sequence_id": sequence_id,
        "frame_count": frame_count,
        "control_hz": args.control_hz,
        "render_hz": args.render_hz,
        "state_dim": int(states.shape[1]),
        "action_dim": int(expert_actions.shape[1]),
        "expert_labels_complete": True,
        "behavior_equals_expert": True,
        "checked_rendered_videos": checked_videos,
        "supervision_file": supervision_path.name,
    }
    (args.output_dir / "supervision_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
