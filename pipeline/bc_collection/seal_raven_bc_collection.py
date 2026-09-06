#!/usr/bin/env python3
"""Validate Raven chunks and publish one additive flat dataset manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np


CHUNK_LENGTH = 81
VIEWS = ("hand_left", "hand_right", "head_color")


class SealError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_chunk(path: Path, *, envs_per_rank: int, run_tag: str) -> dict:
    pattern = re.compile(
        rf"^chunk_{re.escape(run_tag)}_r(?P<rank>[0-9]{{2}})_(?P<index>[0-9]{{7}})\.npz$"
    )
    match = pattern.fullmatch(path.name)
    if match is None:
        raise SealError(f"unexpected Raven chunk name: {path.name}")
    rank = int(match.group("rank"))
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        required = {
            "gt_actions", "dones", "prompt", "env_id", "seq",
            *(f"depth_rawq__{view}" for view in VIEWS),
            *(f"depth_raw__{view}" for view in VIEWS),
            *(f"depth_ctrl__{view}" for view in VIEWS),
            *(f"rgb_robot__{view}" for view in VIEWS),
        }
        missing = sorted(required - keys)
        if missing:
            raise SealError(f"{path.name} lacks fields: {missing}")
        actions = archive["gt_actions"]
        if actions.shape != (CHUNK_LENGTH, 26) or actions.dtype != np.float32:
            raise SealError(f"{path.name} has invalid 26-D teacher actions")
        if not np.isfinite(actions).all():
            raise SealError(f"{path.name} has non-finite teacher actions")
        dones = archive["dones"]
        if dones.shape != (CHUNK_LENGTH,) or dones.dtype != np.bool_ or dones.any():
            raise SealError(f"{path.name} has invalid all-false Raven dones")
        prompt = str(archive["prompt"])
        env_id = int(archive["env_id"])
        sequence = int(archive["seq"])
        if not 0 <= env_id < envs_per_rank or sequence < 0:
            raise SealError(f"{path.name} has invalid env_id/seq")
        obs_keys = sorted(key for key in keys if key.startswith("obs_"))
        if "obs_actor_obs" not in obs_keys:
            raise SealError(f"{path.name} lacks obs_actor_obs")
        for key in obs_keys:
            value = archive[key]
            if value.shape[0] != CHUNK_LENGTH or value.dtype != np.float32:
                raise SealError(f"{path.name}:{key} has invalid time axis/dtype")
            if not np.isfinite(value).all():
                raise SealError(f"{path.name}:{key} contains non-finite values")
        for view in VIEWS:
            expected = {
                f"depth_rawq__{view}": ((CHUNK_LENGTH, 480, 832), np.uint16),
                f"depth_raw__{view}": ((CHUNK_LENGTH, 240, 416), np.float16),
                f"depth_ctrl__{view}": ((CHUNK_LENGTH, 240, 416), np.uint8),
                f"rgb_robot__{view}": ((CHUNK_LENGTH, 240, 416, 3), np.uint8),
            }
            for key, (shape, dtype) in expected.items():
                value = archive[key]
                if value.shape != shape or value.dtype != dtype:
                    raise SealError(
                        f"{path.name}:{key} is {value.shape}/{value.dtype}, "
                        f"expected {shape}/{np.dtype(dtype)}"
                    )
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rank": rank,
        "env_id": env_id,
        "global_env_id": rank * envs_per_rank + env_id,
        "seq": sequence,
        "prompt": prompt,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--envs-per-rank", type=int, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset_dir.resolve()
    contract = args.contract.resolve()
    if args.world_size <= 0 or args.envs_per_rank <= 0:
        raise SealError("world size and environments per rank must be positive")
    partials = list(dataset.glob("*.partial"))
    if partials:
        raise SealError(f"dataset has incomplete files: {[path.name for path in partials]}")
    chunks = sorted(dataset.glob(f"chunk_{args.run_tag}_r??_???????.npz"))
    if not chunks:
        raise SealError("collection produced no Raven chunks")
    items = [
        validate_chunk(
            path,
            envs_per_rank=args.envs_per_rank,
            run_tag=args.run_tag,
        )
        for path in chunks
    ]
    ranks = sorted({item["rank"] for item in items})
    if ranks != list(range(args.world_size)):
        raise SealError(f"published ranks {ranks} differ from expected world size")
    manifest = {
        "schema": "roborender.raven_bc_isaac_dataset.v1",
        "status": "sealed",
        "contract_source": "qingh097/fast_roborender@videogen-pipeline",
        "gr00t_reference_commit": "94d7985",
        "contract_path": str(contract),
        "contract_sha256": sha256_file(contract),
        "run_tag": args.run_tag,
        "world_size": args.world_size,
        "environments_per_rank": args.envs_per_rank,
        "environment_count": args.world_size * args.envs_per_rank,
        "chunk_length": CHUNK_LENGTH,
        "chunk_count": len(items),
        "views": list(VIEWS),
        "items": items,
    }
    output = dataset / "manifest.json"
    temporary = dataset / "manifest.json.partial"
    if output.exists() or temporary.exists():
        raise SealError(f"refusing to overwrite {output}")
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(json.dumps({"manifest": str(output), "chunks": len(items)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
