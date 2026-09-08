#!/usr/bin/env python3
"""Publish an exact per-rank quota from collector staging into the BC dataset."""

from __future__ import annotations

import argparse
import os
import re
import zipfile
from pathlib import Path


class PublishError(RuntimeError):
    pass


def complete_npz(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
        return {"gt_actions.npy", "dones.npy", "env_id.npy", "seq.npy"} <= names
    except (OSError, zipfile.BadZipFile):
        return False


def rank_files(stage: Path, run_tag: str, rank: int) -> list[Path]:
    pattern = re.compile(
        rf"chunk_{re.escape(run_tag)}_r{rank:02d}_(?P<index>[0-9]{{7}})\.npz"
    )
    indexed: list[tuple[int, Path]] = []
    for path in stage.glob(f"chunk_{run_tag}_r{rank:02d}_*.npz"):
        match = pattern.fullmatch(path.name)
        if match is not None and complete_npz(path):
            indexed.append((int(match.group("index")), path))
    return [path for _, path in sorted(indexed)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--chunks-per-rank", type=int, required=True)
    args = parser.parse_args()

    if args.world_size <= 0 or args.chunks_per_rank <= 0:
        raise PublishError("world size and per-rank quota must be positive")
    staging = args.staging_root.resolve()
    dataset = args.dataset_dir.resolve()
    dataset.mkdir(parents=True, exist_ok=True)
    if any(dataset.iterdir()):
        raise PublishError(f"canonical dataset directory is not empty: {dataset}")

    selected: list[Path] = []
    for rank in range(args.world_size):
        files = rank_files(staging / f"rank-{rank:02d}", args.run_tag, rank)
        if len(files) < args.chunks_per_rank:
            raise PublishError(
                f"rank {rank} has {len(files)} complete chunks; "
                f"need {args.chunks_per_rank}"
            )
        selected.extend(files[: args.chunks_per_rank])

    published: list[Path] = []
    try:
        for source in selected:
            destination = dataset / source.name
            os.link(source, destination)
            published.append(destination)
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    print(
        f"published {len(published)} complete Raven chunks from "
        f"{args.world_size} ranks to {dataset}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
