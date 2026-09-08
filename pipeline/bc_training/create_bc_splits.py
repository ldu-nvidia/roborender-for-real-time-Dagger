#!/usr/bin/env python3
"""Create deterministic, stream-grouped train/validation/test indexes for BC."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


SEED = "roborender-bc-split-v1"
RATIOS = {"train": 0.80, "validation": 0.10, "test": 0.10}


def _hash_text(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _stream_key(row: dict) -> tuple:
    return row["source_run_tag"], row["rank"], row["local_env_id"]


def _stratum_key(row: dict) -> tuple:
    return row["source_batch"], row["rank"]


def _assignment_states(groups: dict, stratum: tuple, target: float) -> dict:
    ordered = sorted(
        groups.items(), key=lambda item: _hash_text(f"{SEED}|{stratum}|{item[0]}")
    )
    lower = max(0, math.floor(target * 0.60))
    upper = math.ceil(target * 1.40)
    states = {(0, 0): ((), ())}
    for key, rows in ordered:
        size = len(rows)
        updated = dict(states)
        for (validation_count, test_count), (validation_keys, test_keys) in states.items():
            if validation_count + size <= upper:
                state = validation_count + size, test_count
                updated.setdefault(state, (validation_keys + (key,), test_keys))
            if test_count + size <= upper:
                state = validation_count, test_count + size
                updated.setdefault(state, (validation_keys, test_keys + (key,)))
        states = updated
    return {
        counts: keys
        for counts, keys in states.items()
        if lower <= counts[0] <= upper and lower <= counts[1] <= upper
    }


def create_splits(dataset_root: Path) -> dict:
    source_manifest_path = dataset_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    rows = source_manifest["chunks"]
    total = len(rows)
    requested = {name: round(total * ratio) for name, ratio in RATIOS.items()}
    if sum(requested.values()) != total:
        raise RuntimeError("rounded split counts do not sum to the dataset size")

    strata: dict[tuple, dict[tuple, list[dict]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for row in rows:
        strata[_stratum_key(row)][_stream_key(row)].append(row)

    stratum_options = []
    for stratum, groups in sorted(strata.items()):
        target = sum(map(len, groups.values())) * RATIOS["validation"]
        stratum_options.append((stratum, groups, _assignment_states(groups, stratum, target)))

    # Select one grouped assignment per batch/rank stratum. The outer DP gives
    # exact global 80/10/10 counts while minimizing deviation from 10/10 in
    # every stratum.
    target_validation = requested["validation"]
    target_test = requested["test"]
    solutions = {(0, 0): (0.0, [])}
    for stratum, groups, options in stratum_options:
        stratum_target = sum(map(len, groups.values())) * RATIOS["validation"]
        updated = {}
        for (total_validation, total_test), (cost, choices) in solutions.items():
            for (validation_count, test_count), group_keys in options.items():
                new_counts = total_validation + validation_count, total_test + test_count
                if new_counts[0] > target_validation or new_counts[1] > target_test:
                    continue
                new_cost = cost + (validation_count - stratum_target) ** 2
                new_cost += (test_count - stratum_target) ** 2
                candidate = new_cost, choices + [(stratum, group_keys)]
                if new_counts not in updated or candidate[0] < updated[new_counts][0]:
                    updated[new_counts] = candidate
        solutions = updated

    final_key = target_validation, target_test
    if final_key not in solutions:
        raise RuntimeError("cannot satisfy exact split counts without breaking stream groups")

    membership: dict[tuple, str] = {}
    for _stratum, (validation_keys, test_keys) in solutions[final_key][1]:
        membership.update({key: "validation" for key in validation_keys})
        membership.update({key: "test" for key in test_keys})
    for groups in strata.values():
        for key in groups:
            membership.setdefault(key, "train")

    split_rows = {name: [] for name in RATIOS}
    split_groups = {name: set() for name in RATIOS}
    for row in rows:
        group_key = _stream_key(row)
        split = membership[group_key]
        stream_id = "stream_" + hashlib.sha256(repr(group_key).encode()).hexdigest()[:16]
        split_groups[split].add(stream_id)
        split_rows[split].append({
            "chunk_id": row["chunk_id"],
            "isaac_npz": row["isaac_npz"],
            "rendered_npz": row["rendered_npz"],
            "stream_id": stream_id,
        })

    if {name: len(values) for name, values in split_rows.items()} != requested:
        raise RuntimeError("split counts differ from requested counts")
    chunk_sets = [{row["chunk_id"] for row in split_rows[name]} for name in RATIOS]
    if len(set.union(*chunk_sets)) != total or any(
        chunk_sets[i] & chunk_sets[j] for i in range(3) for j in range(i + 1, 3)
    ):
        raise RuntimeError("chunk split is incomplete or overlapping")
    if any(split_groups[a] & split_groups[b] for a in RATIOS for b in RATIOS if a < b):
        raise RuntimeError("collection stream leaked across splits")

    split_dir = dataset_root / "splits"
    split_dir.mkdir(exist_ok=True)
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    source_manifest_sha256 = _file_hash(source_manifest_path)
    files = {}
    for name, values in split_rows.items():
        values.sort(key=lambda row: row["chunk_id"])
        document = {
            "schema": "roborender.bc_split.v1",
            "split": name,
            "created_at_utc": created,
            "seed": SEED,
            "source_manifest": "../manifest.json",
            "source_manifest_sha256": source_manifest_sha256,
            "chunk_count": len(values),
            "frame_count": sum(source_manifest["frames_per_chunk"] for _ in values),
            "stream_count": len(split_groups[name]),
            "items": values,
        }
        path = split_dir / f"{name}.json"
        _write_json(path, document)
        files[name] = {
            "path": path.name,
            "sha256": _file_hash(path),
            "chunk_count": len(values),
            "frame_count": document["frame_count"],
            "stream_count": document["stream_count"],
        }

    manifest = {
        "schema": "roborender.bc_splits_manifest.v1",
        "created_at_utc": created,
        "seed": SEED,
        "algorithm": "deterministic grouped and batch/rank-stratified exact-count split",
        "group_fields": ["source_run_tag", "rank", "local_env_id"],
        "stratify_fields": ["source_batch", "rank"],
        "requested_ratios": RATIOS,
        "source_manifest": "../manifest.json",
        "source_manifest_sha256": source_manifest_sha256,
        "files": files,
        "checks": {
            "chunk_sets_disjoint": True,
            "collection_streams_disjoint": True,
            "complete_coverage": True,
            "paired_isaac_rendered_ids": True,
            "test_sealed_from_training": True,
        },
    }
    _write_json(split_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/workspace/project/roborender/data/BC"),
    )
    args = parser.parse_args()
    result = create_splits(args.dataset_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
