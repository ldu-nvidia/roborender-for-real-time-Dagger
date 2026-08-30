#!/usr/bin/env python3
"""Rebuild destination-local replay snapshots for a portable smoke handoff.

The copied cohort directories are immutable content.  Replay snapshots are not
portable because they intentionally contain absolute cohort paths.  This tool
pins the copied source-snapshot bytes and semantics, recursively verifies the
destination cohorts, proves exact cohort/payload equivalence and train/eval
disjointness, then publishes new destination-local snapshots.  The completion
manifest is written last; no cohort, shard, sidecar, or NPZ is rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger import artifacts, replay  # noqa: E402


HANDOFF_SCHEMA = "fast_roborender.round_dagger.portable_handoff.v1"
DEFAULT_WORLD_SIZE = 8
DEFAULT_HELDOUT_CASES = 256
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_DIGEST_FIELDS = (
    "policy_sha256",
    "runtime_policy_sha256",
    "teacher_sha256",
    "renderer_sha256",
    "beta_sha256",
    "config_sha256",
)
_TRAIN_SNAPSHOT_KEYS = {
    "schema",
    "campaign_id",
    "through_macro_round",
    "expected_train_cohorts",
    "cohort_manifest_schema",
    "view_contract",
    "action_contract",
    "cohort_count",
    "transition_count",
    "counts_by_round",
    "counts_by_round_cohort",
    "cohorts",
    "snapshot_sha256",
}
_HELDOUT_SNAPSHOT_KEYS = {
    "schema",
    "purpose",
    "campaign_id",
    "cohort_manifest_schema",
    "view_contract",
    "action_contract",
    "cohort_count",
    "transition_count",
    "cohorts",
    "snapshot_sha256",
}


class HandoffError(RuntimeError):
    """Portable handoff verification or publication failed."""


def _absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise HandoffError(f"{label} must be a lowercase SHA-256")
    return value


def _positive(value: int, label: str) -> int:
    if type(value) is not int or value < 1:
        raise HandoffError(f"{label} must be a positive integer")
    return value


def _load_pinned_source_snapshot(
    path: Path,
    *,
    heldout: bool,
    expected_file_sha256: str,
    expected_snapshot_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise HandoffError("source snapshot must be a regular non-symlink file")
    expected_file_sha256 = _sha(expected_file_sha256, "source snapshot file SHA-256")
    expected_snapshot_sha256 = _sha(
        expected_snapshot_sha256, "source snapshot semantic SHA-256"
    )
    actual_file_sha256 = artifacts.sha256_file(path)
    if actual_file_sha256 != expected_file_sha256:
        raise HandoffError("source snapshot file SHA-256 differs from its handoff pin")
    document = artifacts.load_strict_json(path)
    expected_keys = _HELDOUT_SNAPSHOT_KEYS if heldout else _TRAIN_SNAPSHOT_KEYS
    expected_schema = (
        replay.HELDOUT_SNAPSHOT_SCHEMA if heldout else replay.SNAPSHOT_SCHEMA
    )
    if not isinstance(document, Mapping) or set(document) != expected_keys:
        raise HandoffError("source snapshot fields are incomplete or unexpected")
    document = dict(document)
    if document["schema"] != expected_schema:
        raise HandoffError("source snapshot schema is unsupported")
    if heldout and document["purpose"] != "eval":
        raise HandoffError("source held-out snapshot purpose is not eval")
    declared = _sha(document["snapshot_sha256"], "source snapshot semantic SHA-256")
    semantic_body = dict(document)
    semantic_body.pop("snapshot_sha256")
    actual_semantic = artifacts.canonical_json_sha256(semantic_body)
    if declared != actual_semantic or declared != expected_snapshot_sha256:
        raise HandoffError("source snapshot semantic SHA-256 differs from its handoff pin")
    cohorts = document.get("cohorts")
    if (
        document.get("cohort_count") != 1
        or not isinstance(cohorts, list)
        or len(cohorts) != 1
        or not isinstance(cohorts[0], Mapping)
    ):
        raise HandoffError("portable source snapshot must bind exactly one cohort")
    if not heldout and (
        document.get("through_macro_round") != 0
        or document.get("expected_train_cohorts") != 1
    ):
        raise HandoffError("portable training snapshot must bind round 0/cohort 0")
    return document, {
        "path": str(path),
        "file_sha256": actual_file_sha256,
        "snapshot_sha256": declared,
        "campaign_id": document["campaign_id"],
        "transition_count": document["transition_count"],
    }


def _tree_inventory(root: Path) -> tuple[list[dict[str, Any]], dict[str, tuple[int, ...]]]:
    if not root.is_dir() or root.is_symlink():
        raise HandoffError(f"cohort directory is missing or unsafe: {root}")
    content: list[dict[str, Any]] = []
    physical: dict[str, tuple[int, ...]] = {}
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        for name in [*dirnames, *filenames]:
            path = Path(directory) / name
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode):
                raise HandoffError(f"cohort tree contains a symbolic link: {path}")
        for name in filenames:
            path = Path(directory) / name
            info = os.stat(path, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise HandoffError(f"cohort tree contains a non-regular file: {path}")
            relative = path.relative_to(root).as_posix()
            content.append(
                {
                    "path": relative,
                    "size_bytes": info.st_size,
                    "sha256": artifacts.sha256_file(path),
                }
            )
            physical[relative] = (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                stat.S_IMODE(info.st_mode),
            )
    return content, physical


def _payload_set(report: Mapping[str, Any]) -> set[str]:
    values = {str(item["payload_sha256"]) for item in report["shards"]}
    if len(values) != len(report["shards"]):
        raise HandoffError("cohort contains duplicate payload identities")
    return values


def _source_train_payload_set(document: Mapping[str, Any]) -> set[str]:
    cohort = document["cohorts"][0]
    shards = cohort.get("shards")
    if not isinstance(shards, list) or not shards:
        raise HandoffError("source training snapshot does not bind its shards")
    if any(not isinstance(item, Mapping) for item in shards):
        raise HandoffError("source training snapshot contains a malformed shard")
    values = {
        _sha(item.get("payload_sha256"), "source train payload SHA-256")
        for item in shards
    }
    if len(values) != len(shards):
        raise HandoffError("source training snapshot repeats a payload identity")
    return values


def _bind_source_cohort(
    source_snapshot: Mapping[str, Any], destination_report: Mapping[str, Any], label: str
) -> None:
    cohort = source_snapshot["cohorts"][0]
    expected = {
        "logical_id": destination_report["logical_id"],
        "macro_round": destination_report["macro_round"],
        "cohort_index": destination_report["cohort_index"],
        "manifest_sha256": destination_report["cohort_sha256"],
        "transition_count": destination_report["transition_count"],
    }
    mismatches = sorted(key for key, value in expected.items() if cohort.get(key) != value)
    if mismatches:
        raise HandoffError(f"source {label} snapshot/destination cohort mismatch: {mismatches}")


def rematerialize(
    *,
    destination_root: str | os.PathLike[str],
    train_cohort: str | os.PathLike[str],
    heldout_cohort: str | os.PathLike[str],
    source_training_snapshot: str | os.PathLike[str],
    source_heldout_snapshot: str | os.PathLike[str],
    expected_source_training_file_sha256: str,
    expected_source_training_snapshot_sha256: str,
    expected_source_heldout_file_sha256: str,
    expected_source_heldout_snapshot_sha256: str,
    expected_world_size: int = DEFAULT_WORLD_SIZE,
    expected_heldout_cases: int = DEFAULT_HELDOUT_CASES,
) -> dict[str, Any]:
    """Verify copied cohort bytes and publish path-local replay snapshots."""

    root = _absolute(destination_root)
    if not root.is_dir() or root.is_symlink():
        raise HandoffError("destination_root must be an existing non-symlink directory")
    expected_world_size = _positive(expected_world_size, "expected_world_size")
    expected_heldout_cases = _positive(
        expected_heldout_cases, "expected_heldout_cases"
    )
    train_path = _absolute(train_cohort)
    heldout_path = _absolute(heldout_cohort)
    for path, label in ((train_path, "train cohort"), (heldout_path, "held-out cohort")):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise HandoffError(f"{label} must be inside destination_root") from exc
    if train_path == heldout_path:
        raise HandoffError("train and held-out cohort directories must differ")

    output_train = root / "training_snapshot.json"
    output_heldout = root / "heldout_snapshot.json"
    completion_path = root / "portable_handoff_manifest.json"
    for path in (output_train, output_heldout, completion_path):
        if path.exists() or path.is_symlink():
            raise HandoffError(f"destination publication already exists: {path.name}")

    source_train, source_train_identity = _load_pinned_source_snapshot(
        _absolute(source_training_snapshot),
        heldout=False,
        expected_file_sha256=expected_source_training_file_sha256,
        expected_snapshot_sha256=expected_source_training_snapshot_sha256,
    )
    source_heldout, source_heldout_identity = _load_pinned_source_snapshot(
        _absolute(source_heldout_snapshot),
        heldout=True,
        expected_file_sha256=expected_source_heldout_file_sha256,
        expected_snapshot_sha256=expected_source_heldout_snapshot_sha256,
    )

    train_inventory, train_physical_before = _tree_inventory(train_path)
    heldout_inventory, heldout_physical_before = _tree_inventory(heldout_path)
    train_report = artifacts.verify_cohort(train_path)
    heldout_report = artifacts.verify_cohort(heldout_path)
    if train_report["purpose"] != "train" or heldout_report["purpose"] != "eval":
        raise HandoffError("destination cohort purposes are not train/eval")
    if (
        train_report["macro_round"] != 0
        or train_report["cohort_index"] != 0
        or heldout_report["macro_round"] != 0
        or heldout_report["cohort_index"] != 1
    ):
        raise HandoffError("portable cohorts must be round 0 train/eval indices 0/1")
    expected_ranks = list(range(expected_world_size))
    if train_report["ranks"] != expected_ranks or heldout_report["ranks"] != expected_ranks:
        raise HandoffError("destination cohort rank coverage differs from world size")
    if heldout_report["transition_count"] != expected_heldout_cases:
        raise HandoffError("destination held-out case count differs from contract")
    if heldout_report["transition_count"] != len(heldout_report["global_env_ids"]):
        raise HandoffError("held-out cohort must contain exactly one case per environment")
    if (
        train_report["campaign_id"] != heldout_report["campaign_id"]
        or train_report["view_contract"] != heldout_report["view_contract"]
        or train_report["action_contract"] != heldout_report["action_contract"]
        or any(train_report[field] != heldout_report[field] for field in _DIGEST_FIELDS)
    ):
        raise HandoffError("destination train/eval campaign or model lineage differs")
    if (
        source_train["campaign_id"] != train_report["campaign_id"]
        or source_heldout["campaign_id"] != heldout_report["campaign_id"]
    ):
        raise HandoffError("source snapshots and destination cohorts span campaigns")
    _bind_source_cohort(source_train, train_report, "training")
    _bind_source_cohort(source_heldout, heldout_report, "held-out")

    train_payloads = _payload_set(train_report)
    heldout_payloads = _payload_set(heldout_report)
    source_train_payloads = _source_train_payload_set(source_train)
    if source_train_payloads != train_payloads:
        raise HandoffError("source/destination training payload sets differ")
    # The source held-out snapshot binds the identical seal digest, and that
    # seal recursively binds this exact payload set.
    collisions = sorted(train_payloads & heldout_payloads)
    if collisions:
        raise HandoffError(
            "training and held-out payload identities overlap: " + ",".join(collisions)
        )

    with tempfile.TemporaryDirectory(prefix=".portable-replay-", dir=root) as temporary:
        with replay.ReplayLedger(Path(temporary) / "replay.sqlite3") as ledger:
            admission = ledger.ingest_cohort(train_path)
            destination_train = ledger.build_snapshot(
                train_report["campaign_id"],
                through_macro_round=0,
                expected_train_cohorts=1,
                output=output_train,
            )
    destination_heldout = replay.build_heldout_snapshot(
        [heldout_path], output=output_heldout
    )
    loaded_train = replay.load_snapshot(
        output_train, expected_file_sha256=destination_train["file_sha256"]
    )
    loaded_heldout = replay.load_heldout_snapshot(
        output_heldout, expected_file_sha256=destination_heldout["file_sha256"]
    )
    if (
        {item["payload_sha256"] for item in loaded_train["resolved_shards"]}
        != train_payloads
        or {item["payload_sha256"] for item in loaded_heldout["resolved_shards"]}
        != heldout_payloads
    ):
        raise HandoffError("destination snapshot payload resolution changed")

    train_inventory_after, train_physical_after = _tree_inventory(train_path)
    heldout_inventory_after, heldout_physical_after = _tree_inventory(heldout_path)
    if (
        train_inventory_after != train_inventory
        or heldout_inventory_after != heldout_inventory
        or train_physical_after != train_physical_before
        or heldout_physical_after != heldout_physical_before
    ):
        raise HandoffError("cohort bytes or filesystem identities changed during rebasing")

    manifest: dict[str, Any] = {
        "schema": HANDOFF_SCHEMA,
        "complete": True,
        "destination_root": str(root),
        "campaign_id": train_report["campaign_id"],
        "source": {
            "training_snapshot": source_train_identity,
            "heldout_snapshot": source_heldout_identity,
        },
        "destination": {
            "training_snapshot": {
                "path": str(output_train),
                "file_sha256": destination_train["file_sha256"],
                "snapshot_sha256": destination_train["snapshot_sha256"],
                "transition_count": destination_train["transition_count"],
            },
            "heldout_snapshot": {
                "path": str(output_heldout),
                "file_sha256": destination_heldout["file_sha256"],
                "snapshot_sha256": destination_heldout["snapshot_sha256"],
                "transition_count": destination_heldout["transition_count"],
            },
            "training_cohort": {
                "path": str(train_path),
                "seal_sha256": train_report["cohort_sha256"],
            },
            "heldout_cohort": {
                "path": str(heldout_path),
                "seal_sha256": heldout_report["cohort_sha256"],
            },
            "replay_admission_status": admission["status"],
        },
        "equivalence": {
            "source_destination_cohort_seals_exact": True,
            "source_destination_training_payload_sets_exact": True,
            "source_destination_heldout_payload_sets_exact_via_seal": True,
            "training_payload_sha256s": sorted(train_payloads),
            "heldout_payload_sha256s": sorted(heldout_payloads),
            "train_eval_payload_intersection": collisions,
        },
        "cohort_immutability": {
            "verified": True,
            "train_file_count": len(train_inventory),
            "train_inventory_sha256": artifacts.canonical_json_sha256(train_inventory),
            "heldout_file_count": len(heldout_inventory),
            "heldout_inventory_sha256": artifacts.canonical_json_sha256(
                heldout_inventory
            ),
            "content_and_inode_metadata_unchanged": True,
        },
        "builder": {
            "path": str(_absolute(__file__)),
            "sha256": artifacts.sha256_file(_absolute(__file__)),
        },
    }
    manifest["handoff_content_sha256"] = artifacts.canonical_json_sha256(manifest)
    artifacts.publish_immutable_json(completion_path, manifest)
    result = dict(manifest)
    result["manifest_path"] = str(completion_path)
    result["manifest_file_sha256"] = artifacts.sha256_file(completion_path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination-root", required=True, type=Path)
    parser.add_argument("--train-cohort", required=True, type=Path)
    parser.add_argument("--heldout-cohort", required=True, type=Path)
    parser.add_argument("--source-training-snapshot", required=True, type=Path)
    parser.add_argument("--source-heldout-snapshot", required=True, type=Path)
    parser.add_argument("--expected-source-training-file-sha256", required=True)
    parser.add_argument("--expected-source-training-snapshot-sha256", required=True)
    parser.add_argument("--expected-source-heldout-file-sha256", required=True)
    parser.add_argument("--expected-source-heldout-snapshot-sha256", required=True)
    parser.add_argument("--expected-world-size", type=int, default=DEFAULT_WORLD_SIZE)
    parser.add_argument(
        "--expected-heldout-cases", type=int, default=DEFAULT_HELDOUT_CASES
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = rematerialize(
        destination_root=args.destination_root,
        train_cohort=args.train_cohort,
        heldout_cohort=args.heldout_cohort,
        source_training_snapshot=args.source_training_snapshot,
        source_heldout_snapshot=args.source_heldout_snapshot,
        expected_source_training_file_sha256=(
            args.expected_source_training_file_sha256
        ),
        expected_source_training_snapshot_sha256=(
            args.expected_source_training_snapshot_sha256
        ),
        expected_source_heldout_file_sha256=args.expected_source_heldout_file_sha256,
        expected_source_heldout_snapshot_sha256=(
            args.expected_source_heldout_snapshot_sha256
        ),
        expected_world_size=args.expected_world_size,
        expected_heldout_cases=args.expected_heldout_cases,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest_path": result["manifest_path"],
                "manifest_file_sha256": result["manifest_file_sha256"],
                "training_snapshot_sha256": result["destination"][
                    "training_snapshot"
                ]["snapshot_sha256"],
                "heldout_snapshot_sha256": result["destination"][
                    "heldout_snapshot"
                ]["snapshot_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
