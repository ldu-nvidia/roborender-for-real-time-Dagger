"""Additive promotion of admitted BC validation chunks into train-only BC.

The original production admission, split labels, data and test provenance are
immutable history. This is a new dataset identity, not an exact-resume waiver
or an evaluation/compute launch admission. No image arrays are opened here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from pipeline.bc_training import data_admission
from pipeline.bc_training.handoff_common import canonical, publish

SCHEMA = "roborender.bc-train-union-admission.v1"
ROLES = ("train", "validation")


def _read(path):
    return json.loads(data_admission.metadata_bytes(Path(path)))


def _binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": data_admission.digest(path)}


def _plan(base_admission: Path, dataset_root: Path, output_dir: Path):
    base_path = Path(base_admission).resolve(strict=True)
    root = Path(dataset_root).resolve(strict=True)
    out = Path(output_dir).resolve()
    base_binding = _binding(base_path)
    base = _read(base_path)
    roles = base.get("artifact_roles", {})
    if (base.get("schema") != data_admission.SCHEMA_V3
            or base.get("production_admitted") is not True
            or set(base.get("split_files", {})) != set(ROLES)
            or roles.get("offline_roles") != list(ROLES)
            or roles.get("offline_test_chunks") != 0):
        raise ValueError("union requires production v3 train/validation only; never promote test data")
    originals = {role: Path(base["split_files"][role]["path"]).resolve(strict=True)
                 for role in ROLES}
    # Recompute the original collection, independent-render and test-separation
    # admission; a copied production_admitted:true flag is insufficient.
    verified = data_admission.validate_training_admission(
        base_path, originals["train"], originals["validation"])
    if verified != base or _binding(base_path) != base_binding:
        raise ValueError("base admission changed during validation")
    protocol_ref = base["protocol"]
    protocol_path = Path(protocol_ref["path"]).resolve(strict=True)
    if data_admission.digest(protocol_path) != protocol_ref["raw_sha256"]:
        raise ValueError("original production protocol changed")
    protocol = _read(protocol_path)
    # The original audit authenticates this pool and its payloads at this root.
    # Matching filenames and sizes in a different directory are NOT that proof.
    source_pool = Path(protocol["source_pool"]["path"]).resolve(strict=True)
    if root != source_pool.parent:
        raise ValueError("dataset root differs from the admitted original source pool")
    source_path = Path(base["source_manifest"]).resolve(strict=True)
    if data_admission.digest(source_path) != base["source_manifest_sha256"]:
        raise ValueError("base source manifest changed")
    source = _read(source_path)
    if source.get("frames_per_chunk") != 81:
        raise ValueError("union requires complete 81-frame chunks")
    rows = {row["chunk_id"]: row for row in source["chunks"]}
    if len(rows) != len(source["chunks"]):
        raise ValueError("duplicate source chunk identities")
    if out.is_relative_to(root) or out == base_path.parent:
        raise ValueError("publish union outside the original dataset and frozen split directory")
    items, origin_roles, counts, motion_counts = [], {}, {}, Counter()
    seen_paths = {field: set() for field in ("isaac_npz", "rendered_npz")}
    for role in ROLES:
        if _binding(originals[role]) != base["split_files"][role]:
            raise ValueError("original split identity changed")
        index = _read(originals[role])
        selected = index.get("items", [])
        if (index.get("split") != role or not selected
                or index.get("chunk_count") != len(selected)):
            raise ValueError("invalid original split")
        counts[role] = len(selected)
        for item in selected:
            cid = item["chunk_id"]
            if cid in origin_roles or cid not in rows:
                raise ValueError("duplicate or unknown union chunk")
            row = rows[cid]
            if row.get("split") != role or row.get("frames") != 81:
                raise ValueError("historical role or chunk length changed")
            for field, size_field in (("isaac_npz", "source_size_bytes"),
                                      ("rendered_npz", "rendered_size_bytes")):
                relative = Path(item[field])
                path = (root / relative).resolve()
                if (relative.is_absolute() or ".." in relative.parts
                        or not path.is_relative_to(root) or item[field] != row[field]):
                    raise ValueError("payload path escapes dataset or differs from source")
                if path in seen_paths[field]:
                    raise ValueError("duplicate union payload path")
                seen_paths[field].add(path)
                if not path.is_file():
                    raise FileNotFoundError(path)
                if size_field in row and path.stat().st_size != row[size_field]:
                    raise ValueError("payload size differs from admitted source: " + str(path))
            items.append(item)  # Preserve each original item exactly.
            origin_roles[cid] = role
            motion_counts[row["motion_id"]] += 1
    if set(origin_roles) != set(rows) or len(items) % 16:
        raise ValueError("union must cover the exact original pool in complete B16 batches")
    shared = {"schema": "roborender.bc_split.v1", "source_manifest": str(source_path),
              "source_manifest_sha256": base["source_manifest_sha256"]}
    train = {**shared, "split": "train", "chunk_count": len(items),
             "frame_count": len(items) * 81, "items": items}
    retired = {**shared, "split": "validation", "chunk_count": 0, "frame_count": 0,
               "items": [], "retired": True, "reason": "promoted_to_training_by_user",
               "original_validation": base["split_files"]["validation"]}
    documents = {"train.json": train, "validation-retired.json": retired}
    split_files = {role: {"path": str(out / filename),
                         "sha256": hashlib.sha256(canonical(documents[filename]) + b"\n").hexdigest()}
                   for role, filename in (("train", "train.json"),
                                          ("validation", "validation-retired.json"))}
    report = {"schema": SCHEMA, "production_admitted": True,
              "union_validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "data_only_not_training_or_evaluation_launch",
              "training_protocol": "fixed_final", "validation_enabled": False,
              "base_admission": base_binding, "dataset_root": str(root),
              "source_manifest": str(source_path),
              "source_manifest_sha256": base["source_manifest_sha256"],
              "original_split_files": base["split_files"], "split_files": split_files,
              "original_chunk_counts": counts, "chunk_count": len(items),
              "frame_count": len(items) * 81, "updates_per_epoch_b16": len(items) // 16,
              "origin_roles": origin_roles, "motion_chunk_counts": dict(sorted(motion_counts.items())),
              "protected_historical_test": roles["closed_loop_test"],
              "evaluation_launch_admitted": False,
              "limitations": [
                  "Former validation chunks are training data, never held-out MSE data again in this lineage.",
                  "Historical test provenance is preserved, not authorization for its old observation mode.",
                  "New success evaluation requires a current generated-only evaluation admission.",
                  "Metadata/provenance, file existence and sizes checked; payload bytes not rehashed here.",
                  "Changed dataset identity cannot silently resume an old BC checkpoint.",
              ]}
    return documents, report


def validate_training_admission(report_path: Path, train_split: Path,
                                validation_split: Path, dataset_root: Path) -> dict:
    report_path = Path(report_path).resolve(strict=True)
    report = _read(report_path)
    if report.get("schema") != SCHEMA or report.get("training_protocol") != "fixed_final":
        raise ValueError("unknown train-only union admission")
    if str(Path(dataset_root).resolve(strict=True)) != report.get("dataset_root"):
        raise ValueError("union dataset root differs")
    base = report["base_admission"]
    if _binding(base["path"]) != base:
        raise ValueError("union base admission changed")
    for role, path in (("train", train_split), ("validation", validation_split)):
        if _binding(path) != report["split_files"][role]:
            raise ValueError("union split identity changed")
    documents, expected = _plan(Path(base["path"]), dataset_root, report_path.parent)
    if expected != report:
        raise ValueError("union admission differs from recomputed exact union")
    for filename, document in documents.items():
        if data_admission.metadata_bytes(report_path.parent / filename) != canonical(document) + b"\n":
            raise ValueError("union index differs from original train plus validation")
    return report


def build_union(base_admission: Path, dataset_root: Path, output_dir: Path) -> dict:
    """Publish a new manifest/admission, never move or overwrite historical data."""
    out = Path(output_dir).resolve()
    documents, report = _plan(base_admission, dataset_root, out)
    documents["data-admission.json"] = report  # Publish admission last.
    for filename, document in documents.items():
        path = out / filename
        if path.exists() and data_admission.metadata_bytes(path) != canonical(document) + b"\n":
            raise ValueError("refusing to overwrite an existing union artifact: " + str(path))
    for filename, document in documents.items():
        path = out / filename
        expected = canonical(document) + b"\n"
        if path.exists():
            if data_admission.metadata_bytes(path) != expected:
                raise ValueError("refusing to overwrite an existing union artifact: " + str(path))
        else:
            try:
                publish(path, document)
            except FileExistsError:
                if data_admission.metadata_bytes(path) != expected:
                    raise ValueError("concurrent union publication differs: " + str(path))
    return validate_training_admission(out / "data-admission.json", out / "train.json",
                                       out / "validation-retired.json", dataset_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-admission", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_union(args.base_admission, args.dataset_root, args.output_dir)
    print(json.dumps({key: report[key] for key in
        ("schema", "chunk_count", "frame_count", "updates_per_epoch_b16", "split_files")}))


if __name__ == "__main__":
    main()
