#!/usr/bin/env python3
"""Content-addressed gates for the HSG GB200 dual-container DAgger smokes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
VIEWS = ("head_color", "hand_left", "hand_right")
SEALED_SOURCE_PROFILES = frozenset(
    {"calibration_3view", "smoke_32x300", "smoke_256x300", "production_256x300"}
)


class GateError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_tree_evidence(root: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash a staged tree and return its file inventory in one read pass."""

    root = require_dir(root, "source tree")
    digest = hashlib.sha256()
    candidates = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            raise GateError(
                f"sealed source tree contains executable bytecode excluded from "
                f"source provenance: {path}"
            )
        # A symlink to a directory reports is_dir() == True.  Record the link
        # itself before applying the directory skip so staged source cannot
        # redirect a subtree without changing the content-addressed digest.
        if path.is_symlink():
            candidates.append((relative, path))
            continue
        if path.is_dir():
            continue
        candidates.append((relative, path))
    inventory: list[dict[str, Any]] = []
    for relative, path in sorted(candidates, key=lambda item: item[0].as_posix()):
        encoded = relative.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        mode = path.lstat().st_mode & 0o777
        digest.update(mode.to_bytes(2, "big"))
        if path.is_symlink():
            payload = os.readlink(path).encode("utf-8")
            digest.update(b"L" + len(payload).to_bytes(8, "big") + payload)
            inventory.append(
                {
                    "path": relative.as_posix(),
                    "mode": oct(mode),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "kind": "symlink",
                }
            )
        elif path.is_file():
            digest.update(b"F")
            file_digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
                    file_digest.update(chunk)
            inventory.append(
                {
                    "path": relative.as_posix(),
                    "mode": oct(mode),
                    "bytes": path.stat().st_size,
                    "sha256": file_digest.hexdigest(),
                }
            )
        else:
            raise GateError(f"unsupported source-tree entry: {path}")
    return digest.hexdigest(), inventory


def source_tree_sha256(root: Path) -> str:
    """Hash a staged source tree without relying on Git metadata."""

    return source_tree_evidence(root)[0]


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    if not isinstance(result, dict):
        raise GateError(f"{path} must contain one JSON object")
    return result


def require_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise GateError(f"{label} must not be a symbolic link: {path}")
    path = path.resolve(strict=True)
    if not path.is_file() or path.stat().st_size <= 0:
        raise GateError(f"{label} is not a non-empty regular file: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise GateError(f"{label} must not be a symbolic link: {path}")
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise GateError(f"{label} is not a directory: {path}")
    return path


def require_hash(path: Path, expected: str, label: str) -> str:
    if SHA256_RE.fullmatch(expected) is None:
        raise GateError(f"expected {label} SHA-256 is malformed")
    actual = sha256_file(require_file(path, label))
    if actual != expected:
        raise GateError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return actual


def require_revision_marker(repo: Path, expected: str, label: str) -> str:
    if REVISION_RE.fullmatch(expected) is None:
        raise GateError(f"expected {label} revision is malformed")
    marker = require_file(repo / ".source_revision", f"{label} revision marker")
    actual = marker.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise GateError(f"{label} revision mismatch: {actual!r} != {expected!r}")
    return actual


def _require_exact_keys(document: dict[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - document.keys())
    unexpected = sorted(document.keys() - expected)
    if missing or unexpected:
        raise GateError(
            f"{label} field set mismatch: missing={missing!r}, unexpected={unexpected!r}"
        )


def validate_source_snapshot(
    root: Path,
    gr00t: Path,
    fast_revision: str,
    gr00t_revision: str,
    fast_tree_sha256: str,
    gr00t_tree_sha256: str,
    actual_inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate the builder's internal manifest and external sealed evidence."""

    expected_name = f"fast_roborender_{fast_revision[:7]}_{gr00t_revision[:7]}_sealed"
    if root.name != expected_name:
        raise GateError(
            f"sealed source snapshot name mismatch: {root.name!r} != {expected_name!r}"
        )
    if gr00t != root / "GR00T-VisualSim2Real":
        raise GateError("GR00T source is not the exact nested sealed checkout")

    internal_path = require_file(
        root / ".source_snapshot_manifest.json", "source snapshot internal manifest"
    )
    internal = load_json(internal_path)
    _require_exact_keys(
        internal,
        {
            "schema",
            "status",
            "snapshot_name",
            "fast_roborender_revision",
            "gr00t_revision",
            "fast_roborender_index_tree",
            "gr00t_index_tree",
            "export_method",
            "lfs_file_count",
            "lfs_total_bytes",
            "tracked_only",
            "lfs_pointer_oids_verified",
            "symlinks_rejected",
            "bytecode_rejected",
        },
        "source snapshot internal manifest",
    )
    expected_internal = {
        "schema": "fast_roborender.hsg_source_snapshot.v2",
        "status": "sealed_tracked_hydrated_sources",
        "snapshot_name": expected_name,
        "fast_roborender_revision": fast_revision,
        "gr00t_revision": gr00t_revision,
        "export_method": "stage0-git-blobs-plus-verified-lfs-v2",
        "tracked_only": True,
        "lfs_pointer_oids_verified": True,
        "symlinks_rejected": True,
        "bytecode_rejected": True,
    }
    for key, expected in expected_internal.items():
        if internal.get(key) != expected or type(internal.get(key)) is not type(expected):
            raise GateError(f"source snapshot internal manifest {key} mismatch")
    for key in ("fast_roborender_index_tree", "gr00t_index_tree"):
        if REVISION_RE.fullmatch(str(internal.get(key, ""))) is None:
            raise GateError(f"source snapshot internal manifest {key} is malformed")
    for key in ("lfs_file_count", "lfs_total_bytes"):
        value = internal.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GateError(f"source snapshot internal manifest {key} is invalid")

    attestation_path = require_file(
        root.parent / f"{root.name}.snapshot_attestation.json",
        "source snapshot attestation",
    )
    inventory_path = require_file(
        root.parent / f"{root.name}.inventory.json", "source snapshot inventory"
    )
    for path, label in (
        (attestation_path, "source snapshot attestation"),
        (inventory_path, "source snapshot inventory"),
    ):
        if path.stat().st_mode & 0o222:
            raise GateError(f"{label} must be read-only: {path}")

    inventory = load_json(inventory_path)
    _require_exact_keys(
        inventory,
        {"schema", "file_count", "total_file_bytes", "files"},
        "source snapshot inventory",
    )
    if inventory.get("schema") != "fast_roborender.hsg_source_inventory.v1":
        raise GateError("source snapshot inventory schema mismatch")
    records = inventory.get("files")
    if not isinstance(records, list):
        raise GateError("source snapshot inventory files must be a list")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise GateError(f"source snapshot inventory record {index} is malformed")
        _require_exact_keys(
            record, {"path", "mode", "bytes", "sha256"},
            f"source snapshot inventory record {index}",
        )
        relative = record.get("path")
        relative_path = Path(relative) if isinstance(relative, str) else Path("/")
        if (
            not isinstance(relative, str)
            or not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
        ):
            raise GateError(f"source snapshot inventory path {relative!r} is unsafe")
        if record.get("mode") not in {"0o444", "0o555"}:
            raise GateError(f"source snapshot inventory mode is invalid for {relative}")
        size = record.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise GateError(f"source snapshot inventory size is invalid for {relative}")
        if SHA256_RE.fullmatch(str(record.get("sha256", ""))) is None:
            raise GateError(f"source snapshot inventory hash is invalid for {relative}")
    paths = [record["path"] for record in records]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise GateError("source snapshot inventory paths are not sorted and unique")
    if inventory.get("file_count") != len(records):
        raise GateError("source snapshot inventory file_count mismatch")
    if inventory.get("total_file_bytes") != sum(record["bytes"] for record in records):
        raise GateError("source snapshot inventory total_file_bytes mismatch")
    if actual_inventory != records:
        raise GateError("source snapshot inventory does not match sealed source bytes")

    inventory_sha256 = sha256_file(inventory_path)
    attestation = load_json(attestation_path)
    _require_exact_keys(
        attestation,
        {
            "schema",
            "status",
            "snapshot_name",
            "fast_roborender_revision",
            "gr00t_revision",
            "fast_roborender_index_tree",
            "gr00t_index_tree",
            "fast_roborender_source_tree_sha256",
            "gr00t_source_tree_sha256",
            "inventory_sha256",
            "deterministic_archive_sha256",
            "file_count",
            "total_file_bytes",
        },
        "source snapshot attestation",
    )
    expected_attestation = {
        "schema": "fast_roborender.hsg_source_snapshot_attestation.v2",
        "status": "PASS",
        "snapshot_name": expected_name,
        "fast_roborender_revision": fast_revision,
        "gr00t_revision": gr00t_revision,
        "fast_roborender_index_tree": internal["fast_roborender_index_tree"],
        "gr00t_index_tree": internal["gr00t_index_tree"],
        "fast_roborender_source_tree_sha256": fast_tree_sha256,
        "gr00t_source_tree_sha256": gr00t_tree_sha256,
        "inventory_sha256": inventory_sha256,
        "file_count": len(records),
        "total_file_bytes": inventory["total_file_bytes"],
    }
    for key, expected in expected_attestation.items():
        if attestation.get(key) != expected or type(attestation.get(key)) is not type(expected):
            raise GateError(f"source snapshot attestation {key} mismatch")
    if SHA256_RE.fullmatch(str(attestation.get("deterministic_archive_sha256", ""))) is None:
        raise GateError("source snapshot deterministic archive hash is malformed")
    return {
        "source_snapshot_manifest_sha256": sha256_file(internal_path),
        "source_snapshot_attestation": str(attestation_path),
        "source_snapshot_attestation_sha256": sha256_file(attestation_path),
        "source_snapshot_inventory": str(inventory_path),
        "source_snapshot_inventory_sha256": inventory_sha256,
        "source_snapshot_deterministic_archive_sha256": attestation[
            "deterministic_archive_sha256"
        ],
    }


def validate_foundation(model_root: Path, deployment: dict[str, Any]) -> list[dict[str, Any]]:
    foundation = deployment.get("foundation")
    if not isinstance(foundation, dict):
        raise GateError("deployment foundation contract is missing")
    required = foundation.get("required_files")
    if not isinstance(required, list) or not required:
        raise GateError("deployment foundation required_files is empty")
    records = []
    for entry in required:
        if not isinstance(entry, dict):
            raise GateError("malformed foundation file record")
        relative = entry.get("relative_path")
        expected_bytes = entry.get("bytes")
        expected_hash = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_bytes, int):
            raise GateError("malformed foundation path/size record")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise GateError(f"foundation path escapes model root: {relative!r}")
        path = require_file(model_root / relative, f"foundation file {relative}")
        if not path.is_relative_to(model_root):
            raise GateError(f"foundation path escapes model root: {relative!r}")
        if path.stat().st_size != expected_bytes:
            raise GateError(f"foundation size mismatch for {relative}")
        actual = require_hash(path, str(expected_hash), f"foundation file {relative}")
        records.append(
            {
                "relative_path": relative_path.as_posix(),
                "path": str(path),
                "bytes": expected_bytes,
                "sha256": actual,
            }
        )
    return records


def validate_anchor_pool(path: Path, minimum: int) -> dict[str, Any]:
    path = require_dir(path, "anchor pool")
    anchors = sorted(path.glob("*.npz"))
    if len(anchors) < minimum:
        raise GateError(f"anchor pool contains {len(anchors)} files; expected at least {minimum}")
    required = {f"gen__{view}" for view in VIEWS}
    inventory = []
    for anchor in anchors:
        if anchor.is_symlink():
            raise GateError(f"anchor must not be a symlink: {anchor}")
        try:
            with zipfile.ZipFile(anchor) as archive:
                keys = {Path(name).stem for name in archive.namelist() if name.endswith(".npy")}
        except (OSError, zipfile.BadZipFile) as exc:
            raise GateError(f"invalid anchor {anchor}: {exc}") from exc
        missing = sorted(required - keys)
        if missing:
            raise GateError(f"invalid anchor {anchor}: missing={missing}")
        inventory.append(
            {
                "name": anchor.name,
                "bytes": anchor.stat().st_size,
                "sha256": sha256_file(anchor),
            }
        )
    return {
        "count": len(anchors),
        "inventory_sha256": canonical_sha256(inventory),
        "inventory": inventory,
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    root = require_dir(args.root, "fast_roborender source")
    gr00t = require_dir(args.gr00t_root, "GR00T source")
    fast_revision = require_revision_marker(root, args.fast_revision, "fast_roborender")
    gr00t_revision = require_revision_marker(gr00t, args.gr00t_revision, "GR00T")
    fast_tree_hash, fast_inventory = source_tree_evidence(root)
    gr00t_tree_hash = source_tree_sha256(gr00t)
    snapshot = {
        "source_snapshot_manifest_sha256": "NONE",
        "source_snapshot_attestation": "NONE",
        "source_snapshot_attestation_sha256": "NONE",
        "source_snapshot_inventory": "NONE",
        "source_snapshot_inventory_sha256": "NONE",
        "source_snapshot_deterministic_archive_sha256": "NONE",
    }
    if args.profile in SEALED_SOURCE_PROFILES:
        snapshot = validate_source_snapshot(
            root,
            gr00t,
            fast_revision,
            gr00t_revision,
            fast_tree_hash,
            gr00t_tree_hash,
            fast_inventory,
        )
    renderer_hash = require_hash(args.renderer_container, args.renderer_sha256, "renderer container")
    trainer_hash = require_hash(args.trainer_container, args.trainer_sha256, "trainer container")

    image_root = require_dir(args.image_root, "Image3F package")
    deployment_path = require_file(image_root / "config/deployment.json", "deployment manifest")
    deployment_hash = sha256_file(deployment_path)
    deployment = load_json(deployment_path)
    if deployment.get("release_id") != args.release_id:
        raise GateError("Image3F release_id mismatch")
    adapter = deployment.get("adapter")
    generation = deployment.get("generation_contract")
    acceleration = deployment.get("acceleration")
    if not isinstance(adapter, dict) or adapter.get("sha256") != args.adapter_sha256:
        raise GateError("deployment adapter identity mismatch")
    if not isinstance(generation, dict) or generation.get("steps") != 5:
        raise GateError("deployment must use the certified five-step schedule")
    if not isinstance(acceleration, dict):
        raise GateError("deployment acceleration contract is missing")
    adapter_path = image_root / str(adapter.get("path", ""))
    adapter_hash = require_hash(adapter_path, args.adapter_sha256, "Image3F adapter")

    model_root = require_dir(args.model_root, "foundation model root")
    foundation = validate_foundation(model_root, deployment)
    generated_root = require_dir(args.generated_root, "generated accelerator root")
    manifest_path = require_file(
        generated_root / "engines/current/engine_build_manifest.json", "SM100 engine manifest"
    )
    engine_manifest_hash = sha256_file(manifest_path)
    manifest = load_json(manifest_path)
    expected_engine = {
        "status": "built_microbenchmarked_and_correctness_checked",
        "gpu": "NVIDIA GB200",
        "compute_capability": "10.0",
        "precision": "bf16_fp16",
        "adapter_sha256": args.adapter_sha256,
    }
    for key, expected in expected_engine.items():
        if manifest.get(key) != expected:
            raise GateError(f"SM100 engine manifest {key} mismatch")
    graph = manifest.get("cuda_graph")
    if not isinstance(graph, dict) or graph.get("enabled_for_runtime") is not False:
        raise GateError("SM100 CUDA graph must remain disabled until separately certified")
    engine = manifest.get("engine")
    if not isinstance(engine, dict):
        raise GateError("SM100 engine record is missing")
    engine_path = manifest_path.parent / str(engine.get("file", ""))
    engine_hash = require_hash(engine_path, str(engine.get("sha256", "")), "SM100 TensorRT engine")

    teacher_hash = require_hash(args.teacher, args.teacher_sha256, "expert labeler")
    anchor_pool = validate_anchor_pool(args.anchors, args.anchor_min_files)
    foundation_contract = [
        {
            "relative_path": record["relative_path"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
        for record in foundation
    ]
    asset_contract = {
        "schema": "fast_roborender.hsg_immutable_assets.v1",
        "deployment_manifest_sha256": deployment_hash,
        "adapter_sha256": adapter_hash,
        "foundation": foundation_contract,
        "engine_manifest_sha256": engine_manifest_hash,
        "engine_sha256": engine_hash,
        "teacher_sha256": teacher_hash,
        "anchor_count": anchor_pool["count"],
        "anchor_inventory_sha256": anchor_pool["inventory_sha256"],
    }
    asset_contract_hash = canonical_sha256(asset_contract)
    report = {
        "schema": "fast_roborender.hsg_gb200_preflight.v1",
        "status": "validated",
        "profile": args.profile,
        "fast_roborender_revision": fast_revision,
        "fast_roborender_source_tree_sha256": fast_tree_hash,
        "gr00t_revision": gr00t_revision,
        "gr00t_source_tree_sha256": gr00t_tree_hash,
        **snapshot,
        "renderer_container_sha256": renderer_hash,
        "trainer_container_sha256": trainer_hash,
        "release_id": args.release_id,
        "adapter_sha256": adapter_hash,
        "deployment_manifest_sha256": deployment_hash,
        "foundation_contract_sha256": canonical_sha256(foundation_contract),
        "engine_manifest_sha256": engine_manifest_hash,
        "engine_sha256": engine_hash,
        "teacher_sha256": teacher_hash,
        "anchor_count": anchor_pool["count"],
        "anchor_inventory_sha256": anchor_pool["inventory_sha256"],
        "asset_contract": asset_contract,
        "asset_contract_sha256": asset_contract_hash,
        "foundation": foundation,
    }
    return report


def validate_finite_marker(path: Path, world_size: int) -> dict[str, Any]:
    marker = load_json(require_file(path, "finite BC-loss marker"))
    expected = {
        "schema": "fast_roborender.dagger_bc_loss.v1",
        "update": 1,
        "world_size": world_size,
        "finite_verified": True,
        "nonnegative_verified": True,
    }
    for key, value in expected.items():
        if marker.get(key) != value or type(marker.get(key)) is not type(value):
            raise GateError(f"finite BC-loss marker {key} mismatch")
    loss = marker.get("loss")
    if isinstance(loss, bool) or not isinstance(loss, (int, float)):
        raise GateError("finite BC loss is not numeric")
    if not math.isfinite(float(loss)) or float(loss) < 0:
        raise GateError("finite BC loss is invalid")
    return marker


def validate_ddp_marker(path: Path, world_size: int) -> dict[str, Any]:
    marker = load_json(require_file(path, "DDP synchronization marker"))
    expected = {
        "schema": "fast_roborender.ddp_policy_sync.v1",
        "verified": True,
        "optimizer_update": 1,
        "world_size": world_size,
    }
    for key, value in expected.items():
        if marker.get(key) != value or type(marker.get(key)) is not type(value):
            raise GateError(f"DDP synchronization marker {key} mismatch")
    if not isinstance(marker.get("policy_parameter_count"), int) or marker["policy_parameter_count"] <= 0:
        raise GateError("DDP marker policy_parameter_count is invalid")
    if SHA256_RE.fullmatch(str(marker.get("policy_parameters_sha256", ""))) is None:
        raise GateError("DDP marker parameter hash is invalid")
    return marker


def require_telemetry(run_dir: Path) -> dict[str, str]:
    patterns = {
        "scalar": "telemetry/scalars/*.json",
        "image": "telemetry/images/*.png",
        "video": "telemetry/videos/*.mp4",
    }
    result = {}
    for label, pattern in patterns.items():
        candidates = sorted(path for path in run_dir.glob(pattern) if path.stat().st_size > 0)
        if not candidates:
            raise GateError(f"required {label} telemetry is missing under {run_dir}")
        result[label] = str(candidates[-1].resolve())
    return result


def validate_policy_change(
    path: Path, initial: Path, final: Path, final_step: int
) -> dict[str, Any]:
    marker = load_json(require_file(path, "policy-state change marker"))
    if marker.get("schema") != "fast_roborender.checkpoint_policy_change.v1":
        raise GateError("policy-state change marker schema mismatch")
    if marker.get("status") != "PASS":
        raise GateError("policy-state change marker is not passing")
    records = marker.get("initial"), marker.get("final")
    for label, record, checkpoint, step in (
        ("initial", records[0], initial, 0),
        ("final", records[1], final, final_step),
    ):
        if not isinstance(record, dict):
            raise GateError(f"policy-state change {label} record is malformed")
        expected = {
            "path": str(checkpoint.resolve()),
            "checkpoint_step": step,
            "checkpoint_sha256": sha256_file(checkpoint),
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise GateError(f"policy-state change {label} {key} mismatch")
        if SHA256_RE.fullmatch(str(record.get("policy_sha256", ""))) is None:
            raise GateError(f"policy-state change {label} policy digest is malformed")
        count = record.get("policy_tensor_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise GateError(f"policy-state change {label} tensor count is invalid")
    if records[0]["policy_sha256"] == records[1]["policy_sha256"]:
        raise GateError("final policy tensors are identical to the fresh baseline")
    return marker


def validate_renderer_placement(path: Path, expected_renderers: int) -> dict[str, Any]:
    marker = load_json(require_file(path, "renderer placement attestation"))
    expected = {
        "schema": "fast_roborender.renderer_placement_attestation.v1",
        "status": "PASS",
        "renderer_count": expected_renderers,
        "unique_endpoint_count": expected_renderers,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise GateError(f"renderer placement {key} mismatch")
    expected_counts = marker.get("expected_host_counts")
    actual_counts = marker.get("actual_host_counts")
    if not isinstance(expected_counts, dict) or actual_counts != expected_counts:
        raise GateError("renderer placement host histogram mismatch")
    if any(
        not isinstance(host, str)
        or not host
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        for host, count in expected_counts.items()
    ):
        raise GateError("renderer placement host histogram is malformed")
    if sum(expected_counts.values()) != expected_renderers:
        raise GateError("renderer placement host histogram count mismatch")
    return marker


def validate_wandb_evidence(
    run_dir: Path, expected_email: str, expected_entity: str
) -> dict[str, Any]:
    drain_path = run_dir / "validation/wandb_media_drain.json"
    readback_path = run_dir / "validation/wandb_training_readback.json"
    drain = load_json(require_file(drain_path, "W&B media drain marker"))
    if drain.get("schema") != "fast_roborender.wandb_media_drain.v1" or drain.get(
        "status"
    ) != "PASS":
        raise GateError("W&B media drain marker is not passing")
    kinds = drain.get("artifact_kinds")
    if not isinstance(kinds, dict) or any(
        isinstance(kinds.get(kind), bool)
        or not isinstance(kinds.get(kind), int)
        or kinds[kind] < 1
        for kind in ("scalar", "image", "video")
    ):
        raise GateError("W&B media drain lacks scalar/image/video evidence")
    readback = load_json(require_file(readback_path, "W&B remote readback marker"))
    expected = {
        "schema": "fast_roborender.wandb_training_readback.v1",
        "status": "PASS",
        "account_email": expected_email,
        "entity": expected_entity,
        "personal": True,
        "sidecar_run_id": drain.get("wandb_run_id"),
    }
    for key, value in expected.items():
        if readback.get(key) != value:
            raise GateError(f"W&B remote readback {key} mismatch")
    required_trainer = set(readback.get("required_trainer_keys", []))
    observed_trainer = set(readback.get("observed_trainer_keys", []))
    required_sidecar = set(readback.get("required_sidecar_keys", []))
    observed_sidecar = set(readback.get("observed_sidecar_keys", []))
    if "production/dagger_bc_loss" not in required_trainer or not required_trainer <= observed_trainer:
        raise GateError("W&B trainer scalar readback is incomplete")
    if not required_sidecar or not required_sidecar <= observed_sidecar:
        raise GateError("W&B sidecar scalar/image/video readback is incomplete")
    return {
        "drain_path": str(drain_path.resolve()),
        "drain_sha256": sha256_file(drain_path),
        "readback_path": str(readback_path.resolve()),
        "readback_sha256": sha256_file(readback_path),
        "trainer_run_id": readback.get("trainer_run_id"),
        "sidecar_run_id": readback.get("sidecar_run_id"),
    }


def postrun(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = require_dir(args.run_dir, "training run directory")
    initial = require_file(run_dir / "initial_policy.pt", "initial policy")
    contracts = {
        "minimal_e2e": (1, 2, 1, 1, 1),
        "resume_minimal": (1, 2, 1, 2, 1),
        "calibration_3view": (1, 2, 1, 1, 1),
        "smoke_32x300": (32, 300, 1, 1, 7),
        "smoke_256x300": (256, 300, 8, 1, 52),
        "production_256x300": (256, 300, 8, 500, 52),
    }
    try:
        (
            expected_envs,
            expected_rollout,
            expected_world,
            expected_final_step,
            expected_renderers,
        ) = contracts[args.profile]
    except KeyError as exc:
        raise GateError(f"unsupported postrun profile {args.profile!r}") from exc
    actual_topology = (args.global_envs, args.rollout_steps, args.world_size)
    expected_topology = (expected_envs, expected_rollout, expected_world)
    if actual_topology != expected_topology:
        raise GateError(
            f"{args.profile} topology changed: {actual_topology} != {expected_topology}"
        )
    requested_final_step = getattr(args, "final_step", None)
    final_step = expected_final_step if requested_final_step is None else requested_final_step
    if final_step != expected_final_step:
        raise GateError(
            f"{args.profile} final step changed: {final_step} != {expected_final_step}"
        )
    final = require_file(
        run_dir / f"model_step_{final_step:06d}.pt", "final policy"
    )
    finite_path = run_dir / "validation/dagger_bc_loss_step_000000001.json"
    finite = validate_finite_marker(finite_path, args.world_size)
    ddp_path: Path | None = None
    ddp_hash = ""
    ddp_marker: dict[str, Any] | None = None
    if args.world_size > 1:
        ddp_path = run_dir / "validation/ddp_policy_sync_step_000000001.json"
        ddp_marker = validate_ddp_marker(ddp_path, args.world_size)
        ddp_hash = sha256_file(ddp_path)
    telemetry = require_telemetry(run_dir)
    for value, label in (
        (args.fast_source_tree_sha256, "fast_roborender source-tree"),
        (args.gr00t_source_tree_sha256, "GR00T source-tree"),
        (args.asset_contract_sha256, "immutable asset-contract"),
    ):
        if SHA256_RE.fullmatch(value) is None:
            raise GateError(f"{label} SHA-256 is malformed")
    sealed_profile = args.profile in SEALED_SOURCE_PROFILES
    for value, label in (
        (args.source_snapshot_attestation_sha256, "source snapshot attestation"),
        (args.source_snapshot_inventory_sha256, "source snapshot inventory"),
    ):
        if sealed_profile and SHA256_RE.fullmatch(value) is None:
            raise GateError(f"{label} SHA-256 is malformed")
        if not sealed_profile and value != "NONE":
            raise GateError(f"non-sealed profile unexpectedly received {label}")
    placement_path = run_dir / "validation/renderer_placement.json"
    validate_renderer_placement(placement_path, expected_renderers)
    wandb = None
    if args.profile in {"smoke_32x300", "smoke_256x300", "production_256x300"}:
        if not args.wandb_expected_email or not args.wandb_entity:
            raise GateError("W&B identity is required for this profile")
        wandb = validate_wandb_evidence(
            run_dir, args.wandb_expected_email, args.wandb_entity
        )
    policy_change_path = run_dir / "validation/checkpoint_policy_change.json"
    policy_change = validate_policy_change(policy_change_path, initial, final, final_step)
    initial_checkpoint_hash = sha256_file(initial)
    final_checkpoint_hash = sha256_file(final)
    initial_policy_hash = policy_change["initial"]["policy_sha256"]
    final_policy_hash = policy_change["final"]["policy_sha256"]
    # The trainer emits the DDP attestation at its first optimizer update.  A
    # one-update smoke must therefore persist exactly that state; a 500-update
    # production run is instead bound by the checkpoint policy-change gate.
    if (
        ddp_marker is not None
        and final_step == 1
        and ddp_marker["policy_parameters_sha256"] != final_policy_hash
    ):
        raise GateError(
            "DDP synchronized policy digest does not match the persisted final policy"
        )

    report = {
        "schema": "fast_roborender.hsg_gb200_smoke_result.v1",
        "status": "validated",
        "profile": args.profile,
        "global_envs": args.global_envs,
        "rollout_steps": args.rollout_steps,
        "trainer_world_size": args.world_size,
        "fast_roborender_revision": args.fast_revision,
        "fast_roborender_source_tree_sha256": args.fast_source_tree_sha256,
        "gr00t_revision": args.gr00t_revision,
        "gr00t_source_tree_sha256": args.gr00t_source_tree_sha256,
        "source_snapshot_attestation_sha256": args.source_snapshot_attestation_sha256,
        "source_snapshot_inventory_sha256": args.source_snapshot_inventory_sha256,
        "immutable_asset_contract_sha256": args.asset_contract_sha256,
        "renderer_placement_sha256": sha256_file(placement_path),
        "initial_checkpoint_sha256": initial_checkpoint_hash,
        "initial_policy_sha256": initial_policy_hash,
        "final_checkpoint_step": final_step,
        "final_checkpoint_sha256": final_checkpoint_hash,
        "final_policy_sha256": final_policy_hash,
        "policy_change_marker_sha256": sha256_file(policy_change_path),
        "finite_bc_loss": float(finite["loss"]),
        "finite_bc_loss_marker_sha256": sha256_file(finite_path),
        "ddp_marker_sha256": ddp_hash,
        "telemetry": telemetry,
    }
    if wandb is not None:
        report["wandb"] = wandb
    if args.profile == "calibration_3view":
        panels = [
            require_file(run_dir / "render_dumps/step000_env0.png", "calibration panel 0"),
            require_file(run_dir / "render_dumps/step001_env0.png", "calibration panel 1"),
        ]
        technical = {
            "schema": "fast_roborender.camera_calibration.v1",
            "status": "awaiting_visual_acceptance",
            "fast_roborender_revision": args.fast_revision,
            "fast_roborender_source_tree_sha256": args.fast_source_tree_sha256,
            "gr00t_revision": args.gr00t_revision,
            "gr00t_source_tree_sha256": args.gr00t_source_tree_sha256,
            "source_snapshot_attestation_sha256": args.source_snapshot_attestation_sha256,
            "source_snapshot_inventory_sha256": args.source_snapshot_inventory_sha256,
            "immutable_asset_contract_sha256": args.asset_contract_sha256,
            "policy_views": list(VIEWS),
            "columns": list(VIEWS),
            "rows": ["depth_control", "simulator_rgb", "generated_rgb"],
            "tile_shape_hwc": [240, 416, 3],
            "panel_shape_hwc": [720, 1248, 3],
            "panels": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in panels
            ],
        }
        technical_path = run_dir / "validation/camera_calibration_manifest.json"
        technical_path.parent.mkdir(parents=True, exist_ok=True)
        technical_path.write_text(
            json.dumps(technical, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report["camera_calibration_manifest"] = str(technical_path.resolve())
        report["camera_calibration_manifest_sha256"] = sha256_file(technical_path)
    output = run_dir / "hsg_gb200_smoke_result.json"
    if args.profile in {"smoke_32x300", "smoke_256x300"}:
        manifest = run_dir / "profile_validation_manifest.txt"
        fields = {
            "manifest_schema": "fast_roborender.hsg_profile_validation.v2",
            "run_profile": args.profile,
            "fast_roborender_revision": args.fast_revision,
            "fast_roborender_source_tree_sha256": args.fast_source_tree_sha256,
            "gr00t_revision": args.gr00t_revision,
            "gr00t_source_tree_sha256": args.gr00t_source_tree_sha256,
            "source_snapshot_attestation_sha256": (
                args.source_snapshot_attestation_sha256
            ),
            "source_snapshot_inventory_sha256": (
                args.source_snapshot_inventory_sha256
            ),
            "immutable_asset_contract_sha256": args.asset_contract_sha256,
            "global_envs": str(args.global_envs),
            "rollout_steps": str(args.rollout_steps),
            "trainer_nodes": str(args.world_size),
            "policy_views": ",".join(VIEWS),
            "policy_view_count": "3",
            "policy_view_layout": "vertical",
            "policy_image_shape": "324x192x3",
            "initial_policy_checkpoint": str(initial),
            "initial_checkpoint_sha256": report["initial_checkpoint_sha256"],
            "initial_policy_state_sha256": report["initial_policy_sha256"],
            "final_checkpoint": str(final),
            "final_checkpoint_sha256": report["final_checkpoint_sha256"],
            "final_policy_state_sha256": report["final_policy_sha256"],
            "policy_change_marker": str(policy_change_path.resolve()),
            "policy_change_marker_sha256": report["policy_change_marker_sha256"],
            "renderer_placement": str(placement_path.resolve()),
            "renderer_placement_sha256": report["renderer_placement_sha256"],
            "wandb_media_drain": wandb["drain_path"],
            "wandb_media_drain_sha256": wandb["drain_sha256"],
            "wandb_remote_readback": wandb["readback_path"],
            "wandb_remote_readback_sha256": wandb["readback_sha256"],
            "wandb_trainer_run_id": str(wandb["trainer_run_id"]),
            "wandb_sidecar_run_id": str(wandb["sidecar_run_id"]),
            "finite_bc_loss_marker": str(finite_path.resolve()),
            "finite_bc_loss_marker_sha256": report["finite_bc_loss_marker_sha256"],
            "ddp_sync_marker": str(ddp_path.resolve()) if ddp_path else "",
            "ddp_sync_marker_sha256": ddp_hash,
            "camera_calibration_manifest": "",
            "camera_calibration_manifest_sha256": "",
            "rollout_iterations": "1",
            "status": "validated",
        }
        manifest.write_text("".join(f"{key}={value}\n" for key, value in fields.items()), encoding="utf-8")
        report["profile_validation_manifest"] = str(manifest.resolve())
        report["profile_validation_manifest_sha256"] = sha256_file(manifest)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--profile", required=True)
    pre.add_argument("--root", type=Path, required=True)
    pre.add_argument("--gr00t-root", type=Path, required=True)
    pre.add_argument("--fast-revision", required=True)
    pre.add_argument("--gr00t-revision", required=True)
    pre.add_argument("--renderer-container", type=Path, required=True)
    pre.add_argument("--renderer-sha256", required=True)
    pre.add_argument("--trainer-container", type=Path, required=True)
    pre.add_argument("--trainer-sha256", required=True)
    pre.add_argument("--image-root", type=Path, required=True)
    pre.add_argument("--model-root", type=Path, required=True)
    pre.add_argument("--generated-root", type=Path, required=True)
    pre.add_argument("--anchors", type=Path, required=True)
    pre.add_argument("--anchor-min-files", type=int, default=2048)
    pre.add_argument("--teacher", type=Path, required=True)
    pre.add_argument("--teacher-sha256", required=True)
    pre.add_argument("--release-id", required=True)
    pre.add_argument("--adapter-sha256", required=True)

    post = sub.add_parser("postrun")
    post.add_argument("--profile", required=True)
    post.add_argument("--run-dir", type=Path, required=True)
    post.add_argument("--global-envs", type=int, required=True)
    post.add_argument("--rollout-steps", type=int, required=True)
    post.add_argument("--final-step", type=int)
    post.add_argument("--world-size", type=int, required=True)
    post.add_argument("--fast-revision", required=True)
    post.add_argument("--gr00t-revision", required=True)
    post.add_argument("--fast-source-tree-sha256", required=True)
    post.add_argument("--gr00t-source-tree-sha256", required=True)
    post.add_argument("--source-snapshot-attestation-sha256", required=True)
    post.add_argument("--source-snapshot-inventory-sha256", required=True)
    post.add_argument("--asset-contract-sha256", required=True)
    post.add_argument("--wandb-expected-email")
    post.add_argument("--wandb-entity")
    tree = sub.add_parser("tree-hash")
    tree.add_argument("--root", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        if args.command == "preflight":
            report = preflight(args)
        elif args.command == "postrun":
            report = postrun(args)
        else:
            print(source_tree_sha256(args.root))
            return
    except (GateError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
