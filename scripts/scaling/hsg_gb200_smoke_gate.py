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


class GateError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(root: Path) -> str:
    """Hash a staged source tree without relying on Git metadata."""

    root = require_dir(root, "source tree")
    digest = hashlib.sha256()
    candidates = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.is_dir() or path.suffix in {".pyc", ".pyo"}:
            continue
        candidates.append((relative, path))
    for relative, path in sorted(candidates, key=lambda item: item[0].as_posix()):
        encoded = relative.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update((path.lstat().st_mode & 0o777).to_bytes(2, "big"))
        if path.is_symlink():
            payload = os.readlink(path).encode("utf-8")
            digest.update(b"L" + len(payload).to_bytes(8, "big") + payload)
        elif path.is_file():
            digest.update(b"F")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
        else:
            raise GateError(f"unsupported source-tree entry: {path}")
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise GateError(f"{path} must contain one JSON object")
    return result


def require_file(path: Path, label: str) -> Path:
    path = path.resolve(strict=True)
    if not path.is_file() or path.stat().st_size <= 0:
        raise GateError(f"{label} is not a non-empty regular file: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
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
        path = require_file(model_root / relative, f"foundation file {relative}")
        if path.stat().st_size != expected_bytes:
            raise GateError(f"foundation size mismatch for {relative}")
        actual = require_hash(path, str(expected_hash), f"foundation file {relative}")
        records.append({"path": str(path), "bytes": expected_bytes, "sha256": actual})
    return records


def validate_anchor_pool(path: Path, minimum: int) -> int:
    path = require_dir(path, "anchor pool")
    anchors = sorted(path.glob("*.npz"))
    if len(anchors) < minimum:
        raise GateError(f"anchor pool contains {len(anchors)} files; expected at least {minimum}")
    required = {f"gen__{view}" for view in VIEWS}
    for anchor in anchors:
        try:
            with zipfile.ZipFile(anchor) as archive:
                keys = {Path(name).stem for name in archive.namelist() if name.endswith(".npy")}
        except (OSError, zipfile.BadZipFile) as exc:
            raise GateError(f"invalid anchor {anchor}: {exc}") from exc
        missing = sorted(required - keys)
        if missing:
            raise GateError(f"invalid anchor {anchor}: missing={missing}")
    return len(anchors)


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    root = require_dir(args.root, "fast_roborender source")
    gr00t = require_dir(args.gr00t_root, "GR00T source")
    fast_revision = require_revision_marker(root, args.fast_revision, "fast_roborender")
    gr00t_revision = require_revision_marker(gr00t, args.gr00t_revision, "GR00T")
    renderer_hash = require_hash(args.renderer_container, args.renderer_sha256, "renderer container")
    trainer_hash = require_hash(args.trainer_container, args.trainer_sha256, "trainer container")

    image_root = require_dir(args.image_root, "Image3F package")
    deployment_path = require_file(image_root / "config/deployment.json", "deployment manifest")
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
    anchor_count = validate_anchor_pool(args.anchors, args.anchor_min_files)
    report = {
        "schema": "fast_roborender.hsg_gb200_preflight.v1",
        "status": "validated",
        "profile": args.profile,
        "fast_roborender_revision": fast_revision,
        "gr00t_revision": gr00t_revision,
        "gr00t_source_tree_sha256": source_tree_sha256(gr00t),
        "renderer_container_sha256": renderer_hash,
        "trainer_container_sha256": trainer_hash,
        "release_id": args.release_id,
        "adapter_sha256": adapter_hash,
        "engine_sha256": engine_hash,
        "teacher_sha256": teacher_hash,
        "anchor_count": anchor_count,
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


def postrun(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = require_dir(args.run_dir, "training run directory")
    initial = require_file(run_dir / "initial_policy.pt", "initial policy")
    final_step = 2 if args.profile == "resume_minimal" else 1
    final = require_file(
        run_dir / f"model_step_{final_step:06d}.pt", "final policy"
    )
    finite_path = run_dir / "validation/dagger_bc_loss_step_000000001.json"
    finite = validate_finite_marker(finite_path, args.world_size)
    ddp_path: Path | None = None
    ddp_hash = ""
    if args.world_size > 1:
        ddp_path = run_dir / "validation/ddp_policy_sync_step_000000001.json"
        validate_ddp_marker(ddp_path, args.world_size)
        ddp_hash = sha256_file(ddp_path)
    telemetry = require_telemetry(run_dir)

    report = {
        "schema": "fast_roborender.hsg_gb200_smoke_result.v1",
        "status": "validated",
        "profile": args.profile,
        "global_envs": args.global_envs,
        "rollout_steps": args.rollout_steps,
        "trainer_world_size": args.world_size,
        "fast_roborender_revision": args.fast_revision,
        "gr00t_revision": args.gr00t_revision,
        "initial_policy_sha256": sha256_file(initial),
        "final_checkpoint_step": final_step,
        "final_policy_sha256": sha256_file(final),
        "finite_bc_loss": float(finite["loss"]),
        "finite_bc_loss_marker_sha256": sha256_file(finite_path),
        "ddp_marker_sha256": ddp_hash,
        "telemetry": telemetry,
    }
    if args.profile == "calibration_3view":
        panels = [
            require_file(run_dir / "render_dumps/step000_env0.png", "calibration panel 0"),
            require_file(run_dir / "render_dumps/step001_env0.png", "calibration panel 1"),
        ]
        technical = {
            "schema": "fast_roborender.camera_calibration.v1",
            "status": "awaiting_visual_acceptance",
            "fast_roborender_revision": args.fast_revision,
            "gr00t_revision": args.gr00t_revision,
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
            "run_profile": args.profile,
            "fast_roborender_revision": args.fast_revision,
            "gr00t_revision": args.gr00t_revision,
            "global_envs": str(args.global_envs),
            "rollout_steps": str(args.rollout_steps),
            "trainer_nodes": str(args.world_size),
            "policy_views": ",".join(VIEWS),
            "policy_view_count": "3",
            "policy_view_layout": "vertical",
            "policy_image_shape": "324x192x3",
            "initial_policy_checkpoint": str(initial),
            "initial_policy_sha256": report["initial_policy_sha256"],
            "final_checkpoint": str(final),
            "final_checkpoint_sha256": report["final_policy_sha256"],
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
    post.add_argument("--world-size", type=int, required=True)
    post.add_argument("--fast-revision", required=True)
    post.add_argument("--gr00t-revision", required=True)
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
